# NexGPU 4090 24GB 执行 Runbook（MathGPT-30M）

> 目标：单张 4090 24GB，wall-clock 上限 48 GPU 小时，按「六轮贪心消融 → 3×3 Scaling Law → 约 30M 正式训练」跑完并回填 `reports/FINAL_RESULTS.md`。所有结论必须来自真实日志，禁止虚构。

## 0. 角色与前置

- 本机（Windows）只做：改代码、打包、把 zip 传上 Pod、最后把 checkpoint 拉回。
- 训练全部在 NexGPU 的远程 Linux Pod 上执行。
- 已经修好的东西（本机改完再打包）：`scripts/prepare.py` / `overfit.py` / `generate.py` 补了 `sys.path`；`greedy_search.py` 的 `--eval_batches` 默认 8→64；`run_scaling.py` 的 eval_batches 8→32；`docs/48_gpu_hours_plan.md` 预算表已与 `AGENT_HANDOFF.md` 对齐。

## 1. 租卡与开模板

1. 登录 console.nexgpu.net → 市场选 **RTX 4090 24GB**（约 $0.54/卡时）。
2. 模板选 **PyTorch**（CUDA + PyTorch + 驱动预装）。
3. 开机后复制 **SSH 命令**（或直接用网页终端）。SSH 信息（host、port、账号、密码或 key）交给执行 Agent。

## 2. 传代码

本机 PowerShell（或 Git Bash）：

```bash
scp -P <PORT> mathgpt30m.zip root@<HOST>:/workspace/
```

远端：

```bash
cd /workspace && unzip mathgpt30m.zip && cd mathgpt30m
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 环境自检（跑 GPU 前必做）

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
df -h /workspace
python -m pytest -q
```

`is_bf16_supported()` 在 4090 上应为 True（Ada 支持 BF16，训练循环会自动走 `autocast(bfloat16)`）。pytest 不过就停下来修，不要上付费卡白跑。

写 `runs/environment.json`：

```bash
mkdir -p runs && python - <<'PY'
import json, platform, torch, subprocess
info = {
  "gpu": torch.cuda.get_device_name(0),
  "vram_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9, 1),
  "cuda": torch.version.cuda,
  "torch": torch.__version__,
  "bf16": torch.cuda.is_bf16_supported(),
  "python": platform.python_version(),
  "host": platform.node(),
}
json.dump(info, open("runs/environment.json","w"), indent=2)
print(json.dumps(info, indent=2))
PY
```

## 4. 数据准备（冻结 tokenizer / manifest / 验证集）

```bash
python scripts/prepare.py --tokenizer_docs 30000 --docs 200000
ls -lh data/processed/train.bin data/processed/val.bin
```

确认 `data/processed/train.bin`、`val.bin`、`manifest.json` 都存在且 `val.bin` 非空。**之后任何一步都不再改 tokenizer、manifest、验证集。**

`manifest.json` 里 `tokens.train` 就是单遍可用 token 数。正式 30M 训练若目标 6 亿 token（20× 参数量）而单遍不够，要么 `--docs` 加到 400000 重跑 prepare，要么接受多 epoch 并在报告里记录重复采样。

## 5. 六轮贪心消融

```bash
python scripts/greedy_search.py --dry-run          # 先审命令
python scripts/greedy_search.py --max_hours 6       # 正式跑
```

- 产物：`runs/greedy/decisions.json`（每轮 baseline、候选、参数量、val loss、是否晋升）、`runs/greedy/selected_architecture.json`。
- 这是单 seed 初筛。R1/R4/R5 会改变参数量，低 val loss ≠ 同参数下架构更好。
- **人工复核**（预算允许时）：对 `selected_architecture.json` 的赢家与落选最强候选，各用 3 个 seed 重跑比较中位 val loss，例：

```bash
python -m mathgpt.train --data data/processed --out runs/greedy/recheck/win_s43 \
  --steps 150 --batch 4 --accum 1 --warmup 10 --decay 30 --eval_every 50 --eval_batches 64 \
  --seed 43 --lr 3e-4 --dim 128 --layers 4 --heads 4 --kv_heads 2 --hidden 352 --vocab 32000 --seq 128
```

（把 `--kv_heads/--position/--norm_type/--ffn_type/--untie/--no_residual_scale` 换成选中值，seed 换 43/44/45。若证据不足，保留更简单的 baseline。）

## 6. Scaling Law（3×3 代理实验）

```bash
python scripts/run_scaling.py --dry-run
python scripts/run_scaling.py --max_hours 8
python -m mathgpt.scaling runs/scaling/points.jsonl --flops <按实测吞吐算出的FLOPs预算>
```

- 产物：`runs/scaling/points.jsonl`（9 个真实 N/D/L 点）+ 拟合 E/A/B/α/β + RMSE。
- 9 点拟合 5 参数很脆弱，报告里必须写残差与外推风险；不把拟合出的 N* 用来改正式 30M。
- `--flops` 不是 48 GPU 小时，是实际计算量估算（`C≈6ND`），按真实吞吐给，给不出就写明“无法稳定估计”。

## 7. 约 30M 正式训练

```bash
python scripts/scale_to_30m.py                    # 生成 configs/final_30m.json
python scripts/train_selected.py --dry-run
```

先测速定步数：`--batch 4 --accum 8 --seq 1024` 时每步 32768 token。目标 6 亿 token ≈ 18300 步。开一个 50–100 步小跑测 `tok_per_s`，反推 26 小时能跑多少步，据此定 `--steps`；超预算就降目标（不低于 ~3 亿 token），绝不满打满算静默超额。

```bash
tmux new -s train
python scripts/train_selected.py --steps 18300
# Ctrl+B 再 D 分离；tmux attach -t train 重连
```

- 产物：`runs/final_30m/last.pt`、`metrics.jsonl`、`model_config.json`。
- 抽查生成：

```bash
python scripts/generate.py --checkpoint runs/final_30m/last.pt --tokenizer data/tokenizer.json --prompt "Prove that" --tokens 80
python scripts/generate.py --checkpoint runs/final_30m/last.pt --tokenizer data/tokenizer.json --prompt "Prove that" --tokens 80 --no_cache   # 对照 KV cache 是否一致
```

## 8. 结果回收与停卡

本机：

```bash
scp -P <PORT> -r root@<HOST>:/workspace/mathgpt30m/runs ./runs
scp -P <PORT> root@<HOST>:/workspace/mathgpt30m/configs/final_30m.json ./configs/
```

确认 checkpoint 已落盘到本机后，**删除/停止 Pod 实例**（按小时计费，删了才停费）。

回填 `reports/FINAL_RESULTS.md`：设备、seed、真实参数量、训练 token 数、val loss、tokens/s、GPU 小时、每轮消融理由、失败实验、误差范围。未运行的条目一律写“未运行”。

## 附：预算切分（与 AGENT_HANDOFF.md 一致）

| 用途 | 上限小时 |
|---|---:|
| 部署 / 数据 / 正确性验证 | 4 |
| 六轮消融初筛及必要复核 | 6 |
| Scaling Law 代理实验 | 8 |
| 约 30M 正式预训练 | 26 |
| 评估、修复、缓冲 | 4 |
| 合计 | 48 |
