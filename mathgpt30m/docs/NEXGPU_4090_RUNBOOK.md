# NexGPU 4090 24GB 执行 Runbook（MathGPT-60M）

> **状态：已执行完毕（2026-10-25 → 10-26）。** 结果：59.6M 参数 / 6 亿 token / best val_loss 2.1825，详见 `../../results/FINAL_RESULTS.md` 与 `../../PROVENANCE.md`。第 7 节为 60M 正式训练的逐条命令与产物路径。
>
> 目标：单张 4090 24GB，wall-clock 上限 48 GPU 小时，按「六轮贪心消融 → 3×3 Scaling Law → 正式训练」跑完并出报告。所有数字必须来自远端日志。

## 0. 角色与前置

- 本机（Windows）只做：改代码、打包、把 zip 传上 Pod、最后把 checkpoint 拉回。
- 训练全部在 NexGPU 的远程 Linux Pod 上执行。
- 代码要点：`scripts/prepare.py` / `overfit.py` / `generate.py` 自带 `sys.path` 注入，可从任意目录调用；`greedy_search.py` 的 `--eval_batches` 默认 64，`run_scaling.py` 默认 32；预算表见 `48_gpu_hours_plan.md`。

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

`manifest.json` 里 `tokens.train` 是可用 token 池大小（本次 278,476,675）。采样器按随机起点取 batch，6 亿训练 token 会重复覆盖该池约 2.15 次，报告中按重复采样记录。

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
- 9 点拟合 5 参数很脆弱，报告里必须写残差与外推风险；不把拟合出的 N* 用来改正式模型规模（本次 N*=138.9M，最终仍按 60M 训练）。
- `--flops` 不是 48 GPU 小时，是实际计算量估算（`C≈6ND`），按真实吞吐给，给不出就写明“无法稳定估计”。

## 7. 60M 正式训练

```bash
python scripts/scale_to_30m.py --target 60000000 --out configs/final_60m.json   # 实际生成 59,612,640 参数
python scripts/train_selected.py --dry-run
```

先测速定步数：`--batch 4 --accum 8 --seq 1024` 时每步 32,768 token，6 亿 token 对应 18,311 步。60M 实测吞吐 4.1–4.7 万 tok/s，约 4.1–4.7 小时，低于 26 小时分配额。

```bash
tmux new-session -d -s train60
python -m mathgpt.train --data data/processed --out runs/final_60m \
  --vocab 32000 --dim 480 --layers 16 --heads 4 --kv_heads 2 --hidden 1440 \
  --seq 1024 --batch 4 --accum 8 --steps 18311 --warmup 915 --decay 1831 \
  --eval_every 200 --eval_batches 20 --save_every 8000 --log_every 100 \
  --seed 42 --lr 3e-4 --position rope --norm_type rms --ffn_type swiglu
```

**中断续训**：同一命令加 `--resume runs/final_60m/last.pt` 即从最近 checkpoint 接续（本次运行中发生 3 次实例停机，均由此接续）。
torch 2.14.1 对 `set_rng_state` 的张量类型与设备有要求，代码用 try/except 容错：rng 状态未恢复只影响严格复现，不影响训练。

- 产物：`runs/final_60m/last.pt`（715 MB）、`metrics.jsonl`、`metrics.png`。
- 结果：18,311 步 / 600,014,848 token / train_loss 1.8894 / val_loss 2.4153（best 2.1825 @18200）。
- 抽查生成：

```bash
python scripts/generate.py --checkpoint runs/final_60m/last.pt --tokenizer data/tokenizer.json --prompt "Theorem: For any prime number p," --tokens 60
python scripts/generate.py --checkpoint runs/final_60m/last.pt --tokenizer data/tokenizer.json --prompt "Solve the equation x^2" --tokens 60
```

## 8. 结果回收与停卡

本机：

```bash
scp -P <PORT> -r root@<HOST>:/workspace/mathgpt30m/runs ./runs
scp -P <PORT> root@<HOST>:/workspace/mathgpt30m/configs/final_60m.json ./configs/
```

确认 checkpoint 已落盘到本机后，**删除/停止 Pod 实例**（按小时计费，删了才停费）。

把结果整理进 `../../results/FINAL_RESULTS.md`：设备、seed、参数量、训练 token 数、val loss、tokens/s、GPU 小时、每轮消融理由、失败实验、误差范围。未运行的条目一律写"未运行"。

## 附：预算切分（与 AGENT_HANDOFF.md 一致）

| 用途 | 上限小时 |
|---|---:|
| 部署 / 数据 / 正确性验证 | 4 |
| 六轮消融初筛及必要复核 | 6 |
| Scaling Law 代理实验 | 8 |
| 60M 正式预训练（实际 4.67 h） | 26 |
| 评估、修复、缓冲 | 4 |
| 合计 | 48 |
