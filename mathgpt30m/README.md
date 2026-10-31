# MathGPT-60M — 顺序消融 → Scaling Law → 60M 预训练

从零预训练一个 59.6M 参数的数学文本 GPT。流程是：六轮顺序贪心消融定架构 → 9 点 Scaling Law 拟合 → 定规模正式预训练。全部计算在单张 RTX 4090 24GB 上完成，BF16。

| 项 | 值 |
|---|---|
| 参数量 | 59,612,640（dim 480 / 16 层 / 4 heads / GQA kv=2 / hidden 1440） |
| 训练 token | 600,014,848（18,311 步 × 32,768，seq 1024） |
| best val_loss | 2.1825（step 18200） |
| final val_loss | 2.4153 |
| train_loss | 1.889 |
| 最终架构 | GQA(kv=2) + RoPE + RMSNorm + SwiGLU + tied embedding + residual_scale |
| 硬件 / 时间 | RTX 4090 24GB，训练计算 4.67 h，全项目墙钟约 17 h |

交付物位置：

- 完整报告：`../results/FINAL_RESULTS.md`
- 运行凭证（硬件、时间线、逐条命令、sha256）：`../PROVENANCE.md`、`../results/provenance.json`
- 逐 step 日志：`../results/metrics.jsonl`（24,271 行）
- Scaling 观测点：`../results/points.jsonl`（9 点）
- 消融决策账本：`../results/greedy_s4{2,3,4}_decisions.json`
- 权重：`runs/final_60m/last.pt`（715 MB，未入库，sha256 见凭证文件）

## 实验设计

**消融**：六轮顺序贪心。第 r 轮的 baseline 为 `b_r`，候选集为只在第 r 个因素上做单点改动的配置。每个候选使用同一 tokenizer、同一训练/验证数据、同一 seed、同一 token 预算与学习率设置，比较验证交叉熵。候选相对 baseline 的验证 loss 下降 ≥ `min_delta=0.002` 才晋升为下一轮 baseline，否则保留原配置。六轮轮次为 kv_heads → position → norm_type → ffn_type → untie → residual_scale。执行配置：每轮 1000 步，warmup 100，decay 200，3 个 seed（42/43/44）。

**Scaling**：固定架构家族，取 3 档模型规模 × 3 档 token 预算，得 9 个 `(N, D, loss)` 观测。token 预算按模型参数量定倍率，每档模型分别训 10×/20×/40× 参数量的 token。拟合

```
L(N, D) = E + A*(N/1e6)^(-alpha) + B*(D/1e6)^(-beta)
```

**正式训练**：目标规模 60M，训练量 6 亿 token（10 token/参数）。AdamW(0.9, 0.95) + WSD 学习率（warmup 915 / stable / decay 1831，峰值 3e-4），梯度累积 8、全局 batch 32,768，BF16 autocast。

## 结论

- 消融：三项候选（MQA、LayerNorm、GELU）的 loss 优势在 3 seed × 1000 步下不可复现，属单 seed 噪声；`untie` 在 3 个 seed 上稳定更低但参数量翻倍。最终采用 baseline 架构，参数效率优先。
- Scaling：E=2.832、A=2.566、alpha=0.652、B=18.200、beta=1.010，RMSE=0.043；理论最优 N*=138.9M / D*=259.2M。拟合仅作参考，正式模型按 60M 训练。
- 60M 训 6 亿 token = 10 token/参数，低于 Chinchilla 的 20，模型欠训练，生成能模仿 LaTeX 与定理格式，语义不可靠。

## 阅读顺序

1. `AGENT_HANDOFF.md`：执行协议、验收标准、预算、命令链。
2. `docs/experiment_rationale.md`：每个选择的书本依据、基线、对照、风险。
3. `docs/cloud_gpu.md`：GPU Pod、SSH、持久化磁盘。
4. `docs/NEXGPU_4090_RUNBOOK.md`：本次使用的逐条 runbook。
5. `mathgpt/model.py`、`mathgpt/train.py`、`scripts/greedy_search.py`、`scripts/run_scaling.py`：核心代码。

## 目录

```text
mathgpt30m/
├── AGENT_HANDOFF.md                # 执行契约：协议、验收、预算、命令
├── README.md                       # 本文件
├── mathgpt/
│   ├── model.py                    # 统一 GPT 接口；MHA/GQA/MQA、RoPE/NoPE、RMS/LN、SwiGLU/GELU
│   ├── train.py                    # AdamW、BF16、WSD、梯度累积、日志、checkpoint、resume
│   ├── data.py                     # FineMath streaming、BPE、packing、memmap
│   ├── experiment.py               # 结果校验、晋升判定
│   └── scaling.py                  # 五参数 Scaling Law 拟合与理论 N*/D*
├── scripts/
│   ├── prepare.py                  # 数据准备
│   ├── greedy_search.py            # 六轮顺序贪心消融
│   ├── run_scaling.py              # 3×3 代理模型实验（按参数量倍数定 token）
│   ├── scale_to_30m.py             # 参数搜索，--target 60000000
│   ├── train_selected.py           # 读取最终配置启动训练
│   ├── overfit.py                  # 过拟合测试
│   ├── generate.py                 # 生成
│   └── plot.py                     # 曲线
├── configs/                        # 起始配置 + final_60m.json
├── docs/                           # 方法、数据、部署、48h 预算、runbook
├── tests/                          # CPU 单元测试
├── requirements.txt
└── reports/                        # 报告模板（实际报告在 ../results/）
```

## 命令链

```bash
# ① 环境：4090 24GB + torch 2.14.1+cu130
pip install --no-cache-dir -r requirements.txt
python -m pytest -q

# ② 数据：流式下载，不落原始语料
python scripts/prepare.py --tokenizer_docs 30000 --docs 200000
#    → train 278,476,675 token / val 5,716,800 token

# ③ 消融：3 个 seed，每个 seed 跑一遍完整六轮，每轮 1000 步
python scripts/greedy_search.py --steps 1000 --warmup 100 --decay 200 \
       --seed 42 --out runs/greedy_s42 --max_hours 6      # seed 43 / 44 同理

# ④ Scaling：每档模型训 10×/20×/40× 参数量的 token，共 9 点
python scripts/run_scaling.py --max_hours 14
python -m mathgpt.scaling runs/scaling/points.jsonl --flops 2.16e17

# ⑤ 60M 配置搜索 + 正式训练
python scripts/scale_to_30m.py --target 60000000 --out configs/final_60m.json
python scripts/train_selected.py --dry-run
python -m mathgpt.train --data data/processed --out runs/final_60m \
  --vocab 32000 --dim 480 --layers 16 --heads 4 --kv_heads 2 --hidden 1440 \
  --seq 1024 --batch 4 --accum 8 --steps 18311 --warmup 915 --decay 1831 \
  --eval_every 200 --eval_batches 20 --save_every 8000 --log_every 100 \
  --seed 42 --lr 3e-4 --position rope --norm_type rms --ffn_type swiglu
```

中断后用同一条命令加 `--resume` 续训。FLOPs 预算和步数按实测吞吐定，不照抄示例；子进程需外挂 timeout，Runner 不自动硬停。最终架构在 `runs/greedy/selected_architecture.json`，60M 配置在 `configs/final_60m.json`。没有数据或 GPU 时先跑 `--dry-run`。

## 模型统一接口

```python
from mathgpt.model import Config, GPT
model = GPT(Config())
logits, loss, caches = model(ids, targets=targets)
output_ids = model.generate(ids, max_new_tokens=32)
```

`Config` 参数变化不改变前向 API。默认 `Config()` 只有约 13.8M 参数，正式配置由 `scale_to_30m.py --target 60000000` 搜索生成，并以运行时 `actual_parameters` 为准。

## 注意

- 六轮贪心结果是**指定顺序、预算、种子下的局部胜者**，不是全局最优；因素之间存在交互。
- Scaling Law 只估算理论最优 N/D，不改变正式模型规模；9 点拟合 5 参数属病态问题，外推不可当结论。
- 正式训练单 seed，val_loss 有 ±0.02 量级波动；消融为 3 seed。
- 本工程不含 SFT/DPO/GRPO 后训练；未抓取可能受版权保护的教材，数据为 FineMath-4+。
- 环境相关：torch 2.14.1 在纯 CPU 环境下 `kv_heads=4` 的无 cache 前向会段错误（CUDA 上正常）；`--resume` 的 rng 状态按容错处理，恢复失败不中断训练。详见 `../PROVENANCE.md` 与 `../results/FINAL_RESULTS.md`。
