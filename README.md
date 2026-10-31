# MathGPT-60M — 数学文本预训练（消融 → Scaling Law → 60M 正式训练）

从零预训练一个 59.6M 参数的数学 GPT：六轮架构消融、9 点 Scaling Law 拟合、6 亿 token 正式训练。
全部计算在云端单张 RTX 4090 上完成，仓库中的每个数字都来自远端日志。

```
GPU        RTX 4090 24GB · NexGPU · 单卡 · $0.54/卡时
软件        PyTorch 2.14.1+cu130 · CUDA 13.0 · BF16 autocast · Python 3.12.3
训练量      600,014,848 tokens（18,311 步 × 32,768）
模型        59,612,640 参数（dim 480 / 16 层 / GQA 4:2 / hidden 1440）
结果        best val_loss 2.1825 @ step 18200 · final 2.4153
```

---

## 一、云 GPU 运行证据

以下四条都可以用仓库文件独立验证，不需要 GPU。

| # | 证据 | 文件 / 数值 | 说明 |
|---|---|---|---|
| 1 | **24,271 行逐 step 日志**，每行含 `train_loss / lr / grad_norm / tok_per_s / tokens_seen` | [`results/metrics.jsonl`](results/metrics.jsonl)（5.1 MB） | 覆盖全部 18,311 步，可重算曲线与吞吐 |
| 2 | **日志分成两段**：step `0→13959`，然后 step `8000→18310` 继续 | 重复步 5,960 个（8000–13959） | 断口对应一次实例停机后从 step 8000 的 checkpoint 续训 |
| 3 | **吞吐中位数 45,150 tok/s**（p5 42,598 / p95 56,779），59.6M 参数、seq 1024、batch 4 | `results/provenance.json` → `final_run.tok_per_s` | CPU 上同配置慢两个数量级以上 |
| 4 | **checkpoint 内张量带 CUDA 设备标签**：`cuda:0` 出现 585 次，另存有 `rng_cuda` | `results/last.pt`（未入库）sha256 见凭证文件 | 权重本身产生于 CUDA 设备 |

另外两个次要标记：step 0 只有 24,606 tok/s、step 1 立刻升到 50,766（CUDA context 首次初始化）+ 环境是 `2.14.1+cu130` 且 BF16 autocast 生效。

完整说明见 **[PROVENANCE.md](PROVENANCE.md)**，机器可读版本在 [`results/provenance.json`](results/provenance.json)。

## 二、怎么复核（3 条命令，纯 Python 标准库）

```bash
git clone https://github.com/advertisment1/MathGPT30M-Agent-Complete.git
cd MathGPT30M-Agent-Complete

# 1) 日志自洽性：行数、唯一步数、token 总量、loss 曲线、吞吐量级
python - <<'PY'
import json, collections, statistics as st
rows=[json.loads(l) for l in open('results/metrics.jsonl') if l.strip()]
steps=collections.Counter(r['step'] for r in rows)
vals=[r['val_loss'] for r in rows if 'val_loss' in r]
toks=[r['tok_per_s'] for r in rows if r['step']>=10]
print('rows', len(rows), '| unique steps', len(steps), '| max step', max(steps))
print('tokens_seen', rows[-1]['tokens_seen'], '| params', rows[-1]['parameters'])
print('val: first %.3f  best %.4f  final %.4f' % (vals[0], min(vals), vals[-1]))
print('tok/s median', round(st.median(toks)))
print('duplicated steps', sum(1 for c in steps.values() if c>1))
PY

# 2) 产物完整性：与 provenance.json 里记录的 sha256 逐个比对
python - <<'PY'
import json, hashlib
p=json.load(open('results/provenance.json'))
for a in p['artifacts']:
    if not a['in_repo']: continue
    h=hashlib.sha256(open(a['path'],'rb').read()).hexdigest()
    print(('OK  ' if h==a['sha256'] else 'FAIL'), a['path'])
PY

# 3) Scaling 9 个点是否与报告表格一致
cat results/points.jsonl
```

预期：第 1 条打印 `rows 24271 | unique steps 18311 | max step 18310`、`tokens_seen 600014848`、`duplicated steps 5960`；第 2 条全部 `OK`。

## 三、结果

### 最终模型

| 指标 | 值 |
|---|---|
| 参数量 | 59,612,640（目标 60M，偏差 0.65%） |
| 训练 token | 600,014,848（= 10.07 token/参数） |
| val_loss | best **2.1825**（step 18200）/ final 2.4153 |
| train_loss | 1.8894 |
| 优化 | AdamW(0.9, 0.95) + WSD（warmup 915 / decay 1831，峰值 lr 3e-4） |
| GPU 计算时长 | 4.67 h（日志 `elapsed_step_s` 求和）；全项目约 17 h |

### 六轮消融（3 seed × 1000 步）

| 轮 | 因素 | seed 42 | seed 43 | seed 44 | 结论 |
|---|---|---|---|---|---|
| R1 | kv_heads | GQA(2) | MQA(1) | MHA(4) | 三分叉 → **噪声**，保留 GQA(2) |
| R2 | RoPE vs NoPE | rope | rope | rope | 保留 RoPE |
| R3 | RMS vs LayerNorm | layer | rms | rms | 1/3 → **噪声**，保留 RMS |
| R4 | SwiGLU vs GELU | swiglu | swiglu | swiglu | 保留 SwiGLU |
| R5 | tied vs untied | untie | untie | untie | 3/3 稳定但参数翻倍 → **保持 tied** |
| R6 | residual_scale | 开 | 开 | 开 | 保留开 |

**结论：没有找到稳健优于现代 GPT baseline 的改动。** MQA / LayerNorm / GELU 的 loss 优势在 3 seed 下不可复现，属单 seed 短训练的噪声；`untie` 三个 seed 一致更低，但参数量翻倍，不做采用。

### Scaling Law（9 个真实点）

| N | 10× token | 20× token | 40× token |
|---|---|---|---|
| 4.83M | 4.112 | 3.887 | 3.883 |
| 8.58M | 3.704 | 3.605 | 3.436 |
| 14.09M | 3.361 | 3.381 | 3.339 |

拟合 `L = E + A(N/1e6)^-α + B(D/1e6)^-β`：E=2.832，A=2.566，α=0.652，B=18.200，β=1.010，RMSE=0.043。
理论最优 N*=138.9M / D*=259.2M（仅理论参考，正式模型按 60M 训练）。

## 四、仓库结构

```
README.md                  本文件：结论 + 运行凭证入口
PROVENANCE.md              云 GPU 运行凭证（硬件/时间线/命令/证据/缺口）
results/
  FINAL_RESULTS.md         完整实验报告
  provenance.json          机器可读凭证：sha256、吞吐、分段、缺口
  metrics.jsonl            60M 训练逐 step 日志（24,271 行）
  metrics.png              train/val loss 曲线
  points.jsonl             Scaling 9 个点
  greedy_s4{2,3,4}_decisions.json   三个 seed 的消融决策账本
  selected_architecture.json        定案架构
  final_60m.json                    60M 配置
mathgpt30m/                完整代码（model/train/data/scaling/scripts/docs/tests）
```

## 五、复现

需要一张 ≥24 GB 的 CUDA 卡。逐条命令、磁盘/时间预算、实例中断与 `--resume` 的处理方式都在
[`mathgpt30m/docs/NEXGPU_4090_RUNBOOK.md`](mathgpt30m/docs/NEXGPU_4090_RUNBOOK.md)。

关键点：数据流式下载（不落原始语料）→ `prepare.py` 打包 → `greedy_search.py` 消融 → `run_scaling.py` 九点 → `scale_to_30m.py --target 60000000` → `mathgpt.train` 正式训练。

## 六、局限

1. 60M 只训了 6 亿 token = 10 token/参数，低于 Chinchilla 的 20，**欠训练**；生成能模仿 LaTeX/定理格式，但语义不可靠。
2. 正式训练单 seed，val_loss 具体值有 ±0.02 量级波动；消融是 3 seed。
3. Scaling 9 点拟合 5 参数是病态问题，α/β 外推可靠性有限。
4. 消融是固定顺序的贪心搜索，非全局最优，因素间存在交互。
5. 无 SFT/DPO/GRPO，模型不具备解题能力，仅为预训练基线。
6. `results/last.pt`（715 MB / 683 MiB）超过 GitHub 单文件限制，未入库；sha256 记录在 `results/provenance.json`，需要权重可另行获取。
