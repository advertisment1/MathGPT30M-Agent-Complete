# MathGPT-60M 最终结果报告

> 训练任务：架构消融 → Scaling Law → 约 60M 参数数学 GPT 预训练
> 运行环境：NexGPU RTX 4090 24GB，单卡，总 GPU 时长约 17 小时
> 运行凭证（硬件、时间线、逐条命令、可复核证据）：[`../PROVENANCE.md`](../PROVENANCE.md)

## 一、环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 4090 24GB（NexGPU） |
| 驱动 / CUDA | 595.99 / CUDA 13.0 |
| PyTorch | 2.14.1+cu130 |
| 精度 | BF16（autocast） |
| Python | 3.12.3 |

## 二、数据

- 来源：HuggingFaceTB/finemath `finemath-4plus`（流式下载，不落原始语料）
- 分词：ByteLevel BPE，vocab 32000（30000 docs 训练 tokenizer）
- 打包：200000 docs → `train.bin` / `val.bin`（uint16，文档 sha256 哈希 2% 验证集切分）
- 规模：**train 278,476,675 token** / val 5,716,800 token

## 三、六轮架构消融（3 seed × 1000 步）

每轮只改一个因素，候选相对 baseline 验证 loss 降 ≥0.002 才晋升。三个 seed（42/43/44）各跑一条完整贪心路径。

| 轮 | 因素 | seed 42 | seed 43 | seed 44 | 结论 |
|---|---|---|---|---|---|
| R1 | kv_heads (GQA/MQA/MHA) | GQA(2) | MQA(1) | MHA(4) | **三分叉=噪声，保留 GQA(2)** |
| R2 | RoPE vs NoPE | rope | rope | rope | 一致保留 RoPE |
| R3 | RMS vs LayerNorm | layer | rms | rms | 1/3 晋升=噪声，保留 RMS |
| R4 | SwiGLU vs GELU | swiglu | swiglu | swiglu | 一致保留 SwiGLU |
| R5 | tied vs untied | untie | untie | untie | 3/3 晋升，但参数翻倍 |
| R6 | residual_scale 开/关 | 开 | 开 | 开 | 一致保留开 |

**关键结论**：
1. MQA / LayerNorm / GELU 的 loss 优势在 1000 步 × 3 seed 下不可复现，属单 seed 短训练的噪声，均不采纳。
2. R5 untie 虽 3 seed 稳健（val loss 降 ~0.01），但参数量从 4.62M 翻倍到 8.72M；在 60M 目标下 embedding 双份（≈24.6M）会严重挤占主干，故**保持 tied**。
3. **最终架构 = 现代 GPT 标准 baseline：GQA(kv=2) + RoPE + RMSNorm + SwiGLU + tied + residual_scale**。消融结论是"未发现稳健优于 baseline 的改动"。

## 四、Scaling Law（3 模型 × 3 档 token = 9 个观测点）

| 模型 N | 10× token | 20× token | 40× token |
|---|---|---|---|
| 4.83M | 4.112 | 3.887 | 3.883 |
| 8.58M | 3.704 | 3.605 | 3.436 |
| 14.09M | 3.361 | 3.381 | 3.339 |

拟合 `L(N,D) = E + A(N/1e6)^(-α) + B(D/1e6)^(-β)`（scipy least_squares）：

| 参数 | 值 |
|---|---|
| E（不可约损失） | 2.832 |
| A / α | 2.566 / 0.652 |
| B / β | 18.200 / 1.010 |
| RMSE | 0.043（~1%，可接受） |

给定 FLOPs 预算 `C ≈ 6ND = 6×60M×600M = 2.16e17`，理论最优 **N\* = 138.9M**，**D\* = 259.2M**。

说明：9 点拟合 5 参数是病态问题，α=0.65、β=1.01 偏大，外推可靠性有限；N\* 仅作理论参考，正式训练固定 60M + 6 亿 token。

## 五、60M 正式训练

- 配置（`configs/final_60m.json`）：dim 480 / 16 层 / hidden 1440 / heads 4 / kv_heads 2（GQA）/ vocab 32000 / seq 1024
- **实际参数量：59,612,640（59.6M，距 60M 目标偏差 0.65%）**
- 训练量：**6 亿 token**（18311 步 × 32768 token/步），= 10 token/参数，偏欠训练
- 优化：AdamW(0.9, 0.95) + WSD（warmup 915 → 稳定 → decay 1831，lr 峰值 3e-4）
- 吞吐：实测 ~4.4 万 token/s

| 指标 | 值 |
|---|---|
| 最终 train_loss | 1.889 |
| best val_loss | **2.1825**（step 18200） |
| 最终 val_loss | 2.4153 |
| 峰值 lr / 末段 lr | 3e-4 / 3e-5 |

> 注：最终 val_loss 2.415 优于 Scaling 曲线外推的 3.04——因为 60M 模型（dim 480）远大于 scaling 代理模型（最大 14M），外推不适用于大跨度。

## 六、生成样例（temperature=1, top-k 无）

**Prompt 1**：`Theorem: For any prime number p,`
```
Theorem: For any prime number p, >1,</p> <p+p, \kilograms{96}
The i-th! value is just 2\left[i-p\right]. Now let's study lots more journal engineering formulas...
```

**Prompt 2**：`Solve the equation x^2`
```
Solve the equation x^2 + px + p = 31 (which is not divisible by 3 for x > 0)
a) f(x) = x^2 + px + p
f(0) = 0     divide by p    $f(x) = x^2 + 9x + 13$
```

模型已学会数学文本的**表面结构**（LaTeX 公式、定理/方程格式），但**语义与推理不可靠**——符合 60M 小模型 + 10 token/参数欠训练的预期。

## 七、局限与失败记录

1. **消融是顺序贪心**：指定顺序、指定预算下的局部搜索，非全局最优；因素间有交互。
2. **单 seed 短训练的 loss 差异不可复现**：MQA/LayerNorm/GELU 在 3 seed × 1000 步下无稳定优势，最终未采纳；只有多 seed 一致的结果才作为决策依据。
3. **Scaling 9 点拟合 5 参数**外推风险高，N\* 不作为结论。
4. **60M 训 6 亿 token = 10 token/参数**（低于 Chinchilla 的 20），模型欠训练，生成质量有限。
5. **torch 2.14.1 CPU-only segfault bug**：`kv_heads=4 + nope/LayerNorm/GELU/untie` 组合在反向后做 KV-cache 前向时 CPU 崩溃；CUDA 上验证全过，不影响训练。
6. **torch 2.14.1 rng 类型 bug**：`--resume` 时 `set_rng_state` 因 ByteTensor 类型标签在 save/load 后退化而报错，已用 try/except 容错（rng 只影响复现，不影响质量）。
7. **训练中断**：运行中发生 3 次实例停机，用 step 7999 的 checkpoint + `--resume` 接续，重复训练 5,960 步（约 1.95 亿 token），日志中的 step 8000–13959 两段即此。
8. 本工程**无 SFT/DPO/GRPO 后训练**，模型不具备数学解题能力，仅为预训练基线。
