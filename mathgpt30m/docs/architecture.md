# 模型架构与实验设计

| 层/模块 | 基线 | 候选 | 选型动机 | 评估方式 |
|---|---|---|---|---|
| 主干 | Dense decoder-only | MoE | 小规模单卡训练稳定、吞吐可测 | 训练稳定性/吞吐 |
| Attention | GQA: 8Q/2KV | MHA:8KV, MQA:1KV | 推理 KV cache 较小 | val loss / KV 显存 / tok/s |
| 位置 | RoPE | NoPE | 明确相对位置归纳偏置 | 长短上下文验证 |
| Norm | Pre-RMSNorm | Pre-LayerNorm | 实现简单且省均值运算 | 稳定性/耗时 |
| FFN | SwiGLU | GELU | 现代门控 FFN | 等参数量 loss |
| 权重 | token embedding tied | untied | 32k 词表占参数大头 | loss/参数数 |
| 初始化 | residual projection scaled | ordinary | 控制残差累积 | early loss spikes |
| 优化器 | AdamW 0.9/0.95 | AdamW 0.9/0.999 | 常见预训练 recipe | 相同步数 loss |
| 调度 | WSD (linear warmup / flat / linear decay) | cosine | 平台期便于延长训练，衰减期集中收尾 | 验证集 loss / 训练效率 |
| 精度 | bf16 if available | fp32 | 更省显存、动态范围大 | loss / tok/s |
| 语料 | FineMath-4+ | 3+ / 混合教材 | 高质量数学解题解释 | 统一数学 val loss |

## 规模

代理模型基准规模：vocab=32000, dim=256, layers=8, heads=8, kv_heads=2, ffn=704, max_seq=1024。
正式模型：vocab=32000, dim=480, layers=16, heads=4, kv_heads=2, hidden=1440, max_seq=1024，实测 59,612,640 参数。参数总量以运行时 `parameters` 为准。

## 架构消融（代理模型 4.83M → 最终 60M）

按 GQA vs MHA/MQA、RoPE vs NoPE、SwiGLU vs GELU、tied vs untied 的顺序逐一更改，**同参数量/同训练 tokens/同数据顺序/同 seed**。执行规模：6 轮 × 3 seed（42/43/44）× 1000 步，共 39 个 run；MHA/MQA/LayerNorm/GELU 等变体均已实现。只有差异稳定且足以覆盖运行成本才升级基线。

实测结论：kv_heads 三个 seed 给出三个不同答案（GQA/MQA/MHA），norm 仅 1/3 选 LayerNorm，ffn 3/3 保留 SwiGLU；untie 3/3 稳定但参数翻倍，在 60M 下不划算。最终采用 baseline。决策账本见 `../../results/greedy_s4{2,3,4}_decisions.json`。

## Scaling Law（固定胜出架构）

本次采用 3 档模型规模（4.83M / 8.58M / 14.09M，用维度/层数调整实际参数）× 3 档 token 预算（每档模型训 10×/20×/40× 参数量的 token，合计 1.93B token）。更多规模与 token 档位可继续外扩，覆盖更多数量级以降低对噪声的拟合。每个模型按相同 tokenizer、数据混合、固定 holdout、统一 WSD 规则训练。模型架构家族保持一致，但头数/宽度按可整除条件缩放。

验证集交叉熵：`L(N,D)=E + A/(N/1e6)^alpha + B/(D/1e6)^beta`，拟合 `E,A,B,alpha,beta`。用 scipy.optimize.least_squares 进行有界非线性最小二乘，留出部分 N,D 组合检验外推；若置信区间宽/残差大则不得宣称得到最优规模。

在给定算力 C 下，近似 `C=6ND`，搜索最小预测 loss 的 `(N,D)`。真实 GPU 成本还要考虑 attention 的二次项、embedding 参数、有效 FLOPs 利用率、数据准备与 checkpoint I/O。最佳 N/D **取决于预算**，不存在不附带预算的绝对最优值。

## WSD 细节

warmup: `lr = peak*(step+1)/warmup_steps`；stable: `lr=peak`；decay: 最后 decay_steps 线性降到 `0.1*peak`。WSD 相对 cosine 的主要价值是可在稳定阶段延长训练并安排末尾衰减；**延长总步数时需要明确新 decay 起点**，中断恢复不能默默改变已运行阶段的 schedule。

## 实验门槛

1. 单 batch overfit loss 明显下降；2. CPU 小模型测试通过；3. GPU 上至少一次保存-恢复；4. validation loss 下降；5. token/s 与峰值显存记录；6. 固定 prompt 比较 KV cache/no-cache logits 与时间；7. 消融记录实际参数数和 GPU 小时。任何未运行实验的结果保持空白。
