# 架构决策与实验依据：课本方法 → 当前 baseline → 候选 → 提升规则

## 参考来源及证据层级

- **S1**：Hugging Face《The Smol Training Playbook》（用户提供中文 PDF），目录中“每个大模型都始于一次小型消融实验”“设计模型架构”（Attention、Embedding sharing、位置编码、稳定性、tokenizer）以及“优化器与训练超参数”“扩展定律”“数据策划”“训练马拉松”。中文 PDF 约第 9–52 页讨论实验、架构和超参。原书明确强调每次只改一个变量、在目标场景测试后再修改基线。
- **S2**：用户的《AI Infra 逐日执行日历》D33–D40：GQA、RoPE、RMSNorm、SwiGLU、embedding tying、初始化、FineWeb/FineMath 类数据、BF16/AdamW、WSD 对照、KV Cache 和 30M 训练。
- **工程推断**：本项目六轮搜索的具体顺序、`min_delta` 阈值、代理模型大小、48 GPU 小时预算切分、选择 30M 的维度搜索等是**本工程制定的实验协议**，并非书中实验的直接结论。

| 轮次 | 课本讨论/依据 | 起始 baseline | 本工程候选 | 评价维度 | 何时提升 |
|---|---|---|---|---|---|
| 1 Attention | S1 架构 Attention：MHA、GQA、MQA 权衡 | GQA/MQA 小模型可配置 | KV heads 1、2、4 | Val loss、参数量、KV cache、吞吐 | 明显降低验证 Loss 且资源可接受 |
| 2 Position | S1 位置编码/长上下文、RoPE/NoPE | RoPE | NoPE | Val loss、位置泛化 | 统一上下文设置下改善 |
| 3 Norm | S1 稳定性及核心组件 | Pre-RMSNorm | Pre-LayerNorm | Loss、梯度稳定性 | 稳定改善 |
| 4 FFN | S1 其他核心组件、门控 FFN | SwiGLU | GELU | Loss、参数量 | 改善且排除仅因参数变化 |
| 5 Embedding | S1 Embedding sharing | tied | untied | Loss、参数效率 | 收益足以覆盖额外参数 |
| 6 Initialization | S1 提升稳定性 | 缩放残差输出初始化 | 不缩放 | 初期 loss spikes、val loss | 稳定改善 |

### 为什么不是把六个修改同时加上？

因为同时改多个组件无法判断增益来源；S1 的消融方法是从可工作的基线出发，逐个修改，成功后把结果作为新基线。这个流程是**顺序贪心搜索**，优点是计算预算可控、解释性强；缺点是交互项可能让后面的改动在早期无效，因此结果只能称为“在指定搜索顺序与预算下的获胜配置”。

### 比较公平性的特殊问题

1. `kv_heads` 改变 attention 投影参数数；`ffn_type` 改变 FFN 参数数；`untie` 直接增加 vocab×hidden 参数。**严格等参数量**的消融需另设调整隐藏维度的对照。本版本先做相同训练 token 的**工程效果筛选**，然后人工复核。
2. 相同 seed 不保证不同架构拥有完全相同的参数初始化随机数序列。应使用多个 seed 重复。
3. 预训练模型可能在 150 steps 后尚未体现稳定差异。默认 150 steps 是烟测级预算，不是可靠架构研究。实际 Agent 应根据吞吐调整，并报告预算局限。
4. FineMath 的 validation split 固定，所有候选使用相同 tokenizer。训练数据 memmap 通过相同 seed 采样。

## 训练方法依据

- **AdamW**：S1 优化器章讨论 AdamW 与替代优化器。本项目使用 β=(0.9,0.95)、weight decay=0.1（不衰减 embedding/norm），作为基线而非声称最优。
- **BF16**：若 CUDA GPU 支持 BF16 则 autocast；CPU smoke 用 FP32。BF16 不需要 FP16 常见的 loss scaling。
- **WSD**：线性 Warmup → Stable → 线性 Decay（最终学习率为 peak 的 10%）。适合将 Stable 阶段延长后安排收尾，但不能中途修改既有调度而不记录。
- **数据**：FineMath-4+ 默认数学领域数据；ByteLevel BPE 32k、文档哈希分割、packing、memmap；若增加开放教材必须核实许可。
- **后训练**：本工程没有后训练。若未来加入，优先构建数学指令 SFT，并独立验证；不能把普通 next-token 预训练叫做 SFT。

## Scaling Law 方法来源与边界

S1 的“扩展定律：需要多少参数，多少数据？”给出规模/数据/计算预算联动的思路。工程采用可拟合的 Chinchilla 型经验形式：

`L(N,D)=E+A*(N/1e6)^(-alpha)+B*(D/1e6)^(-beta)`。

- 用 SciPy `least_squares` 拟合五参数；并非 sklearn 训练模型参数，也不是神经网络可训练权重。
- 采用 `C≈6ND` 是常用稠密 Transformer 近似，不包含全部注意力计算、I/O 和硬件效率。
- 在固定 C 下计算理论最优 N*, D*；**实际 30M 训练规模不随 N* 改变**。
- 3×3 九点拟合五参数可能病态，必须报告残差、外推风险；需要时增加预算/数据点。

## 项目完成后应当如何写结论

每个结构决策按四段写：**书中为何考虑该方法 → 我们测试了哪些配置 → 真实验证指标是什么 → 为什么保留/替换**。不得用“课本说好”替代真实数据，也不得将本工程尚未运行的实验写成已完成。
