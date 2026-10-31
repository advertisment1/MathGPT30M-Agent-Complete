# MathGPT-60M 最终结果（报告模板）

> 供后续同类实验复用的空白模板。本次实验的完整报告在 [`../../results/FINAL_RESULTS.md`](../../results/FINAL_RESULTS.md)，运行凭证在 [`../../PROVENANCE.md`](../../PROVENANCE.md)。
>
> 本次取值速览：RTX 4090 24GB（NexGPU）/ torch 2.14.1+cu130 / 总 GPU ~17 h；9 个 Scaling 点，E=2.832、A=2.566、α=0.652、B=18.200、β=1.010、RMSE=0.043，N*=138.9M；最终 59,612,640 参数，600,014,848 token，best val_loss 2.1825 / final 2.4153。

## 环境
- GPU 型号/显存：待填写
- CUDA / PyTorch / commit：待填写
- 总 GPU 小时：待填写
- 数据集版本 / tokenizer hash：待填写

## 六轮架构选择
| 轮次 | 当前 baseline | 对照 | N | tokens | seed | val loss | GPU 时间 | 晋升? | 理由 |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 1–6 | 待运行 | 待运行 | | | | | | | |

## Scaling Law
- 九个真实点：`runs/scaling/points.jsonl`（待运行）
- E、A、B、alpha、beta：待拟合
- RMSE / 不确定性：待评估
- FLOPs 预算定义：待填写
- 理论 N*、D*：待计算（不改变最终训练规模）

## 正式训练（本次 60M）
- 最终配置/真实参数量：待生成
- Train/Val loss 曲线：待运行
- tokens/s、峰值显存：待测
- Checkpoint 恢复：待验证
- 数学文本生成示例：待运行

## 局限与失败记录
待填写。禁止把未运行的内容描述为实验结果。
