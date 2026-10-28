# MathGPT-30M — Sequential Ablation → Scaling Law → Fixed 30M Pretraining

**状态：代码已打包并通过本地测试；没有云 GPU 实测权重、消融胜者或 Scaling Law 结果。**

这是一个可交接给其他 Agent 执行的独立数学 GPT 训练工程。训练预算为**单卡最多 48 GPU 小时**。核心流程是：先六轮顺序贪心消融（每轮将优胜者晋升为新 baseline），再固定架构家族拟合 N/D Scaling Law，最后**实际训练始终约 30M 参数**。

## 阅读顺序

1. `AGENT_HANDOFF.md`：Agent 必须遵守的执行协议、验收标准、预算与完整命令。
2. `docs/experiment_rationale.md`：每个选择的书本依据、基线、对照、实验风险。
3. `docs/cloud_gpu.md`：GPU Pod、SSH、持久化磁盘。
4. `mathgpt/model.py`、`mathgpt/train.py`、`scripts/greedy_search.py`、`scripts/run_scaling.py`：核心代码。

## 目录

```text
mathgpt30m/
├── AGENT_HANDOFF.md                # Agent 执行契约、流程图、验收标准
├── README.md                       # 项目入口
├── mathgpt/
│   ├── model.py                    # 统一 GPT 接口；MHA/GQA/MQA、RoPE/NoPE、RMS/LN、SwiGLU/GELU
│   ├── train.py                    # AdamW、BF16、WSD、梯度累积、日志、checkpoint
│   ├── data.py                     # FineMath streaming、BPE、packing、memmap
│   ├── experiment.py               # 结果校验、晋升判定
│   └── scaling.py                  # 五参数 Scaling Law 拟合与理论 N*/D*
├── scripts/
│   ├── prepare.py                  # 数据准备
│   ├── greedy_search.py            # 六轮顺序贪心消融
│   ├── run_scaling.py              # 3×3 代理模型实验
│   ├── scale_to_30m.py             # 选中架构缩放到约 30M（准确统计参数）
│   ├── train_selected.py           # 读取最终配置启动 30M 训练
│   ├── overfit.py                  # 过拟合测试
│   ├── generate.py                 # 生成
│   └── plot.py                     # 曲线
├── configs/                        # 原始对照配置及最终输出目录
├── docs/                           # 方法、数据、部署、48 小时预算
├── tests/                          # CPU 单元测试
├── requirements.txt
└── reports/                         # Agent 实测报告（模板）
```

## 最短执行链

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/prepare.py --tokenizer_docs 30000 --docs 200000
python scripts/greedy_search.py --dry-run
python scripts/greedy_search.py --max_hours 6
python scripts/run_scaling.py --dry-run
python scripts/run_scaling.py --max_hours 8
python -m mathgpt.scaling runs/scaling/points.jsonl --flops 1e18
python scripts/scale_to_30m.py
python scripts/train_selected.py --dry-run
python scripts/train_selected.py --steps 5000
```

**重要**：命令中的 FLOPs 和正式训练步数只是示例，必须根据 GPU 实测吞吐与预算调整；子进程需外部 timeout/报警，不能依赖 Runner 自动硬停止。最终选择文件在 `runs/greedy/selected_architecture.json`，30M 配置在 `configs/final_30m.json`。没有数据或 GPU 时先用 `--dry-run`，不要伪造实验。

## 模型统一接口

```python
from mathgpt.model import Config, GPT
model = GPT(Config())
logits, loss, caches = model(ids, targets=targets)
output_ids = model.generate(ids, max_new_tokens=32)
```

`Config` 参数变化不会改变前向 API。请勿直接使用默认 Config 作为 30M：默认配置只有约 13.8M 参数，正式 30M 必须通过 `scale_to_30m.py` 生成并检查。

## 注意

- 六轮贪心搜索结果是**指定顺序、预算、种子下的局部胜者**，不是全局最优。
- Scaling Law 只估算理论最优 N/D，不改变正式模型 30M。
- 本工程未加入 SFT/DPO/GRPO 后训练；也未抓取可能受版权保护的 Stanford 教材。
- 运行前需自行核实 FineMath 数据集版本、许可、远端配置名称及网络权限。
