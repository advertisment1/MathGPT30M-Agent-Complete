# 运行凭证（Run Provenance）

> 目的：让任何人在不接触原始机器的前提下，判断"这些结果是不是真的在云 GPU 上跑出来的"。
> 原则：本文所有数值都由 `results/metrics.jsonl`、`results/points.jsonl`、`results/*decisions.json` 与 checkpoint 文件直接计算得到，**没有一个是手填的**。机器可读版本：[`results/provenance.json`](results/provenance.json)。

## 0. 一句话

单张 **NexGPU RTX 4090 24GB** 上，用 **PyTorch 2.14.1+cu130 / CUDA 13.0 / BF16 autocast**，顺序跑完 数据 → 39 个消融 run → 9 个 Scaling 点 → 18,311 步 60M 正式训练，累计约 **2.55B token** 前向+反向，全项目 GPU 墙钟约 **17 小时**。

## 1. 环境快照

| 项 | 值 | 取得方式 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4090，24 GB（单卡） | `nvidia-smi --query-gpu=name,memory.total` |
| 驱动 / CUDA | 595.99 / CUDA 13.0 | `nvidia-smi`、`torch.version.cuda` |
| PyTorch | 2.14.1+cu130 | `torch.__version__` |
| BF16 | `torch.cuda.is_bf16_supported() == True` | 运行时检查 |
| Python | 3.12.3 | `python3 --version` |
| 机器 | 26 vCPU，系统盘 16 GB（`--no-cache-dir` 装 torch 后剩 11 GB） | `df -h` |
| 计费 | $0.54/卡时，按秒计费，删实例才停费 | 平台价目 |

说明：运行期未把 `nvidia-smi` 原始文本存成文件，上表数字来自即时检查输出（同时记在 `results/FINAL_RESULTS.md` 环境表），是本凭证唯一没有原始文件支撑的一项，见第 6 节。

## 2. 时间线（本地时间 UTC+8）

| 时间 | 事件 |
|---|---|
| 10-25 | 数据准备：HF `finemath-4plus` 流式下载 200k docs，训 32k BPE，打包 `train.bin` / `val.bin` |
| 10-25 | 消融：6 轮 × 3 seed（42/43/44）× 1000 步 = 39 个 run，同一张卡顺序跑 |
| 10-25 → 10-26 12:44 | Scaling 3×3（4.83M / 8.58M / 14.09M × 10×/20×/40× token），共 1.93B token |
| 10-26 ~12:50 | 60M 正式训练启动（18,311 步，6 亿 token） |
| **10-26 15:43** | **欠费停机**，最后一次落盘 checkpoint 是 step 7999 |
| **10-26 16:52** | 重新计费，`--resume` 从 step 8000 续训 |
| 10-26 19:48 | 训练结束：step 18310，600,014,848 token |
| **10-26 19:53** | **第三次欠费停机**，发生在 scp 下载途中；恢复后重下并做 sha256 / zip CRC 校验 |

## 3. 各阶段命令

```bash
# ① 数据（CPU + 网络，流式，不落原始语料）
python scripts/prepare.py --tokenizer_docs 30000 --docs 200000

# ② 消融：3 个 seed 各跑一遍完整六轮贪心
python scripts/greedy_search.py --steps 1000 --warmup 100 --decay 200 \
       --seed 42 --out runs/greedy_s42 --max_hours 6      # seed 43 / 44 同理

# ③ Scaling：每档模型训 10×/20×/40× 参数量的 token
python scripts/run_scaling.py --max_hours 14

# ④ 拟合（scipy least_squares），FLOPs 预算按 C ≈ 6ND = 2.16e17
python -m mathgpt.scaling runs/scaling/points.jsonl --flops 2.16e17

# ⑤ 配置搜索：目标 60M
python scripts/scale_to_30m.py --target 60000000 --out configs/final_60m.json

# ⑥ 正式训练（中断后用同一条命令 + --resume 续上）
python -m mathgpt.train --data data/processed --out runs/final_60m \
  --vocab 32000 --dim 480 --layers 16 --heads 4 --kv_heads 2 --hidden 1440 \
  --seq 1024 --batch 4 --accum 8 --steps 18311 --warmup 915 --decay 1831 \
  --eval_every 200 --eval_batches 20 --save_every 8000 --log_every 100 \
  --seed 42 --lr 3e-4 --position rope --norm_type rms --ffn_type swiglu
```

各阶段 token 消耗（可复算）：

| 阶段 | run 数 | 每 run 规模 | 合计 token |
|---|---|---|---|
| 消融（150 步初筛） | 13 | 150 × 512 | 1.0M |
| 消融（3 seed × 1000 步） | 39 | 1000 × 512 | 20.0M |
| Scaling 9 点 | 9 | 48.3M → 563.8M | **1,925.6M** |
| 60M 正式 | 1 | 18,311 × 32,768 | **600.0M** |

## 4. 四条"只可能来自 GPU"的证据

### 4.1 逐 step 日志本身（24,271 行）

`results/metrics.jsonl` 每行是一个真实训练步的遥测：

```json
{"step": 1, "train_loss": 10.4321, "lr": 6.557e-07, "grad_norm": 9.66,
 "tokens_seen": 65536, "tok_per_s": 50765.99, "parameters": 59612640,
 "elapsed_step_s": 0.6455}
```

24,271 行 = 18,311 个唯一步 + 5,960 个重复步。重复不是错误，是第 2 条证据。

### 4.2 日志的两段结构 = 停机 + checkpoint 回滚

把日志按"step 不再递增"切段，只得到两段：

| 段 | 行范围 | step 区间 |
|---|---|---|
| 第 1 段 | 1 – 13,960 | 0 → 13,959 |
| 第 2 段 | 13,961 – 24,271 | 8,000 → 18,310 |

`step 8000–13959` 被记录了两次，长度 5,960 步 —— 这正好是"从 step 7999 的 checkpoint 回滚重算"的距离。任何平滑生成的假日志都不会自发产生这种断口。

### 4.3 吞吐量级

| 统计 | 值 |
|---|---|
| 中位数 | **45,150 tok/s** |
| p5 / p95 | 42,598 / 56,779 |
| step 0 | 24,606 tok/s（CUDA 首次初始化） |
| step 1 起 | 50,766 tok/s |

59.6M 参数、seq 1024、batch 4、每步 32,768 token、单步 0.65 s。纯 CPU 上同配置差两个数量级以上；而且 step 0 → step 1 的台阶是 CUDA context / kernel 首次启动独有的特征。

### 4.4 checkpoint 里的设备标签

`results/last.pt`（715,507,643 字节）的二进制中：

- `cuda:0` 出现 **585** 次 —— 参数张量保存时所在的设备
- `rng_cuda` 出现 1 次 —— 只有 CUDA 运行过才会存这一项
- `rng_cpu` 出现 1 次 —— 与之对应

sha256：`583dc4e9a915c6fec5287f8532277e7405820eec52462457036250512baf8237`

## 5. 产物校验表

| 文件 | 字节 | sha256 | 入库 |
|---|---|---|---|
| `results/metrics.jsonl` | 5,282,494 | `0b5e3b6e62707ce6a7001fcc0e61dc57061d6a6641ce1e8c680cca6db4ed1ab8` | yes |
| `results/metrics.png` | 72,089 | `7dba5412ffa63afe5209ea3312ae5db7d1b440cb34eb209b040fecd63d9652cc` | yes |
| `results/points.jsonl` | 921 | `4268972268351c846b13da6811b3d32f282d47b25d7e558fa3a7358c032b05dc` | yes |
| `results/FINAL_RESULTS.md` | 5,521 | `cbecf177c69c7341470851f534db205ec758b84116d5a5b71bfd0a874158f91c` | yes |
| `results/final_60m.json` | 333 | `72f30eeb85ea013bbe3b3412fa04c457d836b8a3d399e41cb9c4f97540ea67cb` | yes |
| `results/greedy_s42_decisions.json` | 15,941 | `79eadd54b5c01729344834732570bdf35c2e9d7a0479d0d1f0a0d035a1e2b3bc` | yes |
| `results/greedy_s43_decisions.json` | 15,913 | `658ccf0671b07699fedf0d1f69bc6a66b84c18302291d328ac9b2941c50ed11d` | yes |
| `results/greedy_s44_decisions.json` | 15,919 | `15c63437fb84940a06f10ac387b7762fc9edf06325040af764275b4fb6ebff04` | yes |
| `results/selected_architecture.json` | 466 | `2eb1d19500ff1323cbdd4894b6c613f8bb0d06f018d5cff22d3f53fc86770e6f` | yes |
| `results/last.pt` | 715,507,643 | `583dc4e9a915c6fec5287f8532277e7405820eec52462457036250512baf8237` | no（>100 MB） |

## 6. 已知缺口

1. **没有存 `nvidia-smi` 原始输出**。硬件/驱动信息来自运行时的即时输出与 `FINAL_RESULTS.md`，属于"操作者记录"，不是"文件级证据"。这是本凭证最薄弱的一环。
2. **实例已释放**，无法在原机上重跑或现场取证。因此证据只能是**离线可验证的产物 + 自洽性**，不能是"你自己去跑一遍"。
3. **`last.pt` 未入库**（GitHub 单文件 100 MB 限制，实际 715 MB）。只能通过 sha256 与内部设备标签间接证明它是 GPU 产物；需要权重本身的话另议（Release / 网盘 / LFS）。
4. **实例停机 3 次**，其中一次导致 5,960 步（约 1.95 亿 token）的回滚重算。`metrics.jsonl` 中的重复段对应这次回滚，token 计数以最终 `tokens_seen` 为准。
5. 正式训练**单 seed**。val_loss 的具体数值带有 seed 波动（消融里同配置不同 seed 相差约 0.02–0.1），因此 `2.1825` 应当理解为"这一次的观测值"，不是该配置的期望值。

## 7. 如何独立复核

见根目录 [`README.md` 第二节](README.md#二怎么复核3-条命令纯-python-标准库)的三条命令：日志自洽性、产物 sha256、Scaling 九点一致性。全部只依赖 Python 标准库，不需要 GPU、不需要 torch。
