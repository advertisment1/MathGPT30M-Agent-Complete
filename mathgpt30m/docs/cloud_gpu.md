# RunPod 云 GPU 部署（Windows 本地 → Linux 远程）

1. 注册 RunPod，创建 Network Volume（推荐保存 `/workspace` 下 checkpoint），选择同机房的 GPU Pod。建议 CUDA 兼容的 PyTorch 模板，GPU 优先 A100 40GB / L40S 48GB；24GB 卡也可尝试减小 batch/seq。
2. 在 Windows PowerShell：`ssh-keygen -t ed25519`。把 `$env:USERPROFILE\.ssh\id_ed25519.pub` 的公钥加入 RunPod 设置；不要上传私钥。
3. 在 Pod 的 Connect 页面获取 SSH 命令（主机/端口每次以实际显示为准），PowerShell 执行该命令；VS Code 安装 Remote-SSH 后使用同一连接配置。
4. 在本机 PowerShell 传输 zip：`scp -P <PORT> mathgpt30m.zip root@<HOST>:/workspace/`（Windows PowerShell 中替换占位符；也可以先传 GitHub 私有仓库再 git clone）。
5. 远端执行：

```bash
cd /workspace
unzip mathgpt30m.zip
cd mathgpt30m
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(),torch.cuda.get_device_name(0))'
python -m pytest -q
python scripts/prepare.py --tokenizer_docs 30000 --docs 200000
python -m mathgpt.train --steps 2000 --warmup 100 --decay 300 --out /workspace/runs/base
python scripts/plot.py /workspace/runs/base/metrics.jsonl
```

6. 为避免断线中断，使用 `tmux new -s train`，运行后 `Ctrl+B` 再 `D` detach；`tmux attach -t train` 重连。`python -m mathgpt.train --resume /workspace/runs/base/last.pt --out /workspace/runs/base --steps 2000` 续训。
7. 下载输出：`scp -P <PORT> -r root@<HOST>:/workspace/runs/base .\runs\`。核对 checkpoint 已保存到持久盘，再停止或销毁 Pod；**停止 Pod 不代表存储零费用**。Network Volume 独立于 Pod 存在。

注意：RunPod 上 Pod 本地 volume 与独立 network volume 生命周期不同，销毁 Pod 前确认数据位置。不要把 HF token、SSH 私钥写进代码或 ZIP。云 GPU 需要用户自己开通/付费，项目代码不会自动租 GPU。
