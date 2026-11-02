# 从私有 GitHub 仓库在服务器训练

代码仓库：`CherrySama/lila-wam`。数据和 DINO 权重保存在私有 Release
`panthera-training-data-v1` 的 8 个分卷中；`git clone` 不会下载这些附件。

服务器需有 Python 3.12、uv、GitHub CLI (`gh`)、NVIDIA GPU 和适合 CUDA 12.8
版 PyTorch 的驱动。先通过 `gh auth login` 登录有该私有仓库权限的账号。

```bash
gh auth login
gh repo clone CherrySama/lila-wam
cd lila-wam
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-panthera.txt
bash download_training.sh
```

`download_training.sh` 会下载分卷，逐片校验，合并并校验原压缩包，然后仅提取
`data/` 和 `dinov3_pretrain/`，保留当前 Git 仓库中的代码。下载中断后可以重试；
如某个分卷校验失败，删除该分卷后重新运行下载脚本。
下载、合并和解压需要约 26 GB 临时及原始数据空间，图像缓存和 checkpoint 另计。

## 选择真实数据

```bash
.venv/bin/python utils/prepare_panthera.py --config configs/panthera_dual.yaml
bash train_panthera.sh
```

50 条 LeRobot 演示，按 45/5 划分训练/验证。图像缓存约 9.72 GiB。
输出：`checkpoints_vla/panthera/`。

## 选择仿真数据

```bash
.venv/bin/python utils/prepare_panthera.py --config configs/panthera_robotwin.yaml
bash train_panthera_robotwin.sh
```

300 条单臂 RoboTwin 演示，按 270/30 划分训练/验证。图像缓存约 25.20 GiB。
输出：`checkpoints_vla/panthera_robotwin/`。离线训练不需要安装 RoboTwin 仿真环境。

两种入口均为双相机、7 维状态和动作、batch 32、40 epochs，使用各自的统计量。
夹爪含义和采样节奏的差异、checkpoint 和恢复说明见 `README_PANTHERA.md`。
