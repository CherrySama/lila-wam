# Panthera 双相机 LiLa-WAM 训练

项目提供两套独立配置，均训练六轴单臂加夹爪的 7 维关节位置策略，并同时使用
顶部和腕部相机。可以选择 LeRobot 真实数据，或选择 RoboTwin 仿真数据。

通过私有 GitHub 仓库克隆代码并下载数据的步骤见 [SERVER_TRAINING.md](SERVER_TRAINING.md)。

数据已经完整复制到项目内：

```text
data/panthera/
  panthera_wrist_top_zero1/
  panthera_wrist_top_zero2/
  panthera_wrist_top_zero3/
  manifest.json
data/robotwin/move_pillbottle_pad/move_pillbottle_pad_panthera_single_dr/
  data/                  # 300 个 HDF5
  video/                 # 顶部和腕部预览视频
  instructions/
  _traj_data/
  collection_manifest.json
  transfer_manifest.json
```

三个目录保留原有 LeRobot 结构，共 21 个原始文件，约 372 MiB。
`manifest.json` 记录每个文件的 SHA256，已与桌面原始文件逐一校验。
配置和训练/验证划分中的数据路径均相对项目根目录，不依赖本机用户名或桌面路径。

仿真目录保留采集时的 HDF5、预览视频、指令、轨迹和采集元信息，共 1,503 个
源文件，约 7.11 GiB。每个复制文件均已与源文件进行 SHA256 比对，300 个
HDF5 还额外与原始 `collection_manifest.json` 的采集哈希核对。
`transfer_manifest.json` 记录本次复制的文件清单和哈希。

## 选择训练数据

| 数据来源 | 配置文件 | 轨迹划分（训练/验证） | 启动脚本 | 输出目录 |
|---|---|---|---|---|
| LeRobot 真实数据 | `configs/panthera_dual.yaml` | 45 / 5 | `train_panthera.sh` | `checkpoints_vla/panthera/` |
| RoboTwin 单臂仿真 | `configs/panthera_robotwin.yaml` | 270 / 30 | `train_panthera_robotwin.sh` | `checkpoints_vla/panthera_robotwin/` |

两套配置分别计算、读取自己的归一化统计，分别保存 checkpoint。无需修改
`train.py` 来切换数据来源。训练仿真数据只需读取 HDF5，不需要在服务器安装
RoboTwin、SAPIEN 或规划器；仿真环境只在采集和 rollout 评估时需要。

## 数据与模型

以下为 LeRobot 配置；仿真配置的区别见后文。

- 50 条演示、22,646 帧，30 Hz。按完整轨迹确定性划分：训练 45 条
  （20,415 帧），验证 5 条（2,231 帧）。划分种子为 42，各来源分别留出 10%。
- 原始数据保持不变。AV1 视频先解码为 320×240 RGB 的磁盘内存映射，
  约占 9.72 GiB，避免训练时反复随机定位视频。准备程序逐帧验证视频时间戳。
- 状态和动作均按 `joint1.pos` 至 `joint6.pos`、`gripper.pos` 的顺序排列。
  `utils/stat-panthera-train.json` 仅使用训练轨迹计算归一化统计，并记录划分。
- 共享冻结的 DINOv3 ViT-L/16 编码两路图像，抽取 `[-12, -8, -4]` 特征。
  每路经共享适配器生成 64 个视觉 tokens，并加入可学习的相机标识。
- 保留原配置的 12 层、768 维、8 头 DiT 和 4 层视觉适配器。
  预测 32 步动作（在 30 Hz 下约 1.07 秒）。
- 4 层未来特征解码器同时监督两路相机，每路 300 个 patch，总共 600 个。
  未来偏移为 32 帧。尾部补齐动作及越界未来目标均通过 mask 排除。
- 单任务配置关闭任务向量。DINO 使用预训练权重，其余模块从头训练；
  未加载 RoboTwin 动作模型 checkpoint。

## 环境和权重

本机项目环境为 `.venv`（Python 3.12），通过 `--system-site-packages` 复用
`panthera-lerobot-v061` 已有的 PyTorch/CUDA 等包；OmegaConf 安装在项目环境内，
没有修改原 Conda 环境。主要依赖版本见 `requirements-panthera.txt`。

若在另一台机器建立完全独立的 Python 3.12 环境，可以执行：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-panthera.txt
```

预训练视觉权重位于 `dinov3_pretrain/dinov3-vitl16-pretrain-lvd1689m/`，
下载来源为 [ModelScope 的 facebook 模型仓库](https://modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m)。
`download_manifest.json` 记录下载文件的版本、长度和 SHA256，下载时已校验。
模型目录保留 LICENSE 和模型说明。完整权重和数据缓存未纳入 Git。

## 传到服务器

已准备完整训练包 `dist/LiLa-WAM-panthera-training.tar.gz`，包含项目代码、
三份真实数据、300 条仿真数据、两套配置与归一化统计，以及 DINOv3 预训练权重。
包内不含本机虚拟环境、解码缓存、Git 历史或训练输出。
旁边的 `.sha256` 文件用于校验传输完整性。

将这两个文件传到服务器后，在服务器执行：

```bash
sha256sum -c LiLa-WAM-panthera-training.tar.gz.sha256
tar -xzf LiLa-WAM-panthera-training.tar.gz
cd LiLa-WAM
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-panthera.txt
.venv/bin/python utils/prepare_panthera.py --config configs/panthera_dual.yaml
bash train_panthera.sh
```

上面最后两行选择的是真实数据。若选择仿真数据，改为：

```bash
.venv/bin/python utils/prepare_panthera.py --config configs/panthera_robotwin.yaml
bash train_panthera_robotwin.sh
```

服务器需要可用的 NVIDIA GPU 和与 CUDA 12.8 版 PyTorch 匹配的驱动。
真实数据缓存由视频生成，约需 9.72 GiB；仿真缓存由 HDF5 中的 JPEG 生成，
约需 25.20 GiB。只训练一种数据时，只需准备对应缓存。
若手动上传整个项目目录，也需同时包含 `data/` 和 `dinov3_pretrain/`；
只进行 Git push 不会上传被忽略的 `dinov3_pretrain/` 权重。

## 训练

### LeRobot 真实数据

从项目根目录执行：

```bash
.venv/bin/python utils/prepare_panthera.py --config configs/panthera_dual.yaml
bash train_panthera.sh
```

准备程序可以重复运行；源文件信息、图像尺寸和帧数均未改变时会复用缓存。
正式配置为 batch 32、BF16 计算、FP32 可训练参数和 Adam 状态、40 epochs，
学习率从 `2e-4` 余弦衰减至 `2e-5`，每个 epoch 执行验证。
该批量已通过本机 RTX 5060 Ti 16 GB 的完整双相机前向、反向与更新测试，
诊断峰值约 10.03 GiB，实际长时间训练以运行日志为准。

### RoboTwin 仿真数据

```bash
.venv/bin/python utils/prepare_panthera.py --config configs/panthera_robotwin.yaml
bash train_panthera_robotwin.sh
```

- 输入为 `panthera-single-v1` HDF5，共 300 条、58,722 帧。
- 270 条训练、30 条验证。每条轨迹最后一帧没有下一帧目标，因此不作为采样锚点；
  实际得到 52,563 个训练样本和 5,859 个验证样本。
- 相机顺序为 `[head_camera, wrist_camera]`，对应顶部和腕部。图像均缩放到
  320×240；源码的 RGB/JPEG 编码方式不需要额外交换颜色通道。
- 观测状态是当前 `joint_action/vector[k]`。这个字段在采集代码里实际记录
  机器人关节状态，所以动作目标使用 `k+1` 到 `k+32` 的关节位置，而非把
  `k` 帧状态重复当作第一步目标。两套配置的网络结构一致，动作时间偏移不同。
- 未来特征目标同样使用 `k+32` 的两路相机图像。越界目标和动作补齐都有 mask。
- 统计文件为 `utils/stat-panthera-robotwin-train.json`，仅从仿真训练集计算。
- batch 32、40 epochs，其余网络深度、Flow Matching 和未来特征损失保持一致。
- 已在本机用仿真数据完成三个 batch 的完整模型前向、反向与优化器更新，
  两个视角均获得梯度，显存峰值约 10.03 GiB。该检查不保存模型权重。
- 输出位于 `checkpoints_vla/panthera_robotwin/sft_<时间>/`，保存规则与真实数据相同。

仿真与真实数据不能只因都是 7 维就直接混合：仿真夹爪值为 0–1，而真实数据
记录夹爪位置；两者相机图像尺寸和采样节奏也不同。仿真 HDF5 没有逐帧时间戳，
采集元信息记录 `save_freq=15`，不能把预览视频的播放帧率当成真实采样频率。
当前两种入口用于分别训练；跨域混合或实机部署还需要对齐控制语义。

### 输出和恢复

真实数据输出目录为 `checkpoints_vla/panthera/sft_<时间>/`，仿真数据为
`checkpoints_vla/panthera_robotwin/sft_<时间>/`。两种目录都包含：

- `config.yaml`、`norm_stats.json`、`split.json`：当前训练设置与划分。
- `environment.json`：Python、PyTorch、CUDA 和 GPU 信息。
- `train.log`：训练日志；`metrics.jsonl`：逐 epoch 训练和验证指标。
- `last.pt`：最近一个完整 epoch，可断点续训。
- `best.pt`：按验证总损失选出的最佳 checkpoint。
- `checkpoint_epoch_<N>.pt`：每 5 epochs 额外保留的 checkpoint。

checkpoint 包含动作模型、优化器、调度器及随机数状态。文件采用临时文件加
原子替换保存。冻结 DINO 权重另存于模型目录，未重复写入 checkpoint。

恢复真实数据训练（使用原运行配置）：

```bash
bash train_panthera.sh \
  --config checkpoints_vla/panthera/sft_<时间>/config.yaml \
  --norm_stats_path checkpoints_vla/panthera/sft_<时间>/norm_stats.json \
  --resume checkpoints_vla/panthera/sft_<时间>/last.pt
```

恢复仿真数据训练：

```bash
bash train_panthera_robotwin.sh \
  --config checkpoints_vla/panthera_robotwin/sft_<时间>/config.yaml \
  --norm_stats_path checkpoints_vla/panthera_robotwin/sft_<时间>/norm_stats.json \
  --resume checkpoints_vla/panthera_robotwin/sft_<时间>/last.pt
```

恢复后会建立新的输出目录，接续剩余 epochs。验证使用固定的噪声种子，并隔离
验证阶段的随机数状态。验证损失用于选择 checkpoint，不等同于实机任务成功率。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python utils/check_panthera_training.py
.venv/bin/python utils/check_panthera_training.py --config configs/panthera_robotwin.yaml
```

数据或代码更新后，可重新生成完整服务器传输包：

```bash
.venv/bin/python utils/package_panthera.py
```

本次工作覆盖训练。仓库原来的 `robotwin_infer.py` 仍是 RoboTwin 专用接口；
Panthera 实机的双相机观测和关节指令接口需在部署阶段接入。
