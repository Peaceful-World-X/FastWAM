# Pi0.5

## 1. 本地-H100、RTX4060、RTX4080

### 初始化
```bash
# 拉取代码
mkdir -p ~/code && cd ~/code
git clone -b sduty https://gitee.com/Peaceful-World-X/openpi.git
GIT_LFS_SKIP_SMUDGE=1 git clone https://gitee.com/Peaceful-World-X/lerobot.git
# 拉取解压镜像（密码cyto）
rsync -avz --progress cyto@172.16.10.40:/home/cyto/docker/images/openpi_v3.tar ~/openpi_v3.tar
sudo apt update && sudo apt install -y pv && pv -p -t -e -r -b openpi_v3.tar | docker load
# 传模型文件
sudo rsync -avzP --mkpath --progress cyto@172.16.10.40:/home/cyto/results/ /home/results/
# 把 swap 调整为 32G
sudo swapoff -a && sudo rm -f /swapfile && \
sudo fallocate -l 32G /swapfile && sudo chmod 600 /swapfile && \
sudo mkswap /swapfile && sudo swapon /swapfile && \
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
# 初始化容器
docker run -it \
  --name openpi \
  --init \
  --gpus all \
  --shm-size=16G \
  --network host \
  -v /home/cyto/code:/home/cyto/code \
  -v /home/models:/home/models \
  -v /home/results:/home/results \
  -v $HOME/.cache/uv:/root/.cache/uv \
  openpi:v3.0
# 同步环境
apt-get update && apt-get install -y cmake
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

### 推理
```bash
# 进入环境
docker start openpi && docker exec -u 0 -it openpi bash -c "cd /home/cyto/code/openpi && exec /bin/bash"

# 推理 分配 8.5G 左右
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
# 第一版
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.dir=/home/results/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000

# 第二版：自动复位、但是反应迟钝
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.dir=/home/results/openpi/checkpoints/pi05_cytoderm14_joint_arm_move/my_experiment_cytoderm13_joint_007/40000/

# ---------------------------------------------------------------------------------------------------------------------
# 使用 8010 端口
uv run scripts/serve_policy.py --port=8010 policy:checkpoint \
  --policy.dir=/home/results/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000

# 原始命令
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_cytoderm11_joint_arm_move \
  --policy.dir=/home/results/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000

# RTX 4060（8G 显存）
export XLA_PYTHON_CLIENT_MEM_FRACTION=1
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

---
## 2. 云端-A800

### 初始化
```bash
# 初始化容器（卷必须用 :z 才能在有 SELinux 的宿主机上写入；:rw 无效）
docker run -it \
  --name openpi \
  --init \
  --gpus all \
  --shm-size=64G \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -p 8000-8010:8000-8010 \
  -v /home/xuewenyao/code:/home/cyto/code:z \
  -v /home/models:/home/models:z \
  -v /home/results:/home/results:z \
  -v $HOME/.cache/uv:/root/.cache/uv:z \
  openpi:v3.0
```

### 推理
```bash
# 进入环境
docker exec -u 0 -it openpi_ymy_new /bin/bash
cd /home/yaomingyuan/Program/openpi_main

# 计算归一化统计量（均值/方差）并保存，供训练与推理使用（仅需运行一次）
uv run --group rlds scripts/compute_norm_stats.py \
  --config-name=pi05_cytoderm11_joint_arm_move \
  --max-frames=500000

# 训练（全部 GPU；若需单卡请先执行 export CUDA_VISIBLE_DEVICES=0）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_cytoderm11_joint_arm_move \
  --exp-name=my_experiment_cytoderm11_joint_007 \
  --overwrite

# 推理（降低 XLA 显存占用，为推理留出显存）
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_cytoderm11_joint_arm_move \
  --policy.dir=/home/pub_envs/openpi/checkpoints/pi05_cytoderm11_joint_arm_move/my_experiment_cytoderm11_joint_007/40000
```

---

## 3. FastWAM（OpenPI 协议 WebSocket 推理，仓库：`~/code/FastWAM`）

FastWAM 为 **PyTorch**，不需要设置 `XLA_PYTHON_CLIENT_*`。可选显存相关：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

依赖（任选其一装全即可）：

- `pip install -e /home/xuewenyao/code/openpi/packages/openpi-client`
- `pip install -e /home/xuewenyao/code/FastWAM[serving]`（在 FastWAM 根目录可写 `pip install -e .[serving]`，会带上 `tyro` / `websockets` / `msgpack`）

在 **FastWAM 仓库根目录** 下使用与 OpenPI 相同风格的子命令（默认端口 **8000**）。

**有 uv 时：**

```bash
cd /home/xuewenyao/code/FastWAM
uv sync --extra serving
pip install -e /home/xuewenyao/code/openpi/packages/openpi-client
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=my_robot_uncond_2cam224_ft \
  --policy.dir=/home/datasets_v2/FastWAM/my_robot_uncond_2cam224_ft/2026-04-07_manual
```

**没有 uv 时（用当前环境的 `python`）：** 先按上面装好依赖，再在仓库根目录执行：

```bash
cd /home/xuewenyao/code/FastWAM
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=my_robot_uncond_2cam224_ft \
  --policy.dir=/home/datasets_v2/FastWAM/my_robot_uncond_2cam224_ft/2026-04-07_manual/checkpoints/weights/step_020000.pt

# 客户端模拟
python /home/xuewenyao/code/FastWAM/scripts/mock_openpi_client.py \
  --host 127.0.0.1 \
  --port 8090 \
  --num-rounds 5

```

`serve_policy.py` 会自动把 `src/` 加入 `sys.path`，一般无需再设 `PYTHONPATH`。若你**没有**执行可编辑安装、只靠拷贝代码，则需：

`PYTHONPATH=/home/xuewenyao/code/FastWAM/src python /home/xuewenyao/code/FastWAM/scripts/serve_policy.py policy:checkpoint ...`

（并自行保证已安装 FastWAM 的 `pyproject.toml` 主依赖与 `tyro` / `websockets` / `msgpack`。）

- `--policy.dir`：可为含 `weights/step_*.pt` 的训练输出目录，或直接指向某个 `.pt`；会在目录及上级查找 `dataset_stats.json`。
- 若省略 `--policy.config`，会尝试从路径 `.../runs/<task>/...` 或运行目录旁的 Hydra `config.yaml` / `.hydra/config.yaml` 推断任务名。
- 指定端口：`python scripts/serve_policy.py --port=8010 policy:checkpoint --policy.dir=...`（有 uv 时把 `python` 换成 `uv run scripts/serve_policy.py` 即可）。
- RoboTwin 三相机权重：加 `--hydra-config-name=sim_robotwin`、`--policy.config=robotwin_uncond_3cam_384_1e-4`、`--image-layout=robotwin`。

实现集中在单文件 `src/fastwam/serving/openpi_serve.py`（解析路径、策略、WebSocket 服务）。客户端使用 `openpi_client.WebsocketClientPolicy`。

