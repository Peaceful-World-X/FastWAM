##  FastWAM（OpenPI 协议 WebSocket 推理，仓库：`~/code/FastWAM`）

FastWAM 为 **PyTorch**，不需要设置 `XLA_PYTHON_CLIENT_*`。可选显存相关：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

依赖（任选其一装全即可）：

- `pip install -e /home/xuewenyao/code/openpi/packages/openpi-client`
- `pip install -e /home/xuewenyao/code/FastWAM[serving]`（在 FastWAM 根目录可写 `pip install -e .[serving]`，会带上 `tyro` / `websockets` / `msgpack`）

在 **FastWAM 仓库根目录** 下使用与 OpenPI 相同风格的子命令（默认端口 **8000**）。

**有 uv 时：**

```bash

# 法2
docker pull docker.1ms.run/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
docker run -it \
  --name FastWAM_xwy \
  --init \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --shm-size=64G \
  --network host \
  -v /home/xuewenyao/code:/home/xuewenyao/code \
  -v /home/models:/home/models \
  -v /home/pub_envs:/home/pub_envs \
  -v /home/results:/home/results \
  -v /home/datasets:/home/datasets \
  -v /home/datasets_v2:/home/datasets_v2 \
  -v /home/datasets_2:/home/datasets_2 \
  docker.1ms.run/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

conda create -n fastwam --clone base -y
conda activate fastwam
conda install python=3.10 -y
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .

docker exec -u 0 -it FastWAM_xwy /bin/bash 
cd /home/xuewenyao/code
conda activate fastwam








# 3. 预计算文本 embedding（训练前必跑）
cd /home/xuewenyao/code/FastWAM
python scripts/precompute_text_embeds.py task=my_robot_uncond_2cam224_ft
# 启动训练
export DIFFSYNTH_MODEL_BASE_PATH=/home/models/FastWAM/checkpoints
export FASTWAM_OUTPUT_DIR=/home/datasets_v2/FastWAM/manual_run_001

cd /home/xuewenyao/code/FastWAM
bash scripts/train_zero1.sh 8 task=my_robot_uncond_2cam224_ft \
  output_dir=/home/datasets_v2/FastWAM/my_robot_uncond_2cam224_ft/2026-04-15_manual

```

```bash
cd /home/xuewenyao/code/FastWAM
uv sync --extra serving
pip install -e /home/xuewenyao/code/openpi/packages/openpi-client
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=my_robot_uncond_2cam224_ft \
  --image-layout=openpi_3cam_ee \
  --policy.dir=/home/datasets_v2/FastWAM/my_robot_uncond_2cam224_ft/2026-04-07_manual
```

**没有 uv 时（用当前环境的 `python`）：** 先按上面装好依赖，再在仓库根目录执行：








```bash
# 2 相机 + 关节（OpenPI 两路图 + state）
cd /home/xuewenyao/code/FastWAM
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=my_robot_uncond_2cam224_ft \
  --image-layout=openpi_2cam_joint \
  --policy.dir=/home/datasets_v2/FastWAM/my_robot_uncond_2cam224_ft/2026-04-07_manual/checkpoints/weights/step_050000.pt \
  --port=8090

# 三相机 + 末端位姿（observation/top_image 等，见 openpi_serve 文档）
# 默认每条动作序列长度为 num_frames-1（如 32），推理较慢时可加 --action-horizon=16，输出变为 (16,7)
cd /home/xuewenyao/code/FastWAM
python scripts/serve_policy.py policy:checkpoint \
  --policy.config=my_robot_uncond_2cam224_ft \
  --image-layout=openpi_3cam_ee \
  --action-horizon=16 \
  --policy.dir=/home/datasets_v2/FastWAM/my_robot_uncond_2cam224_ft/2026-04-15_manual/checkpoints/weights/step_050000.pt \
  --port=8090

# Match server: --image-layout=openpi_2cam_joint
python scripts/mock_openpi_client.py --host 127.0.0.1 --port 8090 --image-layout openpi_2cam_joint

# Match server: --image-layout=openpi_3cam_ee
python scripts/mock_openpi_client.py --host 127.0.0.1 --port 8090 --image-layout openpi_3cam_ee --state-dim 7

```

**动作序列长度（`actions` 形状为 `(action_horizon, 7)`）**

- 默认：与训练一致，一般为 `data.train.num_frames - 1`（例如 33 帧 → **32** 步），即 `(32, 7)`。
- 加速：启动时加 **`--action-horizon=16`**，得到 **`(16, 7)`**（`serve_policy.py` 已支持，会传给 `FastWAMOpenPIBridgePolicy`）。
- 或在对应 task 的 Hydra 配置里增加 `EVALUATION.action_horizon: 16`，则不必每次写 CLI（未传 `--action-horizon` 时从配置读取）。

`serve_policy.py` 会在解析前把 ``--port`` / ``--image-layout`` 等非 ``--policy.*`` 参数挪到 ``policy:checkpoint`` 之前，避免 Tyro 报 “Unrecognized options”。也可手动写成：``--image-layout=openpi_3cam_ee --port=8090 policy:checkpoint --policy.config=... --policy.dir=...``。

```bash
# 客户端模拟（--image-layout 须与上面服务一致）
python /home/xuewenyao/code/FastWAM/scripts/mock_openpi_client.py \
  --host 127.0.0.1 \
  --port 8090 \
  --image-layout openpi_2cam_joint \
  --num-rounds 5

```


