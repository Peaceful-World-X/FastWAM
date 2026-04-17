# OpenPI wire protocol reference: Physical Intelligence websocket_policy_server (msgpack+numpy).
"""OpenPI-compatible WebSocket serving for FastWAM (single module).

Includes: checkpoint path resolution, ``FastWAMOpenPIBridgePolicy``, ``WebsocketPolicyServer``,
and ``launch_from_policy_dir`` / ``launch_openpi_websocket_server``.

Client observation keys depend on ``serve_policy.py`` ``--image-layout``: **two cams + joints**
(``openpi_2cam_joint``) uses OpenPI defaults ``observation/image``, ``observation/wrist_image``,
``state`` (joints), ``prompt``. **Three cams + EE** (``openpi_3cam_ee``) uses
``observation/top_image``, ``observation/front_image``, ``observation/right_wrist_image``, plus
``state`` or ``observation/state`` (7D EE+grip), and ``prompt`` (or ``observation/prompt``).
Nested dict ``{"observation": {"image": ..., "state": ...}}`` matches flat ``observation/*`` keys.
RoboTwin: ``image_layout=robotwin``. Legacy aliases (``img/*``, ``obs/ee`` + ``obs/grip``) still work
for 3-cam. See ``FastWAMOpenPIBridgePolicy.infer``.
"""

from __future__ import annotations

import asyncio
import http
import inspect
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from openpi_client import base_policy as _openpi_base_policy
from openpi_client import msgpack_numpy
from PIL import Image
from typing_extensions import override
import websockets.asyncio.server as ws_server
import websockets.frames

from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

logger = logging.getLogger(__name__)


# --- checkpoint resolution -------------------------------------------------

def _parse_step_number(path: Path) -> int:
    m = re.match(r"step_(\d+)", path.stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return path.stat().st_mtime_ns


def _is_non_model_pt(path: Path) -> bool:
    """Skip DeepSpeed / Accelerate noise (optim states, etc.) when scanning trees."""
    name = path.name.lower()
    if "optim_states" in name or name.endswith("optim_states.pt"):
        return True
    if "rng_state" in name or "scheduler" in name and path.suffix.lower() == ".pt":
        return True
    parts = [p.lower() for p in path.parts]
    if "pytorch_model" in parts and "optim" in name:
        return True
    return False


def _collect_step_weights_pt(search_dir: Path) -> list[Path]:
    if not search_dir.is_dir():
        return []
    return sorted(search_dir.glob("step_*.pt"), key=_parse_step_number)


def resolve_checkpoint_paths(policy_dir: str | Path) -> tuple[Path, Path]:
    root = Path(policy_dir).expanduser().resolve()
    if root.is_file() and root.suffix.lower() == ".pt":
        ckpt = root
        search_roots = [ckpt.parent, *ckpt.parents][:10]
    elif root.is_dir():
        ckpt: Path | None = None
        # Trainer writes FastWAM payloads to ``<checkpoint_root>/weights/step_*.pt``;
        # ``checkpoint_root`` is often ``<run_dir>/checkpoints`` (not ``<run_dir>/weights``).
        weight_dirs = [
            root / "weights",
            root / "checkpoints" / "weights",
        ]
        for wd in weight_dirs:
            step_pts = _collect_step_weights_pt(wd)
            if step_pts:
                ckpt = step_pts[-1]
                break
        if ckpt is None:
            for wd in weight_dirs:
                if wd.is_dir():
                    any_pt = sorted(wd.glob("*.pt"), key=lambda p: p.stat().st_mtime_ns)
                    any_pt = [p for p in any_pt if not _is_non_model_pt(p)]
                    if any_pt:
                        ckpt = any_pt[-1]
                        logger.warning("No step_*.pt under %s; using: %s", wd, ckpt)
                        break
        if ckpt is None:
            candidates = [p for p in root.rglob("*.pt") if not _is_non_model_pt(p)]
            step_like = [p for p in candidates if re.match(r"step_\d+", p.stem, re.I)]
            if step_like:
                ckpt = sorted(step_like, key=_parse_step_number)[-1]
                logger.info("Resolved checkpoint from tree: %s", ckpt)
            elif candidates:
                ckpt = sorted(candidates, key=lambda p: p.stat().st_mtime_ns)[-1]
                logger.warning("No ideal step_*.pt; using newest filtered .pt: %s", ckpt)
            else:
                raise FileNotFoundError(
                    f"No loadable .pt under {root} (DeepSpeed optim_states files are ignored)."
                )
        search_roots = [root, *root.parents][:10]
    else:
        raise FileNotFoundError(f"policy.dir is not a file or directory: {root}")

    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint path is not a file: {ckpt}")

    stats: Path | None = None
    for d in search_roots:
        cand = d / "dataset_stats.json"
        if cand.is_file():
            stats = cand
            break
    if stats is None:
        raise FileNotFoundError(
            f"dataset_stats.json not found near checkpoint (searched from {ckpt.parent})."
        )
    return ckpt, stats


def _infer_task_from_path(ckpt: Path) -> str | None:
    parts = ckpt.resolve().parts
    if "runs" in parts:
        i = parts.index("runs")
        if i + 1 < len(parts):
            return parts[i + 1]
    return None


def _infer_task_from_hydra_yaml(ckpt: Path) -> str | None:
    for d in [ckpt.parent, *ckpt.parents][:12]:
        for rel in (Path(".hydra") / "config.yaml", Path("config.yaml")):
            p = d / rel
            if not p.is_file():
                continue
            try:
                cfg = OmegaConf.load(p)
            except Exception:
                continue
            t = cfg.get("task")
            if t is not None and str(t).strip():
                return str(t)
            try:
                ch = OmegaConf.select(cfg, "hydra.runtime.choices.task")
            except Exception:
                ch = None
            if ch is not None and str(ch).strip():
                return str(ch)
    return None


def resolve_hydra_task_name(*, policy_config: str | None, checkpoint_path: Path) -> str:
    if policy_config is not None and str(policy_config).strip():
        return str(policy_config).strip()
    t = _infer_task_from_path(checkpoint_path)
    if t:
        return t
    t = _infer_task_from_hydra_yaml(checkpoint_path)
    if t:
        logger.info("Inferred Hydra task=%s from config near %s", t, checkpoint_path)
        return t
    raise ValueError(
        "Could not infer Hydra task. Pass --policy.config=<task_name> "
        "(e.g. my_robot_uncond_2cam224_ft), or use .../runs/<task>/..., or save Hydra config.yaml."
    )


# --- WebSocket server (OpenPI protocol) -------------------------------------

def _health_check(
    connection: ws_server.ServerConnection, request: ws_server.Request
) -> ws_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


class WebsocketPolicyServer:
    """First message: metadata (msgpack). Then recv obs -> infer -> send actions (msgpack)."""

    def __init__(
        self,
        policy: _openpi_base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        if not isinstance(policy, _openpi_base_policy.BasePolicy):
            raise TypeError(f"policy must inherit BasePolicy, got {type(policy)}")
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            ping_timeout=60,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.perf_counter()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                infer_start = time.perf_counter()
                action = self._policy.infer(obs)
                infer_ms = (time.perf_counter() - infer_start) * 1000
                client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
                pms = action.get("policy_timing", {}).get("infer_ms")
                if pms is not None:
                    logger.info("[%s] Infer: %.2f ms (server) %.2f ms (model)", client_ip, infer_ms, pms)
                else:
                    logger.info("[%s] Infer: %.2f ms", client_ip, infer_ms)
                action["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                await websocket.send(packer.pack(action))
                prev_total_time = time.perf_counter() - start_time
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


# --- Policy bridge ----------------------------------------------------------

def _parse_image_to_uint8_hwc(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HxWx3 image, got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _lookup(obs: dict[str, Any], key: str) -> Any:
    if key in obs:
        return obs[key]
    parts = key.split("/")
    cur: Any = obs
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _decode_prompt(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


class FastWAMOpenPIBridgePolicy(_openpi_base_policy.BasePolicy):
    """Hydra-loaded FastWAM with OpenPI-style ``infer(obs)``."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        checkpoint_path: Path,
        dataset_stats_path: Path,
        device: str,
        mixed_precision: str,
        action_horizon: int | None,
        num_inference_steps: int | None,
        sigma_shift: float | None,
        seed: int | None,
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        image_layout: str | None = None,
    ) -> None:
        super().__init__()
        mp = str(mixed_precision).strip().lower()
        if mp not in {"no", "fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")
        model_dtype = torch.float32 if mp == "no" else (torch.float16 if mp == "fp16" else torch.bfloat16)

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        model_cfg.load_text_encoder = True
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(str(checkpoint_path))
        self.model = self.model.to(device).eval()

        proc_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train.processor, resolve=True))
        self.processor: FastWAMProcessor = instantiate(proc_cfg).eval()
        self.processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(dataset_stats_path))
        )

        train = cfg.data.train
        cmc = str(train.get("concat_multi_camera", "horizontal") or "horizontal")
        self._concat_mode = image_layout if image_layout not in (None, "auto") else cmc
        nf = int(train.num_frames)
        avfr = int(train.action_video_freq_ratio)
        self._num_video_frames = (nf - 1) // avfr + 1

        eval_block = cfg.get("EVALUATION")
        if action_horizon is None:
            ah = None
            if eval_block is not None and eval_block.get("action_horizon") is not None:
                ah = int(eval_block.get("action_horizon"))
            self._action_horizon = ah if ah is not None else nf - 1
        else:
            self._action_horizon = int(action_horizon)

        if num_inference_steps is None:
            if eval_block is not None and eval_block.get("num_inference_steps") is not None:
                self._num_inference_steps = int(eval_block.get("num_inference_steps"))
            else:
                self._num_inference_steps = int(cfg.get("eval_num_inference_steps", 10))
        else:
            self._num_inference_steps = int(num_inference_steps)

        if sigma_shift is None and eval_block is not None:
            ss = eval_block.get("sigma_shift")
            self._sigma_shift = float(ss) if ss is not None else None
        else:
            self._sigma_shift = float(sigma_shift) if sigma_shift is not None else None

        if seed is None and eval_block is not None:
            s = eval_block.get("seed")
            self._default_seed = int(s) if s is not None else None
        else:
            self._default_seed = int(seed) if seed is not None else None

        self._text_cfg_scale = float(
            text_cfg_scale if eval_block is None else eval_block.get("text_cfg_scale", text_cfg_scale)
        )
        self._negative_prompt = str(
            negative_prompt if eval_block is None else eval_block.get("negative_prompt", negative_prompt)
        )
        self._rand_device = str(
            rand_device if eval_block is None else eval_block.get("rand_device", rand_device)
        )
        self._tiled = bool(tiled if eval_block is None else eval_block.get("tiled", tiled))

        vs = train.get("video_size")
        self._resize = ResizeSmallestSideAspectPreserving(
            args={"img_w": int(vs[1]), "img_h": int(vs[0])}
        )
        self._crop = CenterCrop(args={"img_w": int(vs[1]), "img_h": int(vs[0])})
        self._normalize = Normalize(args={"mean": 0.5, "std": 0.5})

        logger.info(
            "FastWAM OpenPI bridge | ckpt=%s | horizon=%d | concat=%s",
            checkpoint_path,
            self._action_horizon,
            self._concat_mode,
        )

    @classmethod
    def from_hydra(
        cls,
        *,
        config_dir: Path,
        config_name: str,
        hydra_overrides: list[str],
        checkpoint_path: Path,
        dataset_stats_path: Path,
        device: str,
        mixed_precision: str,
        action_horizon: int | None,
        num_inference_steps: int | None,
        sigma_shift: float | None,
        seed: int | None,
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        image_layout: str | None,
    ) -> FastWAMOpenPIBridgePolicy:
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
            cfg = compose(config_name=config_name, overrides=list(hydra_overrides))
        return cls(
            cfg,
            checkpoint_path=checkpoint_path,
            dataset_stats_path=dataset_stats_path,
            device=device,
            mixed_precision=mixed_precision,
            action_horizon=action_horizon,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            text_cfg_scale=text_cfg_scale,
            negative_prompt=negative_prompt,
            rand_device=rand_device,
            tiled=tiled,
            image_layout=image_layout,
        )

    def _normalize_state_vector(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        vec = np.asarray(state, dtype=np.float32).reshape(-1).copy()
        state_batch = {"state": {state_key: torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()

    def _build_image_tensor_two_cam(self, obs: dict[str, Any]) -> torch.Tensor:
        keys0 = (
            "observation/image",
            "observation/exterior_image_1_left",
            "observation/exterior_image",
        )
        keys1 = (
            "observation/wrist_image",
            "observation/wrist_image_left",
            "observation/wrist_image_right",
        )
        im0 = next((v for k in keys0 if (v := _lookup(obs, k)) is not None), None)
        im1 = next((v for k in keys1 if (v := _lookup(obs, k)) is not None), None)
        if im0 is None or im1 is None:
            raise KeyError(
                f"Need two camera images; tried cam0={keys0} cam1={keys1}, keys={list(obs.keys())}"
            )
        a = _parse_image_to_uint8_hwc(im0)
        b = _parse_image_to_uint8_hwc(im1)
        axis = 0 if self._concat_mode == "vertical" else 1
        concat = np.concatenate([a, b], axis=axis).copy()
        x = torch.from_numpy(concat).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        x = self._resize(x)
        x = self._crop(x)
        x = self._normalize(x)
        return x.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _build_image_tensor_triple_cam(self, obs: dict[str, Any]) -> torch.Tensor:
        """Three RGB panels concatenated horizontally, matching ``concat_multi_camera: horizontal`` training.

        Preferred OpenPI-style keys (``--type=ee --camera=3``): ``observation/top_image``,
        ``observation/front_image``, ``observation/right_wrist_image`` (order matches
        ``configs/data/my_robot_lerobot.yaml``: top, front, right_wrist). Legacy aliases included.
        """
        key_groups = (
            ("observation/top_image", "observation/images/top_image", "img/top", "observation/image_top"),
            ("observation/front_image", "observation/images/front_image", "img/front", "observation/image_front"),
            (
                "observation/right_wrist_image",
                "observation/images/right_wrist_image",
                "observation/wrist_image",
                "img/side",
            ),
        )
        panels: list[np.ndarray] = []
        for aliases in key_groups:
            raw = next((v for k in aliases if (v := _lookup(obs, k)) is not None), None)
            if raw is None:
                raise KeyError(
                    f"triple_cam: need one of {aliases} for each panel; "
                    f"have top-level keys={list(obs.keys())!r}"
                )
            panels.append(_parse_image_to_uint8_hwc(raw))
        h0, w0 = panels[0].shape[0], panels[0].shape[1]
        for i in range(1, 3):
            if panels[i].shape[0] != h0 or panels[i].shape[1] != w0:
                logger.warning(
                    "triple_cam: panel %d shape %s != panel0 %s; resizing to match",
                    i,
                    panels[i].shape[:2],
                    (h0, w0),
                )
                panels[i] = _resize_rgb(panels[i], (w0, h0))
        concat = np.concatenate(panels, axis=1).copy()
        x = torch.from_numpy(concat).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        x = self._resize(x)
        x = self._crop(x)
        x = self._normalize(x)
        return x.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _build_image_tensor_robotwin(self, obs: dict[str, Any]) -> torch.Tensor:
        o = _lookup(obs, "observation")
        if not isinstance(o, dict):
            raise KeyError("Robotwin layout requires nested observation dict.")
        head = _resize_rgb(o["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(o["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(o["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        return image_tensor * (2.0 / 255.0) - 1.0

    def _extract_state(self, obs: dict[str, Any]) -> np.ndarray:
        # Prefer top-level ``state``; also accept ``observation/state`` (flat or nested ``observation``).
        st = _lookup(obs, "state")
        if st is None:
            st = _lookup(obs, "observation/state")
        if st is not None:
            return np.asarray(st, dtype=np.float32).reshape(-1)
        ee = _lookup(obs, "obs/ee")
        grip = _lookup(obs, "obs/grip")
        if ee is not None and grip is not None:
            ee_v = np.asarray(ee, dtype=np.float32).reshape(-1)
            g_v = np.asarray(grip, dtype=np.float32).reshape(-1)
            return np.concatenate([ee_v, g_v], axis=0)
        jp = _lookup(obs, "observation/joint_position")
        gp = _lookup(obs, "observation/gripper_position")
        if jp is not None and gp is not None:
            g = np.asarray(gp, dtype=np.float32).reshape(-1)
            j = np.asarray(jp, dtype=np.float32).reshape(-1)
            return np.concatenate([j, g], axis=0)
        raise KeyError(
            'Need "state" or "observation/state", or "obs/ee" + "obs/grip", or '
            "observation/joint_position + observation/gripper_position."
        )

    def _extract_instruction(self, obs: dict[str, Any]) -> str:
        for k in ("prompt", "instruction", "observation/prompt"):
            v = _lookup(obs, k)
            if v is not None:
                return _decode_prompt(v)
        return ""

    @override
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._concat_mode == "robotwin":
            image_tensor = self._build_image_tensor_robotwin(obs)
        elif self._concat_mode in ("triple_cam", "triple_horizontal", "openpi_3cam_ee"):
            image_tensor = self._build_image_tensor_triple_cam(obs)
        else:
            image_tensor = self._build_image_tensor_two_cam(obs)
        state_vec = self._extract_state(obs)
        proprio = self._normalize_state_vector(state_vec)
        instruction = self._extract_instruction(obs)
        prompt = (
            DEFAULT_PROMPT.format(task=instruction)
            if instruction
            else DEFAULT_PROMPT.format(task="")
        )
        seed = self._default_seed
        if "seed" in obs and obs["seed"] is not None:
            seed = int(obs["seed"])
        infer_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "input_image": image_tensor,
            "action_horizon": self._action_horizon,
            "proprio": proprio,
            "negative_prompt": self._negative_prompt,
            "text_cfg_scale": self._text_cfg_scale,
            "num_inference_steps": self._num_inference_steps,
            "sigma_shift": self._sigma_shift,
            "seed": seed,
            "rand_device": self._rand_device,
            "tiled": self._tiled,
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            infer_kwargs["num_video_frames"] = self._num_video_frames
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        actions = self._denormalize_action(pred["action"])[0].astype(np.float32)
        return {
            "actions": actions,
            "state": state_vec,
            "policy_timing": {"infer_ms": elapsed_ms},
        }

    @override
    def reset(self) -> None:
        pass


# --- launch -----------------------------------------------------------------


def _type_camera_hints_from_layout(layout: str | None) -> tuple[str | None, int | None]:
    """Optional hints for WebSocket metadata (mirrors serve_policy --image-layout names)."""
    s = (layout or "").strip()
    if s == "openpi_2cam_joint":
        return "joint", 2
    if s == "openpi_3cam_ee":
        return "ee", 3
    return None, None


def launch_openpi_websocket_server(
    *,
    checkpoint_path: Path,
    dataset_stats_path: Path,
    config_dir: Path,
    hydra_config_name: str,
    hydra_overrides: Sequence[str],
    host: str,
    port: int,
    device: str,
    mixed_precision: str,
    action_horizon: int | None,
    num_inference_steps: int | None,
    sigma_shift: float | None,
    seed: int | None,
    text_cfg_scale: float,
    negative_prompt: str,
    rand_device: str,
    tiled: bool,
    image_layout: str | None,
) -> None:
    layout = None if image_layout in (None, "auto") else image_layout
    policy = FastWAMOpenPIBridgePolicy.from_hydra(
        config_dir=config_dir,
        config_name=hydra_config_name,
        hydra_overrides=list(hydra_overrides),
        checkpoint_path=checkpoint_path,
        dataset_stats_path=dataset_stats_path,
        device=device,
        mixed_precision=mixed_precision,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        image_layout=layout,
    )
    raw_shape = policy.processor.shape_meta["action"][0]["shape"]
    adim = (
        int(raw_shape[-1] if len(raw_shape) > 1 else raw_shape[0])
        if isinstance(raw_shape, (list, tuple))
        else int(raw_shape)
    )
    metadata = {
        "policy_type": "fastwam",
        "action_horizon": policy._action_horizon,
        "action_dim": adim,
        "image_layout": policy._concat_mode,
        "checkpoint_path": str(checkpoint_path),
        "dataset_stats_path": str(dataset_stats_path),
        "observation_keys_doc": "fastwam.serving.openpi_serve.FastWAMOpenPIBridgePolicy.infer",
    }
    hint_t, hint_c = _type_camera_hints_from_layout(str(policy._concat_mode))
    if hint_t is not None:
        metadata["type"] = hint_t
    if hint_c is not None:
        metadata["camera"] = hint_c
    logger.info("Listening on ws://%s:%s", host, port)
    WebsocketPolicyServer(policy, host=host, port=port, metadata=metadata).serve_forever()


def launch_from_policy_dir(
    *,
    policy_dir: str | Path,
    policy_config: str | None,
    config_dir: Path,
    hydra_config_name: str,
    extra_hydra_overrides: Sequence[str],
    host: str,
    port: int,
    device: str,
    mixed_precision: str,
    action_horizon: int | None,
    num_inference_steps: int | None,
    sigma_shift: float | None,
    seed: int | None,
    text_cfg_scale: float,
    negative_prompt: str,
    rand_device: str,
    tiled: bool,
    image_layout: str | None,
) -> None:
    ckpt, stats = resolve_checkpoint_paths(policy_dir)
    task_name = resolve_hydra_task_name(policy_config=policy_config, checkpoint_path=ckpt)
    extra = list(extra_hydra_overrides)
    hydra_overrides = (
        extra
        if any(o.split("=", 1)[0].strip() == "task" for o in extra if "=" in o)
        else [f"task={task_name}", *extra]
    )
    logger.info("Checkpoint=%s dataset_stats=%s hydra=%s %s", ckpt, stats, hydra_config_name, hydra_overrides)
    launch_openpi_websocket_server(
        checkpoint_path=ckpt,
        dataset_stats_path=stats,
        config_dir=config_dir,
        hydra_config_name=hydra_config_name,
        hydra_overrides=hydra_overrides,
        host=host,
        port=port,
        device=device,
        mixed_precision=mixed_precision,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        image_layout=image_layout,
    )
