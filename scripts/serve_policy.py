#!/usr/bin/env python3
"""FastWAM OpenPI-style WebSocket policy server (tyro CLI, same spirit as OpenPI ``serve_policy.py``).

Install (example):
  pip install -e /path/to/openpi/packages/openpi-client
  pip install -e /path/to/FastWAM[serving]

Typical launch (aligned with OpenPI ``policy:checkpoint`` + ``--policy.config`` + ``--policy.dir``)::

  # PyTorch / FastWAM: XLA_* env vars apply to JAX (OpenPI) only, not required here.
  # Optional CUDA fragment tuning, e.g.:
  # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  uv run scripts/serve_policy.py policy:checkpoint \\
    --policy.config=my_robot_uncond_2cam224_ft \\
    --policy.dir=/path/to/run_or_weights_dir

``--policy.dir`` may be:
  - a directory containing ``weights/step_*.pt`` and ``dataset_stats.json`` (or stats in a parent), or
  - a direct path to ``*.pt`` (stats resolved upward).

If ``--policy.config`` is omitted, the task name is inferred from ``.../runs/<task>/...`` or from
``config.yaml`` / ``.hydra/config.yaml`` saved next to the run.

RoboTwin-style checkpoint (three cameras)::

  uv run scripts/serve_policy.py policy:checkpoint \\
    --hydra-config-name=sim_robotwin \\
    --policy.config=robotwin_uncond_3cam_384_1e-4 \\
    --image-layout=robotwin \\
    --policy.dir=/path/to/run_with_weights

Use port 8010::

  uv run scripts/serve_policy.py --port=8010 policy:checkpoint \\
    --policy.dir=/path/to/checkpoint/step_50000
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import sys
from pathlib import Path

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from fastwam.serving.openpi_serve import launch_from_policy_dir


@dataclasses.dataclass
class Checkpoint:
    """Load FastWAM weights + dataset stats from a training run directory or ``.pt`` file."""

    # Hydra task name (e.g. ``my_robot_uncond_2cam224_ft``). If None, inferred from path or saved config.
    config: str | None = None
    # Run root (contains ``weights/``) or path to a single ``.pt`` checkpoint.
    dir: str = ""


@dataclasses.dataclass
class Default:
    """FastWAM does not ship a default public checkpoint; use ``policy:checkpoint``."""


@dataclasses.dataclass
class Args:
    """Arguments for the FastWAM serve_policy script (OpenPI-like)."""

    # How to load weights (same subcommand style as OpenPI: ``policy:checkpoint``).
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    # WebSocket bind address (OpenPI defaults to 8000).
    port: int = 8090
    host: str = "0.0.0.0"

    device: str = "cuda"
    mixed_precision: str = "bf16"

    # Top-level Hydra YAML under ``configs/`` (``train`` for most fine-tunes; ``sim_robotwin`` for RoboTwin eval stack).
    hydra_config_name: str = "train"
    # Extra Hydra overrides (repeatable), e.g. ``data.train.dataset_dirs=...``.
    hydra_override: list[str] = dataclasses.field(default_factory=list)

    image_layout: str = "auto"
    action_horizon: int | None = None
    num_inference_steps: int | None = None
    sigma_shift: float | None = None
    seed: int | None = None
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    rand_device: str = "cpu"
    tiled: bool = False

    # Defaults to ``<repo>/configs`` when None.
    config_dir: Path | None = None


def main(args: Args) -> None:
    ck: Checkpoint
    match args.policy:
        case Default():
            raise SystemExit(
                "FastWAM: specify policy:checkpoint with --policy.dir=... "
                "(and usually --policy.config=<hydra_task_name>)."
            )
        case Checkpoint() as ch:
            ck = ch
            if not str(ck.dir).strip():
                raise SystemExit("policy.dir is required when using policy:checkpoint")
        case _:
            raise SystemExit("Unsupported policy subcommand.")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    cdir = args.config_dir if args.config_dir is not None else PROJECT_ROOT / "configs"
    cdir = cdir.resolve()

    launch_from_policy_dir(
        policy_dir=ck.dir,
        policy_config=ck.config,
        config_dir=cdir,
        hydra_config_name=args.hydra_config_name,
        extra_hydra_overrides=args.hydra_override,
        host=args.host,
        port=args.port,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        text_cfg_scale=args.text_cfg_scale,
        negative_prompt=args.negative_prompt,
        rand_device=args.rand_device,
        tiled=args.tiled,
        image_layout=args.image_layout,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s,%(msecs)03d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(tyro.cli(Args))
