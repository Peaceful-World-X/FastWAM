#!/usr/bin/env python3
"""Mock OpenPI WebSocket client: send N random observations and log server replies.

Requires: pip install -e /path/to/openpi/packages/openpi-client

Example (server on host port 8090):
  python /home/xuewenyao/code/FastWAM/scripts/mock_openpi_client.py \\
    --host 127.0.0.1 --port 8090 --num-rounds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from openpi_client import websocket_client_policy


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logger = logging.getLogger("mock_openpi_client")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (not 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8090, help="WebSocket port (match serve_policy.py).")
    parser.add_argument("--num-rounds", type=int, default=5, help="Number of infer calls.")
    parser.add_argument("--action-dim", type=int, default=8, help="State vector length (training dim).")
    parser.add_argument("--h", type=int, default=224, help="Per-camera height (pixels).")
    parser.add_argument("--w", type=int, default=224, help="Per-camera width (pixels).")
    parser.add_argument(
        "--prompt",
        type=str,
        default="mock pick-and-place",
        help="Instruction string sent as prompt.",
    )
    args = parser.parse_args()

    logger.info("Connecting to ws://%s:%s ...", args.host, args.port)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    meta = policy.get_server_metadata()
    logger.info("Server metadata: %s", meta)

    for i in range(args.num_rounds):
        obs = {
            "observation/image": np.random.randint(0, 255, size=(args.h, args.w, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, size=(args.h, args.w, 3), dtype=np.uint8),
            "state": np.random.randn(args.action_dim).astype(np.float32),
            "prompt": f"{args.prompt} (round {i + 1})",
        }
        t0 = time.perf_counter()
        try:
            out = policy.infer(obs)
        except Exception:
            logger.exception("infer failed on round %d", i + 1)
            sys.exit(1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        actions = out.get("actions")
        st = out.get("server_timing", {})
        pt = out.get("policy_timing", {})
        logger.info(
            "Round %d/%d | client_wall_ms=%.1f | server_timing=%s | policy_timing=%s",
            i + 1,
            args.num_rounds,
            elapsed_ms,
            st,
            pt,
        )
        if actions is not None:
            logger.info(
                "  actions shape=%s dtype=%s | first_row=%s",
                actions.shape,
                actions.dtype,
                actions[0] if getattr(actions, "ndim", 0) == 2 else actions,
            )
        else:
            logger.info("  (no 'actions' key) out keys=%s", list(out.keys()))

    logger.info("Done.")


if __name__ == "__main__":
    main()
