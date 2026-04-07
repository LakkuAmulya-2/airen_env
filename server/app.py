# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""FastAPI application for AIREN environment."""

import os
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_ENV_DIR = _SERVER_DIR.parent
_REPO_ROOT = _ENV_DIR.parents[1]
for _p in [str(_REPO_ROOT / "src"), str(_ENV_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openenv.core.env_server.http_server import create_app

try:
    from ..models import AIRENAction, AIRENObservation
    from .airen_environment import AIRENEnvironment
except ImportError:
    from models import AIRENAction, AIRENObservation
    from server.airen_environment import AIRENEnvironment

app = create_app(
    AIRENEnvironment,
    AIRENAction,
    AIRENObservation,
    env_name="airen_env",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    main(host=args.host, port=args.port)
