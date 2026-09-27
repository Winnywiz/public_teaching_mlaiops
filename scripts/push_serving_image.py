"""Push the locally built serving container and print its immutable ACR digest."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    cfg = config.load(strict=True)
    if cfg.provider != "azure":
        raise SystemExit("This push command expects the Azure Container Registry configured in cloud.env.")
    digest = get_adapter(cfg).push_image(f"itcs355-serve:{args.tag}")
    print(digest)
    print("Copy this full digest reference into SERVING_IMAGE_URI in the private cloud.env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
