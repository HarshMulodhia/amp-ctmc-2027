#!/usr/bin/env python3
"""Download catalog entries only after the source SHA-256 has been pinned."""

import argparse
from pathlib import Path

import yaml

from amp_ctmc_2027.data.sources import download_verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", type=Path, default=Path("configs/data_sources.yaml")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    catalog = yaml.safe_load(args.catalog.read_text(encoding="utf-8"))
    for source in catalog["sources"]:
        path = args.data_dir / source["expected_filename"]
        digest = download_verified(source["url"], path, source["sha256"])
        print(f"{source['name']} {digest} {path}")


if __name__ == "__main__":
    main()
