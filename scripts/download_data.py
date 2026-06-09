#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO = "zhangdw/GABench"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download migrated GABench data from Hugging Face.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset repo ID")
    parser.add_argument("--revision", help="Optional Hugging Face revision")
    parser.add_argument(
        "--target",
        default=Path(__file__).resolve().parent.parent,
        type=Path,
        help="Directory where benchmark/ and dataset/ should be restored",
    )
    args = parser.parse_args()

    if shutil.which("uvx") is None:
        print("Error: uvx is required. Install uv first: https://docs.astral.sh/uv/", file=sys.stderr)
        return 1

    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uvx",
        "--from",
        "huggingface_hub",
        "hf",
        "download",
        args.repo,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(target),
        "--include",
        "benchmark/**",
        "--include",
        "dataset/**",
    ]
    if args.revision:
        cmd.extend(["--revision", args.revision])

    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
