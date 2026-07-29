#!/usr/bin/env python3
"""One-command correctness test.

Generates reference data, builds the CUDA binary, and runs correctness checks.

Usage:
    python test_correctness.py              # auto-detect GPU arch
    python test_correctness.py --arch sm_86 # explicit arch
"""

import argparse
import subprocess
import sys
import os


def run(cmd, **kwargs):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, **kwargs)
    if r.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(r.returncode)
    return r


def detect_arch():
    """Try to detect GPU compute capability."""
    try:
        r = subprocess.run(
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader",
            shell=True, capture_output=True, text=True
        )
        if r.returncode == 0:
            cap = r.stdout.strip().split("\n")[0].strip()
            arch = f"sm_{cap.replace('.', '')}"
            print(f"Detected GPU arch: {arch}")
            return arch
    except Exception:
        pass
    print("Could not detect GPU arch, defaulting to sm_80")
    return "sm_80"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default=None)
    args = parser.parse_args()

    arch = args.arch or detect_arch()

    print("=== Step 1: Generate PyTorch reference data ===")
    run("python3 generate_reference.py")

    print("\n=== Step 2: Build CUDA kernels ===")
    run(f"make clean && make ARCH={arch}")

    print("\n=== Step 3: Run correctness tests ===")
    run("./attention test test_data")

    print("\nDone.")


if __name__ == "__main__":
    main()
