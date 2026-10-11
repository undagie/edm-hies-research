"""Reproduce main benchmark + Phase-1 analyses + Friedman/CD + SHAP.

Usage: python scripts/reproduce_full.py [--skip-benchmark]

``--skip-benchmark`` reuses existing ``artifacts/results/results_raw.csv``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(name: str, cmd: list[str]) -> None:
    """Run a timed subprocess step; exit nonzero on failure."""
    print(f"\n[step] {name}")
    print(f"       {' '.join(cmd)}")
    t0 = time.perf_counter()
    # So subprocesses resolve `src.*` regardless of CWD.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise SystemExit(
            f"[step] {name} FAILED with exit code {result.returncode} "
            f"after {elapsed:.1f}s"
        )
    print(f"[step] {name} OK ({elapsed:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Re-run only the downstream analyses against an existing "
        "artifacts/results/results_raw.csv",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Benchmark config (default: configs/default.yaml).",
    )
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_benchmark:
        _run(
            "main benchmark (11 methods x 3 datasets x 60 fold samples)",
            [py, "-m", "src.main", "--config", args.config],
        )
    else:
        print("[skip] main benchmark — using existing artifacts/results/")

    _run(
        "Phase-1 analyses (TOST, subgroup metrics, calibration, Pareto, boxplot)",
        [py, "scripts/analyze_phase1.py"],
    )

    _run(
        "Friedman omnibus + Nemenyi post-hoc + Critical Difference diagrams",
        [py, "scripts/analyze_friedman_nemenyi.py"],
    )

    _run(
        "SHAP top-15 + per-group SHAP for all three datasets",
        [py, "scripts/run_shap_hies.py"],
    )

    print("\nDone. Paper-ready outputs are under artifacts/ and "
          "artifacts/phase1_analysis/.")


if __name__ == "__main__":
    main()
