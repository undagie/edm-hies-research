"""Report the best method per dataset from an existing table_main_results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.reporting.best_per_dataset import select_best_per_dataset


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select best model per dataset (by macro-F1, with tie-breakers).",
    )
    p.add_argument("--config", default="configs/default.yaml", help="Config for default artifact paths")
    p.add_argument(
        "--input-csv",
        default=None,
        help="Override path to main results table (default: <outputs.root>/tables/table_main_results.csv)",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Override output path (default: <outputs.root>/tables/table_best_per_dataset.csv)",
    )
    p.add_argument(
        "--include-dummy",
        action="store_true",
        help="Allow dummy_most_frequent_baseline to win (default: excluded)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    root = Path(cfg.outputs["root"])
    in_path = Path(args.input_csv) if args.input_csv else root / "tables" / "table_main_results.csv"
    out_path = Path(args.output_csv) if args.output_csv else root / "tables" / "table_best_per_dataset.csv"

    summary = pd.read_csv(in_path)
    exclude = () if args.include_dummy else ("dummy_most_frequent_baseline",)
    best = select_best_per_dataset(summary, exclude_methods=exclude)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.to_csv(out_path, index=False)

    print("Best method per dataset (excl. dummy unless --include-dummy):")
    for _, row in best.iterrows():
        js = row.get("joint_score_mean", float("nan"))
        js_part = f"joint={float(js):.4f}" if pd.notna(js) else "joint=n/a"
        print(
            f"  {row['dataset']}: {row['method']} | "
            f"macro-F1={row['f1_macro_mean']:.4f} | "
            f"{js_part}"
        )
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
