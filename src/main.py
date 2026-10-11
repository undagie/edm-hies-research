"""CLI entry point for the EDM / HIES tabular benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mlflow

from src.config import load_config
from src.reporting import generate_figures, sync_paper_results
from src.runner import finalize_mlflow, run_benchmark
from src.runner.experiment import save_results
from src.tracking import build_run_manifest, make_run_id, setup_run_logger, write_json
from src.utils import silence_known_warnings

silence_known_warnings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDM tabular benchmark.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--allow-synthetic-for-dev",
        action="store_true",
        help="Allow synthetic dataset fallback when raw CSV files are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.allow_synthetic_for_dev:
        cfg.data["use_synthetic_if_missing"] = True
    out_root = Path(cfg.outputs["root"])
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id()
    logger = setup_run_logger(run_id, cfg.outputs["logs_dir"])
    logger.info("Run started | run_id=%s", run_id)
    logger.info("Config loaded from %s", args.config)

    if args.report_only:
        import pandas as pd

        from src.reporting.best_per_dataset import select_best_per_dataset

        summary = pd.read_csv(out_root / "results" / "results_summary.csv")
        raw = pd.read_csv(out_root / "results" / "results_raw.csv")
        logger.info("Report-only mode using existing results files.")
        select_best_per_dataset(summary).to_csv(out_root / "tables" / "table_best_per_dataset.csv", index=False)
    else:
        raw, summary = run_benchmark(cfg, run_id=run_id, logger=logger)
        save_results(
            raw,
            summary,
            out_root,
            run_id=run_id,
            logger=logger,
            optimization=getattr(cfg, "optimization", None),
        )

    try:
        generate_figures(summary, cfg.outputs["figures_dir"], out_root=out_root, run_id=run_id)
        sync_paper_results(summary, cfg.outputs["paper_file"])
    finally:
        finalize_mlflow(logger=logger)
    manifest = build_run_manifest(
        run_id=run_id,
        args=vars(args),
        config_path=args.config,
        config_payload=asdict(cfg),
        raw_data_dir=cfg.data["raw_dir"],
        output_root=out_root,
    )
    manifest["outputs"] = {
        "results_raw": str((out_root / "results" / "results_raw.csv").as_posix()),
        "results_summary": str((out_root / "results" / "results_summary.csv").as_posix()),
        "table_main_results": str((out_root / "tables" / "table_main_results.csv").as_posix()),
        "table_fairness_results": str((out_root / "tables" / "table_fairness_results.csv").as_posix()),
        "table_ablation": str((out_root / "tables" / "table_ablation.csv").as_posix()),
        "table_significance": str((out_root / "tables" / "table_significance.csv").as_posix()),
        "table_tradeoff": str((out_root / "tables" / "table_tradeoff.csv").as_posix()),
        "table_guardrail": str((out_root / "tables" / "table_guardrail.csv").as_posix()),
        "table_best_per_dataset": str((out_root / "tables" / "table_best_per_dataset.csv").as_posix()),
        "figure_manifest": str((out_root / "manifests" / "figure_manifest.csv").as_posix()),
        "table_manifest": str((out_root / "manifests" / "table_manifest.csv").as_posix()),
        "text_manifest": str((out_root / "manifests" / "text_manifest.csv").as_posix()),
        "paper_file": str(Path(cfg.outputs["paper_file"]).as_posix()),
    }
    manifest["mlflow"] = {
        "tracking_uri": mlflow.get_tracking_uri(),
    }
    manifest_path = out_root / "manifests" / f"run_manifest_{run_id}.json"
    write_json(manifest_path, manifest)
    repro_bundle_path = out_root / "manifests" / f"repro_bundle_{run_id}.md"
    repro_bundle_content = "\n".join(
        [
            f"# Repro Bundle {run_id}",
            "",
            "## Core Files",
            f"- {manifest['outputs']['results_raw']}",
            f"- {manifest['outputs']['results_summary']}",
            f"- {manifest['outputs']['table_main_results']}",
            f"- {manifest['outputs']['table_fairness_results']}",
            f"- {manifest['outputs']['table_ablation']}",
            f"- {manifest['outputs']['table_significance']}",
            f"- {manifest['outputs']['table_tradeoff']}",
            f"- {manifest['outputs']['table_guardrail']}",
            f"- {manifest['outputs']['table_best_per_dataset']}",
            f"- {manifest['outputs']['figure_manifest']}",
            f"- {manifest['outputs']['table_manifest']}",
            f"- {manifest['outputs']['text_manifest']}",
            f"- {manifest['outputs']['paper_file']}",
            "",
            f"Run manifest: {manifest_path.as_posix()}",
            f"MLflow tracking URI: {mlflow.get_tracking_uri()}",
            "",
        ]
    )
    repro_bundle_path.write_text(repro_bundle_content, encoding="utf-8")
    logger.info("Run manifest saved at %s", manifest_path.as_posix())
    logger.info("Repro bundle saved at %s", repro_bundle_path.as_posix())
    logger.info("Run finished | run_id=%s", run_id)


if __name__ == "__main__":
    main()

