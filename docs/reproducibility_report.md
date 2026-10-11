# Reproducibility Report

This report is generated and updated as part of final verification.

## Environment

- Python: captured per run in `artifacts/manifests/run_manifest_<run_id>.json`
- Dependencies: `requirements.txt` (include `mlflow`) and runtime environment snapshot in run manifest
- Config: `configs/default.yaml` with `config_sha256` stored in run manifest
- Dataset mode: strict by default (`use_synthetic_if_missing: false`)
- Dataset integrity: CSV checksums recorded per run in run manifest

## Commands

- Full benchmark: `python -m src.main --config configs/default.yaml`
- Report regeneration: `python -m src.main --config configs/default.yaml --report-only`
- Dev fallback run: `python -m src.main --config configs/default.yaml --allow-synthetic-for-dev`
- MLflow tracking UI: `mlflow ui --backend-store-uri artifacts/mlruns`

## Required Artifact Traceability

- Raw metrics: `artifacts/results/results_raw.csv`
- Summary metrics: `artifacts/results/results_summary.csv`
- Main table export: `artifacts/tables/table_main_results.csv`
- Fairness table export: `artifacts/tables/table_fairness_results.csv`
- Ablation table export: `artifacts/tables/table_ablation.csv`
- Significance table export: `artifacts/tables/table_significance.csv`
- Figures: `artifacts/figures/` (`.png` and `.pdf`)
- Diagrams: `artifacts/diagrams/` (`.png` and `.svg`)
- Text narrative output: `artifacts/text/`
- Figure manifest: `artifacts/manifests/figure_manifest.csv`
- Table manifest: `artifacts/manifests/table_manifest.csv`
- Text manifest: `artifacts/manifests/text_manifest.csv`
- Run manifest: `artifacts/manifests/run_manifest_<run_id>.json`
- Repro bundle index: `artifacts/manifests/repro_bundle_<run_id>.md`
- Run log: `artifacts/logs/run_<run_id>.log`
- Paper sync target: `paper/paper_final.md`

## Quality Gate Checklist

- Semua tabel dan gambar yang dipakai di paper terdaftar di manifest.
- Semua angka naratif berasal dari `results_summary.csv` atau `results_raw.csv`.
- `run_manifest_<run_id>.json` memuat commit, config hash, environment, dan checksum data.
- `repro_bundle_<run_id>.md` memuat daftar akhir artefak untuk penulisan tanpa rerun.

