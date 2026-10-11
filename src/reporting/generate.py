"""Generate figures and synchronize markdown results section."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


RESULTS_START = "<!-- AUTO_RESULTS_START -->"
RESULTS_END = "<!-- AUTO_RESULTS_END -->"


def _save_plot_multiple_formats(base_path: Path) -> list[str]:
    outputs = []
    for ext in (".png", ".pdf"):
        out = base_path.with_suffix(ext)
        plt.savefig(out, dpi=150 if ext == ".png" else None, bbox_inches="tight")
        outputs.append(str(out.as_posix()))
    return outputs


def _generate_diagram(path: Path, title: str, nodes: list[str]) -> list[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3))
    ax = plt.gca()
    ax.axis("off")
    x_positions = list(range(len(nodes)))
    for i, label in enumerate(nodes):
        box = FancyBboxPatch((i * 2, 0.8), 1.6, 0.7, boxstyle="round,pad=0.2", edgecolor="black")
        ax.add_patch(box)
        ax.text(i * 2 + 0.8, 1.15, label, ha="center", va="center", fontsize=9)
        if i < len(nodes) - 1:
            ax.annotate("", xy=((i + 1) * 2, 1.15), xytext=(i * 2 + 1.6, 1.15), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.4, len(nodes) * 2)
    ax.set_ylim(0.3, 2.0)
    ax.set_title(title)
    out_paths = []
    for ext in (".png", ".svg"):
        out_file = path.with_suffix(ext)
        plt.savefig(out_file, bbox_inches="tight")
        out_paths.append(str(out_file.as_posix()))
    plt.close()
    return out_paths


def _write_csv_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_text_artifacts(summary: pd.DataFrame, text_dir: Path, run_id: str) -> list[str]:
    text_dir.mkdir(parents=True, exist_ok=True)
    narrative_path = text_dir / f"result_narrative_{run_id}.md"
    notes_path = text_dir / f"method_notes_{run_id}.md"
    lines = ["# Auto Result Narrative", ""]
    for dataset in sorted(summary["dataset"].unique()):
        sub = summary[summary["dataset"] == dataset].sort_values("f1_macro_mean", ascending=False)
        best = sub.iloc[0]
        lines.append(f"## {dataset}")
        lines.append(
            f"Best method: `{best['method']}` with mean macro-F1 "
            f"{best['f1_macro_mean']:.4f}, MCC {best['mcc_mean']:.4f}, "
            f"DP diff {best['dp_diff_mean']:.4f}."
        )
        lines.append("")
    narrative_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    notes_path.write_text(
        "\n".join(
            [
                "# Method Notes",
                "",
                "- Generated automatically from the active run configuration.",
                "- Re-verify fairness-constraint assumptions before final submission.",
                "- Use the run manifest to track command, environment, and data checksums.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [str(narrative_path.as_posix()), str(notes_path.as_posix())]


def generate_figures(
    summary: pd.DataFrame,
    figures_dir: str | Path,
    out_root: str | Path | None = None,
    run_id: str | None = None,
) -> None:
    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = "latest"
    root = Path(out_root) if out_root is not None else out.parent
    diagrams_dir = root / "diagrams"
    manifests_dir = root / "manifests"
    text_dir = root / "text"
    figure_manifest_rows: list[dict] = []
    for dataset in summary["dataset"].unique():
        sub = summary[summary["dataset"] == dataset].copy()
        plt.figure(figsize=(10, 5))
        plt.bar(sub["method"], sub["f1_macro_mean"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("F1-Macro Mean")
        plt.title(f"{dataset} - Main Results")
        plt.tight_layout()
        main_outputs = _save_plot_multiple_formats(out / f"fig_main_performance_{dataset}")
        plt.close()
        figure_manifest_rows.append(
            {
                "id_figure": f"fig_main_performance_{dataset}",
                "caption": f"{dataset} main performance comparison",
                "source_table": "artifacts/results/results_summary.csv",
                "source_script": "src/reporting/generate.py",
                "output_paths": ";".join(main_outputs),
            }
        )

        plt.figure(figsize=(10, 5))
        plt.scatter(sub["dp_diff_mean"], sub["f1_macro_mean"])
        for _, row in sub.iterrows():
            plt.annotate(row["method"], (row["dp_diff_mean"], row["f1_macro_mean"]), fontsize=7)
        plt.xlabel("Demographic Parity Difference (lower is fairer)")
        plt.ylabel("F1-Macro Mean")
        plt.title(f"{dataset} - Fairness/Performance Tradeoff")
        plt.tight_layout()
        fairness_outputs = _save_plot_multiple_formats(out / f"fig_fairness_tradeoff_{dataset}")
        plt.close()
        figure_manifest_rows.append(
            {
                "id_figure": f"fig_fairness_tradeoff_{dataset}",
                "caption": f"{dataset} fairness-performance tradeoff",
                "source_table": "artifacts/results/results_summary.csv",
                "source_script": "src/reporting/generate.py",
                "output_paths": ";".join(fairness_outputs),
            }
        )

        plt.figure(figsize=(10, 5))
        plt.bar(sub["method"], sub["dp_diff_mean"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Demographic Parity Difference")
        plt.title(f"{dataset} - Data Profile (Fairness Dispersion)")
        plt.tight_layout()
        profile_outputs = _save_plot_multiple_formats(out / f"fig_data_profile_{dataset}")
        plt.close()
        figure_manifest_rows.append(
            {
                "id_figure": f"fig_data_profile_{dataset}",
                "caption": f"{dataset} profile of fairness disparity by method",
                "source_table": "artifacts/results/results_summary.csv",
                "source_script": "src/reporting/generate.py",
                "output_paths": ";".join(profile_outputs),
            }
        )

    pipeline_paths = _generate_diagram(
        diagrams_dir / "pipeline_overview",
        "Pipeline Overview",
        ["LoadConfig", "RunBenchmark", "SaveResults", "GenerateFigures", "SyncPaper"],
    )
    flow_paths = _generate_diagram(
        diagrams_dir / "experiment_flow",
        "Experiment Flow",
        ["Dataset", "SeedFoldLoop", "MethodTraining", "Metrics", "Artifacts"],
    )
    figure_manifest_rows.extend(
        [
            {
                "id_figure": "pipeline_overview",
                "caption": "End-to-end experiment pipeline",
                "source_table": "artifacts/results/results_summary.csv",
                "source_script": "src/reporting/generate.py",
                "output_paths": ";".join(pipeline_paths),
            },
            {
                "id_figure": "experiment_flow",
                "caption": "Cross-validation experiment flow",
                "source_table": "artifacts/results/results_raw.csv",
                "source_script": "src/reporting/generate.py",
                "output_paths": ";".join(flow_paths),
            },
        ]
    )
    _write_csv_manifest(manifests_dir / "figure_manifest.csv", figure_manifest_rows)
    text_files = _write_text_artifacts(summary, text_dir=text_dir, run_id=run_id)
    _write_csv_manifest(
        manifests_dir / "text_manifest.csv",
        [
            {
                "id_teks": f"text_{i+1}",
                "section_paper_target": "Results",
                "source_metrics": "artifacts/results/results_summary.csv",
                "output_path": text_file,
            }
            for i, text_file in enumerate(text_files)
        ],
    )


def _build_results_markdown(summary: pd.DataFrame) -> str:
    from src.reporting.best_per_dataset import select_best_per_dataset

    blocks = ["## Results (Auto-Generated)\n"]
    best_ds = select_best_per_dataset(summary)
    if not best_ds.empty:
        blocks.append("### Best method per dataset (macro-F1; ties broken by joint_score, then MCC, then PR-AUC; dummy excluded)\n")
        show_best = best_ds[
            [c for c in ["dataset", "method", "f1_macro_mean", "mcc_mean", "joint_score_mean", "dp_diff_mean"] if c in best_ds.columns]
        ].round(4)
        blocks.append(show_best.to_string(index=False))
        blocks.append("")
    for dataset in sorted(summary["dataset"].unique()):
        sub = summary[summary["dataset"] == dataset].copy()
        blocks.append(f"### {dataset}\n")
        show = sub[
            [
                "method",
                "f1_macro_mean",
                "f1_macro_std",
                "f1_macro_ci_low",
                "f1_macro_ci_high",
                "mcc_mean",
                "dp_diff_mean",
                "eod_diff_mean",
            ]
        ].round(4)
        blocks.append(show.to_string(index=False))
        blocks.append("")
    return "\n".join(blocks).strip()


def sync_paper_results(summary: pd.DataFrame, paper_file: str | Path) -> None:
    path = Path(paper_file)
    if path.exists():
        content = path.read_text(encoding="utf-8")
    else:
        content = (
            "# Paper Draft\n\n"
            f"{RESULTS_START}\n"
            "Results placeholder.\n"
            f"{RESULTS_END}\n"
        )
    if RESULTS_START not in content or RESULTS_END not in content:
        content += f"\n\n{RESULTS_START}\nResults placeholder.\n{RESULTS_END}\n"
    # Last marker pair wins if the paper mentions the same substring earlier.
    start_idx = content.rindex(RESULTS_START) + len(RESULTS_START)
    end_idx = content.index(RESULTS_END, start_idx)
    replacement = "\n\n" + _build_results_markdown(summary) + "\n\n"
    new_content = content[:start_idx] + replacement + content[end_idx:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_content, encoding="utf-8")

