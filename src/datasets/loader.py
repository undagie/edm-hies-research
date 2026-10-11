"""Dataset adapters: load CSVs, binarise targets, drop leakage columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    target_col: str
    sensitive_col: str
    secondary_sensitive_col: str | None
    leakage_columns: list[str]


def _ensure_sensitive(df: pd.DataFrame, preferred: list[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for col in preferred:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    # Always provide a fallback column to keep experiments runnable.
    df["gender_fallback"] = "unknown"
    return "gender_fallback"


def _binaryize_from_threshold(series: pd.Series, threshold: float) -> pd.Series:
    return (series < threshold).astype(int)


def _load_or_synthetic(path: Path, name: str, seed: int, allow_synthetic: bool) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    if not allow_synthetic:
        raise FileNotFoundError(
            f"Missing dataset file: {path}. Provide the raw CSV or enable synthetic fallback for development."
        )
    rng = np.random.default_rng(seed)
    n = 600
    synthetic = pd.DataFrame(
        {
            "gender": rng.choice(["F", "M"], n, p=[0.52, 0.48]),
            "ses_proxy": rng.choice(["low", "mid", "high"], n, p=[0.3, 0.5, 0.2]),
            "feat1": rng.normal(0, 1, n),
            "feat2": rng.normal(0, 1, n),
            "feat3": rng.normal(0, 1, n),
        }
    )
    score = 10 + 1.5 * synthetic["feat1"] + 0.8 * synthetic["feat2"] - 1.2 * synthetic["feat3"]
    if name == "uci_dropout_success":
        score -= 0.4
    if name == "xapi":
        score += 0.4
    synthetic["final_score"] = np.clip(score + rng.normal(0, 2, n), 0, 20)
    return synthetic


def load_dataset_bundle(
    name: str,
    raw_dir: str | Path,
    seed: int = 42,
    allow_synthetic: bool = True,
) -> DatasetBundle:
    raw_path = Path(raw_dir)
    if name == "uci_student":
        df = _load_or_synthetic(raw_path / "uci_student.csv", name, seed, allow_synthetic=allow_synthetic)
        if "G3" in df.columns:
            df["target_at_risk"] = _binaryize_from_threshold(df["G3"], threshold=10)
        else:
            df["target_at_risk"] = _binaryize_from_threshold(df["final_score"], threshold=10)
        sensitive_col = _ensure_sensitive(df, ["sex", "gender"])
        secondary = "Medu" if "Medu" in df.columns else ("ses_proxy" if "ses_proxy" in df.columns else None)
        leakage_columns = ["G3", "final_score"]
    elif name == "uci_dropout_success":
        df = _load_or_synthetic(
            raw_path / "uci_dropout_success.csv",
            name,
            seed,
            allow_synthetic=allow_synthetic,
        )
        if "Target" in df.columns:
            df["target_at_risk"] = df["Target"].astype(str).str.lower().eq("dropout").astype(int)
        elif "target" in df.columns:
            df["target_at_risk"] = df["target"].astype(str).str.lower().eq("dropout").astype(int)
        else:
            df["target_at_risk"] = _binaryize_from_threshold(df["final_score"], threshold=10)
        sensitive_col = _ensure_sensitive(df, ["Gender", "gender", "sex"])
        secondary = (
            "Scholarship holder"
            if "Scholarship holder" in df.columns
            else ("Nacionality" if "Nacionality" in df.columns else ("ses_proxy" if "ses_proxy" in df.columns else None))
        )
        leakage_columns = ["Target", "target", "final_score"]
    elif name == "xapi":
        df = _load_or_synthetic(raw_path / "xapi.csv", name, seed, allow_synthetic=allow_synthetic)
        if "Class" in df.columns:
            df["target_at_risk"] = df["Class"].astype(str).str.upper().eq("L").astype(int)
        elif "class" in df.columns:
            df["target_at_risk"] = df["class"].astype(str).str.upper().eq("L").astype(int)
        else:
            df["target_at_risk"] = _binaryize_from_threshold(df["final_score"], threshold=10)
        sensitive_col = _ensure_sensitive(df, ["gender", "sex"])
        secondary = "NationalITy" if "NationalITy" in df.columns else ("ses_proxy" if "ses_proxy" in df.columns else None)
        leakage_columns = ["Class", "class", "final_score"]
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return DatasetBundle(
        name=name,
        frame=df,
        target_col="target_at_risk",
        sensitive_col=sensitive_col,
        secondary_sensitive_col=secondary,
        leakage_columns=[col for col in leakage_columns if col in df.columns],
    )

