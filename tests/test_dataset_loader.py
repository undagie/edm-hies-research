from pathlib import Path

import pytest

from src.datasets.loader import load_dataset_bundle


def test_loader_raises_when_missing_and_synthetic_disabled():
    with pytest.raises(FileNotFoundError):
        load_dataset_bundle(
            "uci_student",
            Path("this/path/does/not/exist"),
            seed=42,
            allow_synthetic=False,
        )


def test_loader_exposes_leakage_columns_metadata():
    bundle = load_dataset_bundle("xapi", Path("this/path/does/not/exist"), seed=42, allow_synthetic=True)
    assert "final_score" in bundle.leakage_columns
