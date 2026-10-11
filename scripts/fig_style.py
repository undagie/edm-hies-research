"""Shared IJIES print-ready matplotlib style (full/column width, >= 10 pt)."""

from __future__ import annotations

import matplotlib

# A4 two-column: full text ~17.5 cm, single column ~8.4 cm.
FULL_WIDTH_IN = 6.89
COL_WIDTH_IN = 3.30
PNG_DPI = 600

DATASET_LABELS = {
    "uci_student": "UCI Student",
    "uci_dropout_success": "UCI Dropout",
    "xapi": "xAPI-Edu",
}


def apply_ijies_style(base_size: float = 10.0) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": base_size,
            "axes.titlesize": base_size,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size,
            "ytick.labelsize": base_size,
            "legend.fontsize": base_size,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
