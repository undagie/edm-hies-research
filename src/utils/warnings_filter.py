"""Silence known noisy third-party warnings during benchmark runs."""

from __future__ import annotations

import warnings


def silence_known_warnings() -> None:
    """Silence noisy third-party warnings observed during full benchmark runs."""

    warnings.filterwarnings(
        "ignore",
        message=r"Setting an item of incompatible dtype is deprecated.*",
        category=FutureWarning,
        module=r"fairlearn(\.|$)",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*X has feature names.*",
        category=UserWarning,
        module=r"sklearn(\.|$)",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*does not have valid feature names.*",
        category=UserWarning,
        module=r"sklearn(\.|$)",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"lightgbm(\.|$)",
    )
