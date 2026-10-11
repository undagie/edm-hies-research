"""Run tracking and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return "unavailable"
    if completed.returncode != 0:
        return "unavailable"
    return completed.stdout.strip() or "unavailable"


def compute_file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return "missing"
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_config_hash(config_path: str | Path) -> str:
    return compute_file_sha256(config_path)


def collect_dataset_checksums(raw_dir: str | Path) -> list[dict[str, Any]]:
    folder = Path(raw_dir)
    if not folder.exists():
        return []
    records: list[dict[str, Any]] = []
    for csv_path in sorted(folder.glob("*.csv")):
        records.append(
            {
                "file": str(csv_path.as_posix()),
                "sha256": compute_file_sha256(csv_path),
                "size_bytes": csv_path.stat().st_size,
            }
        )
    return records


def make_run_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{now}_{uuid.uuid4().hex[:8]}"


def collect_environment_snapshot() -> dict[str, Any]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
    }


def collect_git_snapshot() -> dict[str, str]:
    return {
        "commit": _run_command(["git", "rev-parse", "HEAD"]),
        "branch": _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "is_dirty": "true"
        if _run_command(["git", "status", "--porcelain"]) not in ("", "unavailable")
        else "false",
    }


def build_run_manifest(
    run_id: str,
    args: dict[str, Any],
    config_path: str | Path,
    config_payload: dict[str, Any],
    raw_data_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": args,
        "output_root": str(Path(output_root).as_posix()),
        "config_path": str(Path(config_path).as_posix()),
        "config_sha256": compute_config_hash(config_path),
        "config": config_payload,
        "datasets": collect_dataset_checksums(raw_data_dir),
        "environment": collect_environment_snapshot(),
        "git": collect_git_snapshot(),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def setup_run_logger(run_id: str, logs_dir: str | Path) -> logging.Logger:
    logger = logging.getLogger("edm_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(Path(logs_dir) / f"run_{run_id}.log", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
