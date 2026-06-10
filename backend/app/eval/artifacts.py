"""Persistence of evaluation runs as on-disk artifacts.

Hard constraint: experiment results are persisted as CSV/JSON, never only shown in the UI.
Each run gets its own folder under ``results/runs/<run_id>/``:

    config.json    the exact PipelineConfig used
    results.csv    one row per question
    summary.json   aggregate metrics + run metadata
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.settings import BACKEND_ROOT

logger = logging.getLogger(__name__)

# results/ lives at the project root, one level above backend/.
RUNS_DIR: Path = BACKEND_ROOT.parent / "results" / "runs"


def new_run_id(config_hash: str) -> str:
    """Build a sortable, unique run id from the current time and a config hash.

    Args:
        config_hash: Short hash of the configuration used.

    Returns:
        str: e.g. ``20260610T141502Z_ab12cd34ef56``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{config_hash}"


def write_run(
    run_id: str,
    config: dict,
    rows: list[dict],
    summary: dict,
) -> Path:
    """Write a run's config, per-question rows, and summary to disk.

    Args:
        run_id: The run identifier (folder name).
        config: The PipelineConfig as a dict.
        rows: Per-question result rows (uniform keys).
        summary: Aggregate metrics and metadata.

    Returns:
        Path: The run directory.
    """
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Wrote run %s (%d rows) to %s", run_id, len(rows), run_dir)
    return run_dir


def list_runs() -> list[dict]:
    """List all persisted run summaries, newest first.

    Returns:
        list[dict]: Each run's ``summary.json`` content (skips unreadable runs).
    """
    if not RUNS_DIR.exists():
        return []
    summaries: list[dict] = []
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            try:
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable summary: %s", summary_path)
    return summaries


def load_run(run_id: str) -> dict | None:
    """Load a single run's summary and per-question rows.

    Args:
        run_id: The run identifier.

    Returns:
        dict | None: ``{"summary": ..., "config": ..., "rows": [...]}`` or ``None`` if absent.
    """
    run_dir = RUNS_DIR / run_id
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    rows: list[dict] = []
    results_path = run_dir / "results.csv"
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    return {"summary": summary, "config": config, "rows": rows}
