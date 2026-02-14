from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass
class RunContext:
    run_dir: Path
    tables_dir: Path
    figures_dir: Path
    artifacts_dir: Path
    log_path: Path
    metrics_path: Path

def new_run(outputs_root: Path, run_name: str | None = None) -> RunContext:
    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    run_dir = outputs_root / "runs" / run_name
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    artifacts_dir = run_dir / "artifacts"

    for d in (tables_dir, figures_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "logs.txt"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        metrics_path.write_text(json.dumps({}, indent=2))

    return RunContext(
        run_dir=run_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        artifacts_dir=artifacts_dir,
        log_path=log_path,
        metrics_path=metrics_path,
    )

def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_artifact_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def update_metrics(ctx: RunContext, **kwargs: Any) -> None:
    data: Dict[str, Any] = json.loads(ctx.metrics_path.read_text())
    data.update(kwargs)
    ctx.metrics_path.write_text(json.dumps(data, indent=2))
