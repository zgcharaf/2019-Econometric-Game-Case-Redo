# src/utils/export_tables.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_csv_and_latex(df: pd.DataFrame, csv_path: Path, tex_path: Path, float_fmt: str = "%.3f") -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    tex = df.to_latex(index=False, float_format=lambda x: float_fmt % x)
    tex_path.write_text(tex, encoding="utf-8")
