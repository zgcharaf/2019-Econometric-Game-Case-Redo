# src/utils/export_tables.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def _safe_latex_value(v, float_fmt: str) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        return float_fmt % v
    return str(v)


def _fallback_latex_table(df: pd.DataFrame, float_fmt: str) -> str:
    cols = [str(c) for c in df.columns]
    align = "l" * len(cols)
    header = " & ".join(cols) + r" \\"
    rows = []
    for _, row in df.iterrows():
        vals = [_safe_latex_value(v, float_fmt) for v in row.values]
        rows.append(" & ".join(vals) + r" \\")

    body = "\n".join(rows)
    return (
        "\\begin{tabular}{" + align + "}\n"
        "\\hline\n"
        + header
        + "\n\\hline\n"
        + body
        + "\n\\hline\n"
        + "\\end{tabular}\n"
    )


def save_csv_and_latex(df: pd.DataFrame, csv_path: Path, tex_path: Path, float_fmt: str = "%.3f") -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

    try:
        tex = df.to_latex(index=False, float_format=lambda x: float_fmt % x)
    except Exception:
        tex = _fallback_latex_table(df, float_fmt=float_fmt)

    tex_path.write_text(tex, encoding="utf-8")
