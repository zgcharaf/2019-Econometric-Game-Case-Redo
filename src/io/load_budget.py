from pathlib import Path
import pandas as pd

from src.utils.constants import YEAR, EFF, ELUC, SOCN, SLND, GATM, BIM


def load_budget_excel(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Global Carbon Budget", skiprows=19)
    df = df.copy()
    df[YEAR] = df[YEAR].astype(int)

    if BIM not in df.columns:
        df[BIM] = df[EFF] + df[ELUC] - df[SOCN] - df[SLND] - df[GATM]

    return df.sort_values(YEAR).reset_index(drop=True)
