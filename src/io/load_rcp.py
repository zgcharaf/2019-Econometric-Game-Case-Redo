from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

from src.utils.constants import PPM_TO_GTC


def load_rcp_dat(path: Path) -> pd.DataFrame:
    """
    Parse RCP*_MIDYR_CONC.DAT-like file:
    - Finds first row starting with 4-digit year
    - Reads whitespace-separated table with unknown number of columns
    - Names Year and CO2, converts CO2 ppm to dCO2 and G_ATM_rcp_GtC (=2.12*dCO2_ppm)
    Returns columns: Year, CO2_ppm, dCO2_ppm, G_ATM_rcp_GtC
    """
    start_row = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if re.match(r"^\s*\d{4}\s+", line):
                start_row = i
                break
    if start_row is None:
        raise ValueError("Could not find a line starting with a 4-digit year.")

    df = pd.read_csv(path, sep=r"\s+", engine="python", skiprows=start_row, header=None)
    if df.shape[1] < 4:
        raise ValueError(f"Unexpected column count: {df.shape[1]}")

    df = df.copy()
    df.columns = ["Year", "CO2EQ", "KYOTO_CO2EQ", "CO2"] + [f"Gas_{i}" for i in range(4, df.shape[1])]
    df["Year"] = df["Year"].astype(int)
    df = df.rename(columns={"CO2": "CO2_ppm"}).sort_values("Year").reset_index(drop=True)
    df["dCO2_ppm"] = df["CO2_ppm"].diff()
    df["G_ATM_rcp_GtC"] = PPM_TO_GTC * df["dCO2_ppm"]
    return df[["Year", "CO2_ppm", "dCO2_ppm", "G_ATM_rcp_GtC"]]
