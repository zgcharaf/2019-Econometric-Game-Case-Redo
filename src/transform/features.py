import numpy as np
import pandas as pd
from src.utils.constants import EFF, ELUC


def zscore(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return (x - x.mean()) / x.std(ddof=0)


def add_emissions_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["E_total"] = df[EFF] + df[ELUC]
    df["dE_total"] = df["E_total"].diff()
    return df


def add_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    direct = [
        "nino34_mean", "nino34_max",
        "scpdsi_global_mean", "scpdsi_global_max",
        "tau_global_mean", "tau_global_max",
        "DM_g_year",
    ]
    for c in direct:
        if c in df.columns:
            df[c + "_z"] = zscore(df[c])

    for c in ["nino34_var", "scpdsi_global_var", "tau_global_var"]:
        if c in df.columns:
            df[c + "_logz"] = zscore(np.log1p(df[c]))

    return df
