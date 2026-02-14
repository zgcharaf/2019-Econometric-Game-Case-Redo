import numpy as np
import pandas as pd

from src.utils.constants import YEAR, SOCN, SLND, GATM, BIM
from src.models.sarimax_dynreg import fit_dyn_ar1, fit_ar1


def implied_gatm(df_window: pd.DataFrame, land_drivers: list[str], label: str):
    df = df_window.copy()
    need = ["E_total", "dE_total", SOCN, SLND, BIM, GATM] + land_drivers
    df = df.dropna(subset=need).copy()

    X_ocean = df[["E_total", "dE_total"]]
    X_land = df[["E_total", "dE_total"] + land_drivers]

    res_ocn = fit_dyn_ar1(df[SOCN], X_ocean)
    res_lnd = fit_dyn_ar1(df[SLND], X_land)
    res_bim = fit_ar1(df[BIM])

    df["S_OCN_hat"] = res_ocn.fittedvalues
    df["S_LND_hat"] = res_lnd.fittedvalues
    df["BIM_hat"] = res_bim.fittedvalues

    df["G_ATM_hat"] = df["E_total"] - df["S_OCN_hat"] - df["S_LND_hat"] - df["BIM_hat"]
    df["G_ATM_err"] = df[GATM] - df["G_ATM_hat"]

    rmse = float(np.sqrt(np.mean(df["G_ATM_err"] ** 2)))
    mae = float(np.mean(np.abs(df["G_ATM_err"])))
    corr = float(df[GATM].corr(df["G_ATM_hat"]))

    metrics = {"label": label, "n_obs": len(df), "rmse": rmse, "mae": mae, "corr": corr}
    return df[[YEAR, GATM, "G_ATM_hat"]].copy(), metrics, (res_ocn, res_lnd, res_bim)
