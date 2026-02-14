# src/diagnostics/regression_diagnostics.py
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


def coef_table(res, model_name: str) -> pd.DataFrame:
    """
    Clean coefficient table for SARIMAX results.
    """
    params = res.params
    bse = res.bse
    z = params / bse
    p = res.pvalues
    out = pd.DataFrame(
        {
            "model": model_name,
            "param": params.index,
            "coef": params.values,
            "std_err": bse.values,
            "z": z.values,
            "p_value": p.values,
        }
    )
    return out


def residual_diagnostics(res, name: str, lags: int = 10) -> pd.DataFrame:
    """
    Paper-ish residual diagnostics:
    - JB normality
    - Ljung-Box autocorrelation at multiple lags
    """
    resid = pd.Series(res.resid).dropna().astype(float)
    jb_stat, jb_p, skew, kurt = jarque_bera(resid)

    lb = acorr_ljungbox(resid, lags=list(range(1, lags + 1)), return_df=True)

    rows = []
    rows.append(
        {
            "series": name,
            "jb_stat": float(jb_stat),
            "jb_p": float(jb_p),
            "skew": float(skew),
            "kurtosis": float(kurt),
        }
    )
    out = pd.DataFrame(rows)

    # Add a few LB p-values as columns (easy to cite in text)
    for k in [1, 2, 5, 10]:
        if k <= lags:
            out[f"lb_p_{k}"] = float(lb.loc[k, "lb_pvalue"])
    return out


def fit_metrics(obs: pd.Series, pred: pd.Series, label: str) -> pd.DataFrame:
    obs = pd.to_numeric(obs, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    m = obs.notna() & pred.notna()
    err = obs[m] - pred[m]
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    corr = float(obs[m].corr(pred[m]))
    return pd.DataFrame([{"spec": label, "n_obs": int(m.sum()), "rmse": rmse, "mae": mae, "corr": corr}])
