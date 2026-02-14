import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def adf_kpss(series: pd.Series) -> dict:
    s = series.dropna().astype(float)
    adf = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression="c", nlags="auto")
    return {
        "adf_stat": float(adf[0]),
        "adf_p": float(adf[1]),
        "adf_lags": int(adf[2]),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "kpss_lags": int(kpss_lags),
    }
