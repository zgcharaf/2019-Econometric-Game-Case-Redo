import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_dyn_ar1(endog: pd.Series, exog: pd.DataFrame, intercept: bool = True):
    model = SARIMAX(
        endog=endog,
        exog=exog,
        order=(1, 0, 0),
        trend="c" if intercept else "n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return model.fit(disp=False)


def fit_ar1(endog: pd.Series):
    model = SARIMAX(
        endog=endog,
        order=(1, 0, 0),
        trend="n",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    return model.fit(disp=False)
