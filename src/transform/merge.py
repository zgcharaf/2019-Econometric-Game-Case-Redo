import pandas as pd
from src.utils.constants import YEAR


def merge_drivers(df_budget: pd.DataFrame, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    nino = dfs["nino"][["year", "nino34_mean", "nino34_max", "nino34_var"]]
    pdsi = dfs["pdsi"][["year", "scpdsi_global_mean", "scpdsi_global_max", "scpdsi_global_var"]]
    tau  = dfs["tau"][["year", "tau_global_mean", "tau_global_max", "tau_global_var"]]
    burn = dfs["burned"][["year", "DM_g_year"]]

    df = df_budget.copy()
    df[YEAR] = df[YEAR].astype(int)

    for d in [nino, pdsi, tau, burn]:
        df = df.merge(d, left_on=YEAR, right_on="year", how="left").drop(columns=["year"])

    return df
