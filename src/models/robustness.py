
# src/models/robustness.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.implied_gatm import implied_gatm


def run_spec_grid(
    df_window: pd.DataFrame,
    base_label: str,
    base_drivers: list[str],
    nino_choices: dict[str, str],
    pdsi_choices: dict[str, str],
    fixed_drivers: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Runs a grid of model specs:
      land drivers = [E_total, dE_total] + (one nino stat) + (one pdsi stat) + fixed_drivers

    Returns:
      - results table (metrics + driver coefs)
      - series dictionary {spec_label: df_series}
    """
    if fixed_drivers is None:
        fixed_drivers = []

    results = []
    series_map: dict[str, pd.DataFrame] = {}

    for nino_name, nino_col in nino_choices.items():
        for pdsi_name, pdsi_col in pdsi_choices.items():
            drivers = [nino_col, pdsi_col] + fixed_drivers
            label = f"{base_label}: {nino_name}+{pdsi_name} + {'+'.join(fixed_drivers) if fixed_drivers else 'no-fixed'}"

            series, metrics, (res_ocn, res_lnd, res_bim) = implied_gatm(df_window, drivers, label=label)

            # Pull land coefficients for interpretability
            land_params = res_lnd.params
            for d in drivers:
                metrics[f"coef_{d}"] = float(land_params.get(d, np.nan))

            results.append(metrics)
            series_map[label] = series

    tab = pd.DataFrame(results).sort_values(["corr", "rmse"], ascending=[False, True]).reset_index(drop=True)
    return tab, series_map


def nested_specs(
    df_window: pd.DataFrame,
    specs: list[tuple[str, list[str]]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Runs a nested set of specs, useful for incremental (baseline -> +controls -> +extra).
    """
    results = []
    series_map: dict[str, pd.DataFrame] = {}
    for label, drivers in specs:
        series, metrics, (res_ocn, res_lnd, _) = implied_gatm(df_window, drivers, label=label)
        land_params = res_lnd.params
        for d in drivers:
            metrics[f"coef_{d}"] = float(land_params.get(d, np.nan))
        results.append(metrics)
        series_map[label] = series

    tab = pd.DataFrame(results).sort_values(["corr", "rmse"], ascending=[False, True]).reset_index(drop=True)
    return tab, series_map
