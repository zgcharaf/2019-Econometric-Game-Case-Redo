from __future__ import annotations

import numpy as np
import pandas as pd


def get_param(res, name: str, default: float = 0.0) -> float:
    return float(res.params[name]) if name in res.params.index else default


def compute_driver_term(res, driver_values_dict: dict[str, float]) -> float:
    total = 0.0
    for k, v in driver_values_dict.items():
        if k in res.params.index:
            total += float(res.params[k]) * float(v)
    return total


def solve_emissions_closed_form(rcp: pd.DataFrame, res_ocn, res_lnd, E0: float, driver_dict=None) -> pd.DataFrame:
    sim = rcp[["Year", "G_ATM_rcp_GtC"]].dropna().copy().reset_index(drop=True)
    if driver_dict is None:
        driver_dict = {}

    ao = get_param(res_ocn, "intercept", 0.0) + get_param(res_ocn, "const", 0.0)
    bo = get_param(res_ocn, "E_total", 0.0)
    co = get_param(res_ocn, "dE_total", 0.0)

    al = get_param(res_lnd, "intercept", 0.0) + get_param(res_lnd, "const", 0.0)
    bl = get_param(res_lnd, "E_total", 0.0)
    cl = get_param(res_lnd, "dE_total", 0.0)

    Dl = compute_driver_term(res_lnd, driver_dict)

    A = ao + al
    B = bo + bl
    C = co + cl
    denom = 1.0 - B - C

    sim["E_total"] = np.nan
    sim["dE_total"] = np.nan
    sim.loc[0, "E_total"] = E0
    sim.loc[0, "dE_total"] = 0.0

    for t in range(1, len(sim)):
        Gt = float(sim.loc[t, "G_ATM_rcp_GtC"])
        E_prev = float(sim.loc[t - 1, "E_total"])
        E_t = (Gt + A + Dl + C * E_prev) / denom
        sim.loc[t, "E_total"] = E_t
        sim.loc[t, "dE_total"] = E_t - E_prev

    return sim


def solve_emissions_levels_only(rcp: pd.DataFrame, res_ocn, res_lnd, E0: float, driver_dict=None):
    sim = rcp[["Year", "G_ATM_rcp_GtC"]].dropna().copy().reset_index(drop=True)
    if driver_dict is None:
        driver_dict = {}

    ao = get_param(res_ocn, "intercept", 0.0) + get_param(res_ocn, "const", 0.0)
    bo = get_param(res_ocn, "E_total", 0.0)

    al = get_param(res_lnd, "intercept", 0.0) + get_param(res_lnd, "const", 0.0)
    bl = get_param(res_lnd, "E_total", 0.0)

    A = ao + al
    B = bo + bl
    denom = 1.0 - B

    D = compute_driver_term(res_lnd, driver_dict)

    sim["E_total"] = np.nan
    sim["dE_total"] = np.nan
    sim.loc[0, "E_total"] = E0
    sim.loc[0, "dE_total"] = 0.0

    for t in range(1, len(sim)):
        Gt = float(sim.loc[t, "G_ATM_rcp_GtC"])
        E_t = (Gt + A + D) / denom
        sim.loc[t, "E_total"] = E_t
        sim.loc[t, "dE_total"] = E_t - float(sim.loc[t - 1, "E_total"])

    info = {"A": A, "B": B, "D": D, "denom": denom}
    return sim, info
