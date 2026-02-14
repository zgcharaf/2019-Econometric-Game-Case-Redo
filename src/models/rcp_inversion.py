# src/models/rcp_inversion.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PPM_TO_GTC = 2.12  # GtC per ppm (approx)


def scenario_metrics(sim: pd.DataFrame, col: str = "E_total") -> Dict[str, object]:
    """
    Paper-friendly summary metrics for an implied emissions path.
    """
    s = sim.dropna(subset=[col]).copy()
    out: Dict[str, object] = {}

    if s.empty:
        out.update(
            peak_year=None,
            peak_E_total=None,
            net_zero_year=None,
            min_year=None,
            min_E_total=None,
            avg_slope_GtC_per_year=None,
        )
        return out

    peak_idx = s[col].idxmax()
    out["peak_year"] = int(s.loc[peak_idx, "Year"])
    out["peak_E_total"] = float(s.loc[peak_idx, col])

    nz = s[s[col] <= 0]
    out["net_zero_year"] = int(nz.iloc[0]["Year"]) if len(nz) else None

    min_idx = s[col].idxmin()
    out["min_year"] = int(s.loc[min_idx, "Year"])
    out["min_E_total"] = float(s.loc[min_idx, col])

    post = s[s["Year"] >= out["peak_year"]].copy()
    if len(post) >= 2:
        slope = np.polyfit(post["Year"].astype(float), post[col].astype(float), 1)[0]
        out["avg_slope_GtC_per_year"] = float(slope)
    else:
        out["avg_slope_GtC_per_year"] = None

    for y in [2030, 2050, 2100]:
        out[f"E_{y}"] = float(s.loc[s["Year"] == y, col].iloc[0]) if (s["Year"] == y).any() else np.nan

    return out


def _get_param(res, name: str) -> float:
    return float(res.params[name]) if name in res.params.index else 0.0


def _driver_term(res_lnd, driver_dict: Dict[str, float]) -> float:
    s = 0.0
    for k, v in driver_dict.items():
        if k in res_lnd.params.index:
            s += float(res_lnd.params[k]) * float(v)
    return float(s)


def rcp_from_concentration(df_rcp: pd.DataFrame, year_col="Year", ppm_col="CO2_ppm") -> pd.DataFrame:
    """
    Takes a CO2 concentration trajectory in ppm and returns yearly G_ATM in GtC/yr.
    """
    df = df_rcp[[year_col, ppm_col]].dropna().copy()
    df = df.sort_values(year_col).reset_index(drop=True)
    df["dCO2_ppm"] = df[ppm_col].diff()
    df["G_ATM_rcp_GtC"] = PPM_TO_GTC * df["dCO2_ppm"]
    return df.dropna(subset=["G_ATM_rcp_GtC"]).reset_index(drop=True)


def solve_emissions_levels_linear(
    rcp: pd.DataFrame,
    res_ocn,
    res_lnd,
    E0: float,
    driver_dict: Dict[str, float] | None = None,
    BIM: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Levels-only inversion (no saturation):
        E_t = (G_t + A + D + BIM) / (1 - B)
    where A = intercept_o + intercept_l, B = slope_o(E) + slope_l(E),
    D = driver term from land model.
    """
    if driver_dict is None:
        driver_dict = {}

    r = rcp[["Year", "G_ATM_rcp_GtC"]].dropna().copy().reset_index(drop=True)

    ao = _get_param(res_ocn, "intercept") + _get_param(res_ocn, "const")
    al = _get_param(res_lnd, "intercept") + _get_param(res_lnd, "const")
    bo = _get_param(res_ocn, "E_total")
    bl = _get_param(res_lnd, "E_total")

    A = ao + al
    B = bo + bl
    denom = 1.0 - B
    D = _driver_term(res_lnd, driver_dict)

    sim = r.copy()
    sim["E_total"] = np.nan
    sim["dE_total"] = np.nan
    sim.loc[0, "E_total"] = float(E0)
    sim.loc[0, "dE_total"] = 0.0

    for t in range(1, len(sim)):
        Gt = float(sim.loc[t, "G_ATM_rcp_GtC"])
        E_t = (Gt + A + D + float(BIM)) / denom
        sim.loc[t, "E_total"] = E_t
        sim.loc[t, "dE_total"] = E_t - float(sim.loc[t - 1, "E_total"])

    info = {"A": float(A), "B": float(B), "D": float(D), "denom": float(denom), "BIM": float(BIM)}
    return sim, info


def _invert_quadratic(G: float, A: float, B: float, D: float, BIM: float, kappa: float) -> float:
    """
    With saturation term: kappa * E^2 added to sinks (reduces admissible E).
    We solve:
        kappa E^2 + (1-B) E - (G + A + D + BIM) = 0
    Choose root closest to linear solution (kappa->0).
    """
    a = float(kappa)
    b = float(1.0 - B)
    c = -float(G + A + D + BIM)

    if abs(a) < 1e-12:
        return (G + A + D + BIM) / b

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return np.nan

    sqrt_disc = float(np.sqrt(disc))
    r1 = (-b + sqrt_disc) / (2.0 * a)
    r2 = (-b - sqrt_disc) / (2.0 * a)

    lin = (G + A + D + BIM) / b
    return r1 if abs(r1 - lin) <= abs(r2 - lin) else r2


def solve_emissions_levels_saturation(
    rcp: pd.DataFrame,
    res_ocn,
    res_lnd,
    E0: float,
    kappa: float,
    driver_dict: Dict[str, float] | None = None,
    BIM: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Levels-only inversion with saturation:
        kappa E^2 + (1-B)E - (G + A + D + BIM) = 0
    """
    if driver_dict is None:
        driver_dict = {}

    r = rcp[["Year", "G_ATM_rcp_GtC"]].dropna().copy().reset_index(drop=True)

    ao = _get_param(res_ocn, "intercept") + _get_param(res_ocn, "const")
    al = _get_param(res_lnd, "intercept") + _get_param(res_lnd, "const")
    bo = _get_param(res_ocn, "E_total")
    bl = _get_param(res_lnd, "E_total")

    A = ao + al
    B = bo + bl
    D = _driver_term(res_lnd, driver_dict)

    sim = r.copy()
    sim["E_total"] = np.nan
    sim["dE_total"] = np.nan
    sim.loc[0, "E_total"] = float(E0)
    sim.loc[0, "dE_total"] = 0.0

    for t in range(1, len(sim)):
        Gt = float(sim.loc[t, "G_ATM_rcp_GtC"])
        E_t = _invert_quadratic(Gt, A, B, D, float(BIM), float(kappa))
        sim.loc[t, "E_total"] = E_t
        sim.loc[t, "dE_total"] = E_t - float(sim.loc[t - 1, "E_total"])

    info = {
        "A": float(A),
        "B": float(B),
        "D": float(D),
        "kappa": float(kappa),
        "BIM": float(BIM),
        "denom": float(1.0 - B),
    }
    return sim, info

def ensure_rcp_gatm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have Year and G_ATM_rcp_GtC.
    If loader already computed it, just return those columns.
    If only CO2_ppm is present, compute via diff and PPM_TO_GTC.
    """
    if "Year" not in df.columns:
        raise ValueError("RCP df must include 'Year'.")

    if "G_ATM_rcp_GtC" in df.columns:
        out = df[["Year", "G_ATM_rcp_GtC"]].dropna().copy()
        return out.sort_values("Year").reset_index(drop=True)

    # fallback: compute from CO2_ppm
    if "CO2_ppm" not in df.columns:
        raise ValueError("RCP df must include either 'G_ATM_rcp_GtC' or 'CO2_ppm'.")

    tmp = df[["Year", "CO2_ppm"]].dropna().copy().sort_values("Year").reset_index(drop=True)
    tmp["dCO2_ppm"] = tmp["CO2_ppm"].diff()
    tmp["G_ATM_rcp_GtC"] = PPM_TO_GTC * tmp["dCO2_ppm"]
    return tmp.dropna(subset=["G_ATM_rcp_GtC"])[["Year", "G_ATM_rcp_GtC"]].reset_index(drop=True)


def run_rcp_suite(
    rcp: pd.DataFrame,
    res_ocn,
    res_lnd,
    E0: float,
    kappas: List[float],
    scenarios: Dict[str, Dict[str, float]],
    BIM: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Runs (linear + saturation) across scenarios and kappas.
    Returns:
      - long results table (paper Table)
      - paths dict: key->sim dataframe
      - info table (A,B,D,denom,kappa,...)
    """
    rows = []
    info_rows = []
    paths: Dict[str, pd.DataFrame] = {}

    for scen_name, driver_dict in scenarios.items():
        # Linear
        sim_lin, info_lin = solve_emissions_levels_linear(rcp, res_ocn, res_lnd, E0, driver_dict, BIM=BIM)
        key = f"{scen_name} | linear"
        sim_lin = sim_lin.copy()
        sim_lin["spec"] = key
        paths[key] = sim_lin
        m = scenario_metrics(sim_lin)
        m.update(spec=key, scenario=scen_name, model="linear", kappa=0.0)
        rows.append(m)
        info_rows.append({"spec": key, **info_lin})

        # Saturation
        for k in kappas:
            sim_sat, info_sat = solve_emissions_levels_saturation(rcp, res_ocn, res_lnd, E0, kappa=k, driver_dict=driver_dict, BIM=BIM)
            key = f"{scen_name} | sat(k={k})"
            sim_sat = sim_sat.copy()
            sim_sat["spec"] = key
            paths[key] = sim_sat
            m = scenario_metrics(sim_sat)
            m.update(spec=key, scenario=scen_name, model="saturation", kappa=float(k))
            rows.append(m)
            info_rows.append({"spec": key, **info_sat})

    tab = pd.DataFrame(rows).sort_values(["scenario", "model", "kappa"]).reset_index(drop=True)
    info = pd.DataFrame(info_rows).sort_values("spec").reset_index(drop=True)
    return tab, paths, info
