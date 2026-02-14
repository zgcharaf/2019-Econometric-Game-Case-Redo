# src/pipeline.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.utils.tracking import new_run, save_table, save_artifact_df, update_metrics
from src.io.load_budget import load_budget_excel
from src.io.load_drivers import load_driver_csvs
from src.io.load_rcp import load_rcp_dat
from src.transform.merge import merge_drivers
from src.transform.features import add_emissions_features, add_driver_features
from src.models.implied_gatm import implied_gatm
from src.viz.plots_budget import plot_budget_components, plot_imbalance
from src.diagnostics.stationarity import adf_kpss
from src.utils.constants import YEAR, EFF, ELUC, BIM


def cmd_run(args) -> None:
    ctx = new_run(Path(args.outputs))

    budget = load_budget_excel(Path(args.budget_xlsx))
    drivers = load_driver_csvs(Path(args.processed_dir))

    df_all = merge_drivers(budget, drivers)
    df_all = add_emissions_features(df_all)
    df_all = add_driver_features(df_all)

    save_artifact_df(df_all, ctx.artifacts_dir / "df_all.parquet")

    # ---- paper tables: missingness + coverage
    core_cols = [
        "Year",
        "E_total", "dE_total",
        "fossil fuel and industry", "land-use change emissions",
        "ocean sink", "land sink", "atmospheric growth", "budget imbalance",
        "nino34_mean_z", "nino34_max_z", "nino34_var_logz",
        "scpdsi_global_mean_z", "scpdsi_global_max_z", "scpdsi_global_var_logz",
        "tau_global_mean_z", "tau_global_max_z", "tau_global_var_logz",
        "DM_g_year_z",
    ]
    present = [c for c in core_cols if c in df_all.columns]

    miss = df_all[present].isna().sum().reset_index()
    miss.columns = ["variable", "missing_n"]
    miss["missing_share"] = miss["missing_n"] / len(df_all)
    save_table(miss, ctx.tables_dir / "table_missingness.csv")

    summ = df_all[present].describe().T.reset_index().rename(columns={"index": "variable"})
    save_table(summ, ctx.tables_dir / "table_summary_stats.csv")

    corr_vars = [c for c in present if c != "Year"]
    df_all[corr_vars].corr().to_csv(ctx.tables_dir / "table_correlations.csv")

    # ---- plots
    plot_budget_components(budget, ctx.figures_dir / "budget_components.png")
    plot_imbalance(budget, ctx.figures_dir / "budget_imbalance.png")

    # ---- stationarity on imbalance
    stat = adf_kpss(budget[BIM])
    save_table(pd.DataFrame([stat]), ctx.tables_dir / "stationarity_imbalance.csv")
    update_metrics(ctx, **{f"imb_{k}": v for k, v in stat.items()})

    # =========================
    # MAIN MODEL
    # =========================
    from src.diagnostics.regression_diagnostics import coef_table, residual_diagnostics
    from src.utils.export_tables import save_csv_and_latex

    drivers_list = [s.strip() for s in args.drivers.split(",")] if args.drivers else []
    df_win = df_all[(df_all[YEAR] >= args.start) & (df_all[YEAR] <= args.end)].copy()

    series, metrics, (res_ocn, res_lnd, res_bim) = implied_gatm(df_win, drivers_list, label=args.label)

    save_artifact_df(series, ctx.artifacts_dir / "implied_gatm.parquet")
    save_csv_and_latex(
        pd.DataFrame([metrics]),
        ctx.tables_dir / "implied_gatm_metrics.csv",
        ctx.tables_dir / "implied_gatm_metrics.tex",
    )

    coef_all = pd.concat(
        [
            coef_table(res_ocn, "Ocean sink"),
            coef_table(res_lnd, "Land sink"),
            coef_table(res_bim, "Imbalance AR(1)"),
        ],
        ignore_index=True,
    )
    save_csv_and_latex(
        coef_all,
        ctx.tables_dir / "table_coefficients.csv",
        ctx.tables_dir / "table_coefficients.tex",
    )

    rd = pd.concat(
        [
            residual_diagnostics(res_ocn, "Ocean sink"),
            residual_diagnostics(res_lnd, "Land sink"),
            residual_diagnostics(res_bim, "Imbalance"),
        ],
        ignore_index=True,
    )
    save_csv_and_latex(
        rd,
        ctx.tables_dir / "table_residual_diagnostics.csv",
        ctx.tables_dir / "table_residual_diagnostics.tex",
    )

    update_metrics(ctx, **{f"implied_{k}": v for k, v in metrics.items() if k != "label"})
    update_metrics(ctx, implied_label=metrics["label"])

    # ====================================
    # OPTIONAL: RCP inversion (MUST be INSIDE cmd_run)
    # ====================================
    if args.rcp_dat and Path(args.rcp_dat).exists():
        from src.models.rcp_inversion import rcp_from_concentration, run_rcp_suite
        from src.viz.rcp_plots import (
            plot_rcp_atm_growth,
            plot_rcp_emissions_paths,
            plot_linear_vs_saturation_for_scenario,
        )

        # 1) Load RCP and convert concentration -> G_ATM
        rcp_raw = load_rcp_dat(Path(args.rcp_dat))  # must return Year + CO2_ppm
        rcp = rcp_from_concentration(rcp_raw, year_col="Year", ppm_col="CO2_ppm")
        rcp = rcp[(rcp["Year"] >= args.rcp_start) & (rcp["Year"] <= args.rcp_end)].copy().reset_index(drop=True)

        plot_rcp_atm_growth(
            rcp,
            ctx.figures_dir / "rcp_implied_gatm.png",
            title="Atmospheric growth implied by RCP CO₂ concentration path",
        )

        # 2) E0 from last observed budget year (or user arg)
        E0_year = int(budget[YEAR].max()) if args.E0_year is None else int(args.E0_year)
        e0_row = budget.loc[budget[YEAR] == E0_year, [EFF, ELUC]]
        if e0_row.empty:
            raise ValueError(f"E0_year={E0_year} not found in budget.")
        E0 = float(e0_row[EFF].iloc[0] + e0_row[ELUC].iloc[0])

        scenarios = {
            "baseline": {},
            "bad_climate": {
                "nino34_max_z": +1.0,
                "scpdsi_global_mean_z": -1.0,
                "scpdsi_global_max_z": -1.0,
                "DM_g_year_z": +1.0,
                "tau_global_mean_z": +1.0,
            },
            "good_climate": {
                "nino34_max_z": -1.0,
                "scpdsi_global_mean_z": +1.0,
                "scpdsi_global_max_z": +1.0,
                "DM_g_year_z": -1.0,
                "tau_global_mean_z": -1.0,
            },
        }

        kappas = [0.005, 0.01]

        tab_rcp, paths_rcp, info_rcp = run_rcp_suite(
            rcp=rcp,
            res_ocn=res_ocn,
            res_lnd=res_lnd,
            E0=E0,
            kappas=kappas,
            scenarios=scenarios,
            BIM=0.0,
        )

        save_csv_and_latex(
            tab_rcp,
            ctx.tables_dir / "table_rcp_emissions_paths_summary.csv",
            ctx.tables_dir / "table_rcp_emissions_paths_summary.tex",
            float_fmt="%.3f",
        )
        save_csv_and_latex(
            info_rcp,
            ctx.tables_dir / "table_rcp_inversion_params.csv",
            ctx.tables_dir / "table_rcp_inversion_params.tex",
            float_fmt="%.4f",
        )

        df_long = pd.concat(
            [dfp[["Year", "G_ATM_rcp_GtC", "E_total", "dE_total", "spec"]] for dfp in paths_rcp.values()],
            ignore_index=True,
        )
        save_artifact_df(df_long, ctx.artifacts_dir / "rcp_emissions_paths_long.parquet")

        plot_rcp_emissions_paths(
            paths_rcp,
            ctx.figures_dir / "rcp_emissions_all_specs.png",
            title="RCP-implied admissible emissions: linear vs saturation (all scenarios)",
        )
        plot_linear_vs_saturation_for_scenario(
            paths_rcp, "baseline", kappas, ctx.figures_dir / "rcp_emissions_baseline_linear_vs_sat.png"
        )
        plot_linear_vs_saturation_for_scenario(
            paths_rcp, "bad_climate", kappas, ctx.figures_dir / "rcp_emissions_bad_linear_vs_sat.png"
        )
        plot_linear_vs_saturation_for_scenario(
            paths_rcp, "good_climate", kappas, ctx.figures_dir / "rcp_emissions_good_linear_vs_sat.png"
        )


def main():
    ap = argparse.ArgumentParser(prog="carbon-budget-pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run end-to-end pipeline")
    run.add_argument("--budget_xlsx", default="data/raw/Global_Carbon_Budget_2018v1.0.xlsx")
    run.add_argument("--processed_dir", default="data/processed")
    run.add_argument("--outputs", default="outputs")
    run.add_argument("--start", type=int, default=1989)
    run.add_argument("--end", type=int, default=2012)
    run.add_argument("--drivers", default="nino34_max_z,scpdsi_global_max_z,tau_global_mean_z")
    run.add_argument("--label", default="default window model")
    run.add_argument("--rcp_dat", default="data/raw/RCP3PD_MIDYR_CONC.DAT")
    run.add_argument("--rcp_start", type=int, default=2020)
    run.add_argument("--rcp_end", type=int, default=2100)
    run.add_argument("--E0_year", type=int, default=None)
    run.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
