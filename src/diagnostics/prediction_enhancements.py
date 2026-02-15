from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.regressionplots import plot_partregress_grid, influence_plot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import acf, pacf

from src.utils.constants import YEAR, SLND, SOCN, BIM, GATM
from src.models.sarimax_dynreg import fit_dyn_ar1, fit_ar1
from src.utils.export_tables import save_csv_and_latex


def _mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denom = y_true.abs().replace(0, np.nan)
    v = ((y_true - y_pred).abs() / denom).dropna()
    return float(v.mean() * 100) if len(v) else np.nan


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _fit_spec(df: pd.DataFrame, land_drivers: list[str], split_idx: int, label: str) -> dict:
    need = ["E_total", "dE_total", SOCN, SLND, BIM, GATM] + land_drivers
    d = df.dropna(subset=need).copy()

    train = d.iloc[:split_idx].copy()
    test = d.iloc[split_idx:].copy()

    Xo_tr = train[["E_total", "dE_total"]]
    Xl_tr = train[["E_total", "dE_total"] + land_drivers]

    ro = fit_dyn_ar1(train[SOCN], Xo_tr)
    rl = fit_dyn_ar1(train[SLND], Xl_tr)
    rb = fit_ar1(train[BIM])

    # In-sample fit on train
    g_hat_tr = train["E_total"] - ro.fittedvalues - rl.fittedvalues - rb.fittedvalues

    out = {
        "spec": label,
        "n": int(len(train)),
        "r2": float(np.corrcoef(train[GATM], g_hat_tr)[0, 1] ** 2),
        "adj_r2": float(1 - (1 - np.corrcoef(train[GATM], g_hat_tr)[0, 1] ** 2) * (len(train) - 1) / max(len(train) - len(land_drivers) - 3, 1)),
        "aic": float(ro.aic + rl.aic + rb.aic),
        "bic": float(ro.bic + rl.bic + rb.bic),
        "rmse": _rmse(train[GATM], g_hat_tr),
        "mae": float(np.mean((train[GATM] - g_hat_tr).abs())),
        "mape_pct": _mape(train[GATM], g_hat_tr),
    }

    if len(test) > 0:
        Xo_te = test[["E_total", "dE_total"]]
        Xl_te = test[["E_total", "dE_total"] + land_drivers]

        po = ro.get_forecast(steps=len(test), exog=Xo_te).predicted_mean
        pl = rl.get_forecast(steps=len(test), exog=Xl_te).predicted_mean
        pb = rb.get_forecast(steps=len(test)).predicted_mean
        g_hat_te = test["E_total"].reset_index(drop=True) - po.reset_index(drop=True) - pl.reset_index(drop=True) - pb.reset_index(drop=True)
        out["oos_rmse"] = _rmse(test[GATM].reset_index(drop=True), g_hat_te)
        out["oos_mae"] = float(np.mean((test[GATM].reset_index(drop=True) - g_hat_te).abs()))
    else:
        out["oos_rmse"] = np.nan
        out["oos_mae"] = np.nan

    return out


def _land_ols_base(df: pd.DataFrame, land_drivers: list[str]) -> tuple[pd.DataFrame, pd.Series, sm.regression.linear_model.RegressionResultsWrapper]:
    cols = ["E_total", "dE_total"] + land_drivers
    d = df[[YEAR, SLND] + cols].dropna().copy()
    X = sm.add_constant(d[cols], has_constant="add")
    y = d[SLND]
    res = sm.OLS(y, X).fit()
    return d, y, res


def generate_prediction_outputs(
    df_win: pd.DataFrame,
    series: pd.DataFrame,
    res_ocn,
    res_lnd,
    res_bim,
    land_drivers: list[str],
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    np.random.seed(42)

    split_idx = max(int(len(df_win.dropna(subset=[GATM])) * 0.8), 5)

    specs = [
        ("baseline_drivers", land_drivers),
        ("no_climate_drivers", []),
        ("enso_only", [d for d in land_drivers if "nino" in d][:1]),
        ("reduced_first2", land_drivers[:2]),
    ]
    model_cmp = pd.DataFrame([_fit_spec(df_win, drv, split_idx, lab) for lab, drv in specs])
    save_csv_and_latex(model_cmp, tables_dir / "table_pred_model_comparison.csv", tables_dir / "table_pred_model_comparison.tex")

    d, y_land, ols_base = _land_ols_base(df_win, land_drivers)

    # coefficient stability (rolling)
    w = max(10, min(15, len(d) - 1))
    rolling_rows = []
    for end in range(w, len(d) + 1):
        sub = d.iloc[end - w:end]
        Xs = sm.add_constant(sub[["E_total", "dE_total"] + land_drivers], has_constant="add")
        rs = sm.OLS(sub[SLND], Xs).fit()
        cis = rs.conf_int()
        for p in rs.params.index:
            rolling_rows.append(
                {
                    "window_start_year": int(sub[YEAR].min()),
                    "window_end_year": int(sub[YEAR].max()),
                    "param": p,
                    "coef": float(rs.params[p]),
                    "ci_width": float(cis.loc[p, 1] - cis.loc[p, 0]),
                }
            )
    coef_stab = pd.DataFrame(rolling_rows)
    save_csv_and_latex(coef_stab, tables_dir / "table_pred_coefficient_stability.csv", tables_dir / "table_pred_coefficient_stability.tex")

    # robustness table
    X = sm.add_constant(d[["E_total", "dE_total"] + land_drivers], has_constant="add")
    y = d[SLND]
    r_hc3 = sm.OLS(y, X).fit(cov_type="HC3")
    r_hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})

    d_alt = d.copy()
    d_alt["SLND_lag1"] = d_alt[SLND].shift(1)
    d_alt = d_alt.dropna().copy()
    X_alt = sm.add_constant(d_alt[["E_total", "dE_total", "SLND_lag1"] + land_drivers], has_constant="add")
    r_alt = sm.OLS(d_alt[SLND], X_alt).fit(cov_type="HC3")

    rob_rows = []
    for name, rr in [("HC3", r_hc3), ("HAC(2)", r_hac), ("Alt lag + HC3", r_alt)]:
        for p in rr.params.index:
            rob_rows.append({"spec": name, "param": p, "coef": float(rr.params[p]), "std_err": float(rr.bse[p]), "p_value": float(rr.pvalues[p])})
    robustness = pd.DataFrame(rob_rows)
    save_csv_and_latex(robustness, tables_dir / "table_pred_robustness.csv", tables_dir / "table_pred_robustness.tex")

    # collinearity
    Xv = d[["E_total", "dE_total"] + land_drivers].copy()
    vif = pd.DataFrame(
        {
            "variable": Xv.columns,
            "vif": [variance_inflation_factor(Xv.values, i) for i in range(Xv.shape[1])],
        }
    ).sort_values("vif", ascending=False)
    save_csv_and_latex(vif, tables_dir / "table_pred_collinearity_vif.csv", tables_dir / "table_pred_collinearity_vif.tex")

    corr = Xv.corr()
    pairs = []
    for i, c1 in enumerate(corr.columns):
        for j, c2 in enumerate(corr.columns):
            if j > i:
                pairs.append({"var1": c1, "var2": c2, "corr": float(corr.loc[c1, c2]), "abs_corr": float(abs(corr.loc[c1, c2]))})
    top_corr = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).head(10)
    save_csv_and_latex(top_corr, tables_dir / "table_pred_collinearity_top_corr.csv", tables_dir / "table_pred_collinearity_top_corr.tex")

    # residual diagnostics
    diag_rows = []
    diag_inputs = [
        ("Ocean sink", pd.Series(res_ocn.resid).dropna(), sm.add_constant(df_win.loc[pd.Series(res_ocn.resid).dropna().index, ["E_total", "dE_total"]], has_constant="add")),
        ("Land sink", pd.Series(res_lnd.resid).dropna(), sm.add_constant(df_win.loc[pd.Series(res_lnd.resid).dropna().index, ["E_total", "dE_total"] + land_drivers], has_constant="add")),
        ("Imbalance", pd.Series(res_bim.resid).dropna(), sm.add_constant(np.ones((len(pd.Series(res_bim.resid).dropna()), 1)), has_constant="add")),
        ("G_ATM prediction error", (series[GATM] - series["G_ATM_hat"]).dropna(), sm.add_constant(series[["G_ATM_hat"]], has_constant="add")),
    ]
    for name, resid, exog in diag_inputs:
        jb_s, jb_p, _, _ = jarque_bera(resid)
        try:
            bp_s, bp_p, _, _ = het_breuschpagan(resid, exog)
        except Exception:
            bp_p = np.nan
        try:
            wh_s, wh_p, _, _ = het_white(resid, exog)
        except Exception:
            wh_p = np.nan
        lb_p = float(acorr_ljungbox(resid, lags=[min(10, max(1, len(resid) // 3))], return_df=True)["lb_pvalue"].iloc[0])
        dw = float(durbin_watson(resid))
        diag_rows.append(
            {
                "series": name,
                "jb_p": float(jb_p),
                "bp_p": float(bp_p) if pd.notna(bp_p) else np.nan,
                "white_p": float(wh_p) if pd.notna(wh_p) else np.nan,
                "durbin_watson": dw,
                "ljung_box_p": lb_p,
                "flag_jb_pass_5pct": "pass" if jb_p > 0.05 else "fail",
                "flag_bp_pass_5pct": "pass" if (pd.notna(bp_p) and bp_p > 0.05) else "fail",
                "flag_white_pass_5pct": "pass" if (pd.notna(wh_p) and wh_p > 0.05) else "fail",
                "flag_dw_pass": "pass" if 1.5 <= dw <= 2.5 else "fail",
                "flag_lb_pass_5pct": "pass" if lb_p > 0.05 else "fail",
            }
        )
    diag_df = pd.DataFrame(diag_rows)
    save_csv_and_latex(diag_df, tables_dir / "table_pred_residual_diagnostics.csv", tables_dir / "table_pred_residual_diagnostics.tex")

    # influence
    infl = OLSInfluence(ols_base)
    infl_df = pd.DataFrame(
        {
            "year": d[YEAR].values,
            "cooks_d": infl.cooks_distance[0],
            "leverage": infl.hat_matrix_diag,
            "dffits": infl.dffits[0],
            "student_resid": infl.resid_studentized_external,
        }
    )
    infl_top = infl_df.reindex(infl_df.cooks_d.abs().sort_values(ascending=False).index).head(10)
    save_csv_and_latex(infl_top, tables_dir / "table_pred_influence_top10.csv", tables_dir / "table_pred_influence_top10.tex")

    # rolling error summary
    ser = series.dropna(subset=[GATM, "G_ATM_hat"]).copy()
    ser["err"] = ser[GATM] - ser["G_ATM_hat"]
    rw = max(5, min(10, len(ser)))
    ser["rolling_rmse"] = ser["err"].pow(2).rolling(rw).mean().pow(0.5)
    ser["rolling_mae"] = ser["err"].abs().rolling(rw).mean()
    roll_tab = pd.DataFrame(
        [
            {
                "window": rw,
                "rolling_rmse_mean": float(ser["rolling_rmse"].mean()),
                "rolling_rmse_max": float(ser["rolling_rmse"].max()),
                "rolling_mae_mean": float(ser["rolling_mae"].mean()),
                "rolling_mae_max": float(ser["rolling_mae"].max()),
                "last_available_year": int(ser[YEAR].dropna().max()),
            }
        ]
    )
    save_csv_and_latex(roll_tab, tables_dir / "table_pred_rolling_error_summary.csv", tables_dir / "table_pred_rolling_error_summary.tex")

    # ---- figures ----
    plt.style.use("ggplot")

    # 1) Actual vs fitted with simple 95% band
    fig, ax = plt.subplots(figsize=(9, 5))
    sigma = ser["err"].std()
    ax.plot(ser[YEAR], ser[GATM], label="Actual G_ATM", lw=2)
    ax.plot(ser[YEAR], ser["G_ATM_hat"], label="Fitted G_ATM", lw=2)
    ax.fill_between(ser[YEAR], ser["G_ATM_hat"] - 1.96 * sigma, ser["G_ATM_hat"] + 1.96 * sigma, alpha=0.2, label="Approx. 95% band")
    ax.set_title("Prediction model: Actual vs fitted atmospheric growth")
    ax.set_xlabel("Year")
    ax.set_ylabel("G_ATM (GtC/yr)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_actual_vs_fitted.png", dpi=180)
    plt.close(fig)

    # 2) Residuals vs fitted + LOWESS
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ser["G_ATM_hat"], ser["err"], alpha=0.7, label="Residuals")
    lw = lowess(ser["err"], ser["G_ATM_hat"], frac=0.6)
    ax.plot(lw[:, 0], lw[:, 1], color="black", lw=2, label="LOWESS")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_title("Residuals vs fitted values")
    ax.set_xlabel("Fitted G_ATM (GtC/yr)")
    ax.set_ylabel("Residual (GtC/yr)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_residuals_vs_fitted.png", dpi=180)
    plt.close(fig)

    # 3) QQ plot
    fig = qqplot(ser["err"], line="45", fit=True)
    fig.axes[0].set_title("Q-Q plot of G_ATM prediction residuals")
    fig.axes[0].set_xlabel("Theoretical quantiles")
    fig.axes[0].set_ylabel("Sample quantiles")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_qq_residuals.png", dpi=180)
    plt.close(fig)

    # 4) Scale-location
    fig, ax = plt.subplots(figsize=(8, 5))
    scl = np.sqrt(np.abs(ser["err"]))
    ax.scatter(ser["G_ATM_hat"], scl, alpha=0.7)
    lw2 = lowess(scl, ser["G_ATM_hat"], frac=0.6)
    ax.plot(lw2[:, 0], lw2[:, 1], color="black", lw=2)
    ax.set_title("Scale-location plot")
    ax.set_xlabel("Fitted G_ATM (GtC/yr)")
    ax.set_ylabel("sqrt(|Residual|)")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_scale_location.png", dpi=180)
    plt.close(fig)

    # 5) Residual autocorrelation + Ljung-Box p-values
    err = ser["err"].dropna()
    lmax = min(10, len(err) - 1)
    acf_vals = acf(err, nlags=lmax)
    pacf_vals = pacf(err, nlags=lmax, method="yw")
    lb_df = acorr_ljungbox(err, lags=list(range(1, lmax + 1)), return_df=True)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].stem(range(0, lmax + 1), acf_vals)
    axs[0].set_title("Residual ACF")
    axs[0].set_xlabel("Lag")
    axs[0].set_ylabel("ACF")

    axs[1].stem(range(0, lmax + 1), pacf_vals)
    axs[1].set_title("Residual PACF")
    axs[1].set_xlabel("Lag")
    axs[1].set_ylabel("PACF")

    axs[2].plot(lb_df.index, lb_df["lb_pvalue"], marker="o")
    axs[2].axhline(0.05, color="red", ls="--", lw=1)
    axs[2].set_title("Ljung-Box p-values")
    axs[2].set_xlabel("Lag")
    axs[2].set_ylabel("p-value")
    fig.suptitle("Residual autocorrelation diagnostics", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_residual_acf_pacf_ljungbox.png", dpi=180)
    plt.close(fig)

    # 6) rolling rmse/mae
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ser[YEAR], ser["rolling_rmse"], label=f"Rolling RMSE (window={rw})", lw=2)
    ax.plot(ser[YEAR], ser["rolling_mae"], label=f"Rolling MAE (window={rw})", lw=2)
    ax.set_title("Rolling prediction error metrics")
    ax.set_xlabel("Year")
    ax.set_ylabel("Error (GtC/yr)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_rolling_rmse_mae.png", dpi=180)
    plt.close(fig)

    # 7) coefficient paths + 95% CI ribbon for key betas
    key_params = [p for p in ["E_total", "dE_total"] + land_drivers[:2] if p in coef_stab["param"].unique()]
    fig, axs = plt.subplots(len(key_params), 1, figsize=(9, 3 * max(1, len(key_params))), sharex=True)
    if len(key_params) == 1:
        axs = [axs]
    for ax, kp in zip(axs, key_params):
        sub = coef_stab[coef_stab["param"] == kp].copy()
        x = sub["window_end_year"]
        yv = sub["coef"]
        hw = 0.5 * sub["ci_width"]
        ax.plot(x, yv, label=kp, lw=2)
        ax.fill_between(x, yv - hw, yv + hw, alpha=0.2)
        ax.set_ylabel("Coefficient")
        ax.legend(loc="best")
    axs[-1].set_xlabel("Window end year")
    fig.suptitle("Rolling-window coefficient paths (95% CI)", y=1.01)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_coefficient_paths.png", dpi=180)
    plt.close(fig)

    # 8) partial regression grid
    fig = plt.figure(figsize=(12, 8))
    plot_partregress_grid(ols_base, fig=fig)
    fig.suptitle("Partial regression diagnostics for land sink model", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_partial_regression_grid.png", dpi=180)
    plt.close(fig)

    # 9) Cook's distance
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stem(d[YEAR], infl.cooks_distance[0])
    ax.set_title("Cook's distance by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cook's D")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_cooks_distance.png", dpi=180)
    plt.close(fig)

    # 10) leverage vs residual squared with influence contours
    fig, ax = plt.subplots(figsize=(8, 6))
    influence_plot(ols_base, ax=ax, criterion="cooks")
    ax.set_title("Leverage vs residuals squared (Cook's contours)")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pred_leverage_vs_resid2.png", dpi=180)
    plt.close(fig)
