from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.viz.savefig import savefig


SCENARIO_COLORS = {
    "baseline": "#1f77b4",
    "bad_climate": "#d62728",
    "good_climate": "#2ca02c",
}

MODEL_STYLES = {
    "linear": "-",
    "saturation": "--",
}


def _spec_parts(spec: str) -> tuple[str, str]:
    if "|" not in spec:
        return spec.strip(), "linear"
    left, right = [s.strip() for s in spec.split("|", 1)]
    model = "saturation" if "sat(" in right else "linear"
    return left, model


def _label_from_spec(spec: str) -> str:
    scenario, model = _spec_parts(spec)
    if model == "linear":
        return f"{scenario} · linear"
    return spec.replace("|", "·")


def plot_rcp_atm_growth(rcp: pd.DataFrame, outpath: Path, title: str = "") -> None:
    plt.figure(figsize=(11, 4.8))
    plt.plot(rcp["Year"], rcp["G_ATM_rcp_GtC"], color="#333333", linewidth=2)
    plt.axhline(0, linewidth=1, color="black", alpha=0.8)
    plt.title(title or "RCP implied atmospheric growth")
    plt.xlabel("Year")
    plt.ylabel("Implied atmospheric growth (GtC/year)")
    plt.grid(True, alpha=0.3)
    savefig(outpath)


def plot_rcp_emissions_paths(paths_rcp: dict[str, pd.DataFrame], outpath: Path, title: str = "") -> None:
    plt.figure(figsize=(12, 6))

    for spec, df in sorted(paths_rcp.items()):
        scenario, model = _spec_parts(spec)
        color = SCENARIO_COLORS.get(scenario, "#7f7f7f")
        linestyle = MODEL_STYLES.get(model, "-")
        linewidth = 2.5 if model == "linear" else 1.6
        alpha = 0.95 if model == "linear" else 0.8

        plt.plot(
            df["Year"],
            df["E_total"],
            label=_label_from_spec(spec),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
        )

    plt.axhline(0, color="black", linewidth=1, alpha=0.8)
    plt.title(title or "RCP-implied admissible total emissions")
    plt.xlabel("Year")
    plt.ylabel("Total emissions E_total (GtC/year)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", frameon=True, ncol=2, fontsize=9)
    savefig(outpath)


def plot_linear_vs_saturation_for_scenario(
    paths_rcp: dict[str, pd.DataFrame], scenario: str, kappas: list[float], outpath: Path
) -> None:
    plt.figure(figsize=(11, 5.5))

    key_linear = f"{scenario} | linear"
    if key_linear in paths_rcp:
        df_lin = paths_rcp[key_linear]
        plt.plot(
            df_lin["Year"],
            df_lin["E_total"],
            label="linear",
            color=SCENARIO_COLORS.get(scenario, "#1f77b4"),
            linestyle="-",
            linewidth=2.6,
        )

    for k in kappas:
        key_sat = f"{scenario} | sat(k={k})"
        if key_sat in paths_rcp:
            df_sat = paths_rcp[key_sat]
            plt.plot(
                df_sat["Year"],
                df_sat["E_total"],
                label=f"saturation (k={k:.3f})",
                color=SCENARIO_COLORS.get(scenario, "#1f77b4"),
                linestyle="--",
                linewidth=1.8,
                alpha=0.85,
            )

    plt.axhline(0, color="black", linewidth=1, alpha=0.8)
    plt.title(f"{scenario.replace('_', ' ').title()}: linear vs saturation assumptions")
    plt.xlabel("Year")
    plt.ylabel("Total emissions E_total (GtC/year)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", frameon=True)
    savefig(outpath)


def plot_rcp_emissions_scenario_envelope(paths_rcp: dict[str, pd.DataFrame], outpath: Path) -> None:
    rows = []
    for spec, df in paths_rcp.items():
        tmp = df[["Year", "E_total"]].copy()
        tmp["spec"] = spec
        tmp["scenario"] = _spec_parts(spec)[0]
        rows.append(tmp)

    if not rows:
        return

    long_df = pd.concat(rows, ignore_index=True)
    grouped = (
        long_df.groupby(["scenario", "Year"], as_index=False)
        .agg(E_min=("E_total", "min"), E_max=("E_total", "max"), E_median=("E_total", "median"))
        .sort_values(["scenario", "Year"])
    )

    plt.figure(figsize=(12, 6))
    for scenario, dfg in grouped.groupby("scenario"):
        color = SCENARIO_COLORS.get(scenario, "#7f7f7f")
        plt.fill_between(dfg["Year"], dfg["E_min"], dfg["E_max"], color=color, alpha=0.15)
        plt.plot(dfg["Year"], dfg["E_median"], color=color, linewidth=2.2, label=f"{scenario} median")

    plt.axhline(0, color="black", linewidth=1, alpha=0.8)
    plt.title("RCP implied emissions envelope by climate scenario")
    plt.xlabel("Year")
    plt.ylabel("Total emissions E_total (GtC/year)")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", frameon=True)
    savefig(outpath)
