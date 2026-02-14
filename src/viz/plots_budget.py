from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.constants import YEAR, EFF, ELUC, SOCN, SLND, GATM, BIM
from src.viz.savefig import savefig


def plot_budget_components(df_budget: pd.DataFrame, outpath: Path) -> None:
    df = df_budget.sort_values(YEAR).copy()

    plt.figure(figsize=(11, 6))
    plt.plot(df[YEAR], df[EFF],  label="Fossil fuel & industry (E_FF)")
    plt.plot(df[YEAR], df[ELUC], label="Land-use change (E_LUC)")
    plt.plot(df[YEAR], df[SOCN], label="Ocean sink (S_OCN)")
    plt.plot(df[YEAR], df[SLND], label="Land sink (S_LND)")
    plt.plot(df[YEAR], df[GATM], label="Atmospheric growth (G_ATM)")
    plt.title("Global Carbon Budget components (GtC/year)")
    plt.xlabel("Year")
    plt.ylabel("GtC/year")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outpath)


def plot_imbalance(df_budget: pd.DataFrame, outpath: Path) -> None:
    df = df_budget.sort_values(YEAR).copy()

    plt.figure(figsize=(11, 4.8))
    plt.plot(df[YEAR], df[BIM], label="Budget imbalance")
    plt.axhline(0, linewidth=1)
    plt.title("Budget imbalance over time (GtC/year)")
    plt.xlabel("Year")
    plt.ylabel("GtC/year")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outpath)
