from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.viz.savefig import savefig


def plot_rcp_atm_growth(rcp: pd.DataFrame, outpath: Path, title: str = "") -> None:
    plt.figure(figsize=(11, 4.8))
    plt.plot(rcp["Year"], rcp["G_ATM_rcp_GtC"])
    plt.axhline(0, linewidth=1)
    plt.title(title or "RCP implied atmospheric growth")
    plt.xlabel("Year")
    plt.ylabel("GtC/year")
    plt.grid(True, alpha=0.3)
    savefig(outpath)
