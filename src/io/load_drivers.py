from pathlib import Path
import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    df["year"] = df["year"].astype(int)
    return df


def load_driver_csvs(processed_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "burned": processed_dir / "DM_Burned_97_22.csv",
        "nino":   processed_dir / "nino34_yearly_stats.csv",
        "pdsi":   processed_dir / "scPDSI_yearly_global_stats.csv",
        "tau":    processed_dir / "tau_yearly_stats.csv",
    }
    dfs = {k: _clean(pd.read_csv(v)) for k, v in files.items()}
    return dfs
