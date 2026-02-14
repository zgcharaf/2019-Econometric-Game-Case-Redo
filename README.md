# carbon-budget

Minimal reproducible pipeline for Global Carbon Budget + climate drivers + SARIMAX sink models + implied G_ATM + RCP inversion.

## Quick start

1) Put your data in place (or symlink):
- `data/raw/Global_Carbon_Budget_2018v1.0.xlsx`
- `data/processed/DM_Burned_97_22.csv`
- `data/processed/nino34_yearly_stats.csv`
- `data/processed/scPDSI_yearly_global_stats.csv`
- `data/processed/tau_yearly_stats.csv`
- `data/raw/RCP3PD_MIDYR_CONC.DAT` (optional, for projections)

2) Install deps:
```bash
pip install -r requirements.txt
```

3) Run pipeline:
```bash
python -m src.pipeline run --start 1989 --end 2012 --drivers nino34_max_z,scpdsi_global_max_z,tau_global_mean_z
```

Outputs go to `outputs/runs/<timestamp>/`.
