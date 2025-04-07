#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

# read from /maps/jh2589/exante/parquets
# write csv to /maps/jh2589/exante/csvs
PARQUET_DIR = Path("/maps/jh2589/exante/parquets")
CSV_DIR = Path("/maps/jh2589/exante/csvs")
CSV_DIR.mkdir(exist_ok=True)
OUTPUT_CSV = CSV_DIR / "project_rates.csv"

def annual_rate(start, end):
    return -100 * ((end / start) ** (1 / 10) - 1)

rows = []
for matches_pq in sorted(PARQUET_DIR.glob("*_matches.parquet")):
    if "_expost_" in matches_pq.name:
        continue

    project = matches_pq.stem.replace("_matches", "")
    regional_pq = PARQUET_DIR / f"{project}_regional.parquet"
    expost_pq   = PARQUET_DIR / f"{project}_expost_matches.parquet"

    # project and time-lagged ex-ante
    df = pd.read_parquet(
        matches_pq,
        columns=["k_luc_10", "k_luc_0", "k_luc_-10", "s_luc_10", "s_luc_0"]
    )
    k_rate = annual_rate(
        df["k_luc_0"].eq(1).sum(),
        df["k_luc_10"].eq(1).sum()
    )
    k_hist_rate = annual_rate(
        df["k_luc_-10"].eq(1).sum(),
        df["k_luc_0"].eq(1).sum()
    )
    s_exante_rate = annual_rate(
        df["s_luc_0"].eq(1).sum(),
        df["s_luc_10"].eq(1).sum()
    )

    # regional ex-ante
    if regional_pq.exists():
        reg = pd.read_parquet(regional_pq, columns=["luc_2001", "luc_2011"])
        regional_rate = annual_rate(
            reg["luc_2001"].eq(1).sum(),
            reg["luc_2011"].eq(1).sum()
        )
    else:
        regional_rate = np.nan

    # ex-post matched
    if expost_pq.exists():
        exp = pd.read_parquet(expost_pq, columns=["s_luc_2011", "s_luc_2021"])
        s_expost_rate = annual_rate(
            exp["s_luc_2011"].eq(1).sum(),
            exp["s_luc_2021"].eq(1).sum()
        )
    else:
        s_expost_rate = np.nan

    rows.append({
        "project": project,
        "k_rate": k_rate,
        "k_exante_rate": k_hist_rate,
        "s_exante_rate": s_exante_rate,
        "regional_exante_rate": regional_rate,
        "s_expost_rate": s_expost_rate
    })

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
print(f"saved updated rates csv to {OUTPUT_CSV}")
