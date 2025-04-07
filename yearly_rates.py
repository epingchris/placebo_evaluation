#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path

# prompt for the baseline year
default_baseline = 2001
user_input = input(
    f"which baseline year do you want for project/regional ex-ante? (default={default_baseline}): "
).strip()

if user_input == "":
    baseline_year = default_baseline
else:
    baseline_year = int(user_input)

# number of years from baseline_year to 2011
n_years_exante = 2011 - baseline_year

# set up paths for parquets and csvs
PARQUET_DIR = Path("/maps/jh2589/exante/parquets")
CSV_DIR = Path("/maps/jh2589/exante/csvs")
CSV_DIR.mkdir(exist_ok=True)

OUTPUT_K             = CSV_DIR / "k_rates.csv"
OUTPUT_K_EXANTE      = CSV_DIR / "k_exante_rates.csv"
OUTPUT_S_EXANTE      = CSV_DIR / "s_exante_rates.csv"
OUTPUT_REGIONAL      = CSV_DIR / "regional_exante_rates.csv"
OUTPUT_S_EXPOST      = CSV_DIR / "s_expost_rates.csv"

def annual_rate(start, end, n_years):
    """
    annualised percent rate of change, applying '-100 * (...)'.
    returns nan if start=0, missing, or n_years <= 0.
    """
    if n_years <= 0 or start == 0 or pd.isna(start) or pd.isna(end):
        return np.nan
    return -100.0 * ((end / start) ** (1.0 / n_years) - 1.0)

# we will store rows for each approach
k_rows               = []
k_exante_rows        = []
s_exante_rows        = []
regional_exante_rows = []
s_expost_rows        = []

# for the "k" approach, "s_exante," and "s_expost," we do periods=1..10
# => years 2012..2021 from the 2011 baseline
periods = range(1, 11)

for matches_pq in sorted(PARQUET_DIR.glob("*_matches.parquet")):
    # skip the _expost_ ones
    if "_expost_" in matches_pq.name:
        continue

    project = matches_pq.stem.replace("_matches", "")
    regional_pq = PARQUET_DIR / f"{project}_regional.parquet"
    expost_pq   = PARQUET_DIR / f"{project}_expost_matches.parquet"

    # read the main matches file
    df = pd.read_parquet(matches_pq)

    # (a) baselines for k / s_exante
    if "k_luc_0" in df:
        k_start_count = df["k_luc_0"].eq(1).sum()
    else:
        k_start_count = np.nan

    if "s_luc_0" in df:
        s_exante_start_count = df["s_luc_0"].eq(1).sum()
    else:
        s_exante_start_count = np.nan

    # (b) project ex-ante => user-chosen baseline
    project_period = baseline_year - 2011
    project_col_baseline = f"k_luc_{project_period}"
    if (project_col_baseline in df) and ("k_luc_0" in df):
        k_ex_start = df[project_col_baseline].eq(1).sum()
        k_ex_end   = df["k_luc_0"].eq(1).sum()
        k_exante_rate_single = annual_rate(k_ex_start, k_ex_end, n_years_exante)
    else:
        k_exante_rate_single = np.nan

    # (c) regional ex-ante => luc_{baseline_year}.. luc_2011
    if regional_pq.exists():
        reg_df = pd.read_parquet(regional_pq)
        region_baseline_col = f"luc_{baseline_year}"
        if (region_baseline_col in reg_df) and ("luc_2011" in reg_df):
            reg_start = reg_df[region_baseline_col].eq(1).sum()
            reg_end   = reg_df["luc_2011"].eq(1).sum()
            regional_ex_rate_single = annual_rate(reg_start, reg_end, n_years_exante)
        else:
            regional_ex_rate_single = np.nan
    else:
        regional_ex_rate_single = np.nan

    # (d) s_expost => baseline = s_luc_2011
    if expost_pq.exists():
        exp_df = pd.read_parquet(expost_pq)
        if "s_luc_2011" in exp_df:
            s_expost_start_count = exp_df["s_luc_2011"].eq(1).sum()
        else:
            s_expost_start_count = np.nan
    else:
        exp_df = None
        s_expost_start_count = np.nan

    # build row dicts for each approach
    k_row               = {"project": project}
    k_exante_row        = {"project": project}
    s_exante_row        = {"project": project}
    regional_exante_row = {"project": project}
    s_expost_row        = {"project": project}

    # loop periods = 1..10 => 2012..2021
    for period in periods:
        year_label = 2011 + period
        colname = f"rate_{year_label}"

        # (1) k approach
        end_col_k = f"k_luc_{period}"
        if end_col_k in df and not pd.isna(k_start_count):
            end_count_k = df[end_col_k].eq(1).sum()
            k_row[colname] = annual_rate(k_start_count, end_count_k, period)
        else:
            k_row[colname] = np.nan

        # (2) k_exante => single user-chosen baseline rate
        k_exante_row[colname] = k_exante_rate_single

        # (3) s_exante => baseline = s_luc_0 => 2011->(2011+period)
        end_col_s = f"s_luc_{period}"
        if end_col_s in df and not pd.isna(s_exante_start_count):
            s_end_count = df[end_col_s].eq(1).sum()
            s_exante_row[colname] = annual_rate(s_exante_start_count, s_end_count, period)
        else:
            s_exante_row[colname] = np.nan

        # (4) regional_exante => single user-chosen baseline rate
        regional_exante_row[colname] = regional_ex_rate_single

        # (5) s_expost => baseline= s_luc_2011 => 2011->(2011+period)
        if exp_df is not None:
            exp_end_col = f"s_luc_{year_label}"
            if exp_end_col in exp_df and not pd.isna(s_expost_start_count):
                end_count = exp_df[exp_end_col].eq(1).sum()
                s_expost_row[colname] = annual_rate(s_expost_start_count, end_count, period)
            else:
                s_expost_row[colname] = np.nan
        else:
            s_expost_row[colname] = np.nan

    # store each approach row
    k_rows.append(k_row)
    k_exante_rows.append(k_exante_row)
    s_exante_rows.append(s_exante_row)
    regional_exante_rows.append(regional_exante_row)
    s_expost_rows.append(s_expost_row)

# convert to dataframes and write csvs
k_df               = pd.DataFrame(k_rows)
k_exante_df        = pd.DataFrame(k_exante_rows)
s_exante_df        = pd.DataFrame(s_exante_rows)
regional_exante_df = pd.DataFrame(regional_exante_rows)
s_expost_df        = pd.DataFrame(s_expost_rows)

k_df.to_csv(OUTPUT_K, index=False)
k_exante_df.to_csv(OUTPUT_K_EXANTE, index=False)
s_exante_df.to_csv(OUTPUT_S_EXANTE, index=False)
regional_exante_df.to_csv(OUTPUT_REGIONAL, index=False)
s_expost_df.to_csv(OUTPUT_S_EXPOST, index=False)

print("\n✅ done! wrote five csvs into", CSV_DIR)
print(f"using baseline_year={baseline_year} => {n_years_exante} years for project and regional ex-ante.")