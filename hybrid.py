#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# config and paths
PARQUET_DIR = Path('/maps/jh2589/exante/parquets')
OBSERVED_CSV = Path('/maps/jh2589/exante/csvs/k_rates.csv')

PROJECT_BASELINES  = range(2001, 2012)
REGIONAL_BASELINES = range(2001, 2012)
YEARS_POST = range(1, 11)
YEARS_LABELS = [2011 + x for x in YEARS_POST]

# helper functions
def annual_rate(start, end, n_years):
    if n_years <= 0 or start == 0 or pd.isna(start) or pd.isna(end):
        return np.nan
    return -100.0 * ((end / start)**(1.0 / n_years) - 1.0)

def mae(obs, pred):
    df_temp = pd.DataFrame({'obs': obs, 'pred': pred}).dropna()
    return np.mean(np.abs(df_temp['obs'] - df_temp['pred'])) if len(df_temp) else np.nan

def bias(obs, pred):
    df_temp = pd.DataFrame({'obs': obs, 'pred': pred}).dropna()
    return np.mean(df_temp['pred'] - df_temp['obs']) if len(df_temp) else np.nan

def r2_identity(obs, pred):
    df_temp = pd.DataFrame({'obs': obs, 'pred': pred}).dropna()
    if len(df_temp) < 2:
        return np.nan
    sse = np.sum((df_temp['obs'] - df_temp['pred'])**2)
    sst = np.sum((df_temp['obs'] - df_temp['obs'].mean())**2)
    return 1 - sse / sst if sst != 0 else np.nan

# load observed data
df_obs = pd.read_csv(OBSERVED_CSV)

# get project list
all_projects = []
for f in PARQUET_DIR.glob('*_matches.parquet'):
    if '_expost_' in f.name:
        continue
    project_id = f.stem.replace('_matches', '')
    all_projects.append(project_id)
all_projects = sorted(list(set(all_projects)))

# loop over all baseline combinations
results_all = []
combination_predictions = {}

for pb in PROJECT_BASELINES:
    for rb in REGIONAL_BASELINES:
        rows_data = []
        for project in all_projects:
            row_dict = {'project': project}
            matches_file  = PARQUET_DIR / f'{project}_matches.parquet'
            regional_file = PARQUET_DIR / f'{project}_regional.parquet'
            if not (matches_file.exists() and regional_file.exists()):
                for yr in YEARS_LABELS:
                    row_dict[f'rate_{yr}_proj'] = np.nan
                    row_dict[f'rate_{yr}_reg']  = np.nan
                    row_dict[f'rate_{yr}_hyb']  = np.nan
                rows_data.append(row_dict)
                continue

            df_matches  = pd.read_parquet(matches_file)
            df_regional = pd.read_parquet(regional_file)

            col_proj_baseline = f'k_luc_{pb - 2011}'
            if col_proj_baseline in df_matches.columns and 'k_luc_0' in df_matches.columns:
                proj_start = df_matches[col_proj_baseline].eq(1).sum()
                proj_end   = df_matches['k_luc_0'].eq(1).sum()
                n_years_exante_proj = 2011 - pb
                proj_rate = annual_rate(proj_start, proj_end, n_years_exante_proj)
            else:
                proj_rate = np.nan

            if (f'luc_{rb}' in df_regional.columns) and ('luc_2011' in df_regional.columns):
                reg_start = df_regional[f'luc_{rb}'].eq(1).sum()
                reg_end   = df_regional['luc_2011'].eq(1).sum()
                n_years_exante_reg = 2011 - rb
                reg_rate = annual_rate(reg_start, reg_end, n_years_exante_reg)
            else:
                reg_rate = np.nan

            for offset in YEARS_POST:
                yr = 2011 + offset
                col_proj = f'rate_{yr}_proj'
                col_reg  = f'rate_{yr}_reg'
                col_hyb  = f'rate_{yr}_hyb'
                row_dict[col_proj] = proj_rate
                row_dict[col_reg]  = reg_rate

                w = (yr - 2011) / (2021 - 2011)
                if pd.isna(proj_rate) or pd.isna(reg_rate) or (proj_rate <= 0) or (reg_rate <= 0):
                    row_dict[col_hyb] = np.nan
                else:
                    row_dict[col_hyb] = proj_rate**(1 - w) * reg_rate**w

            rows_data.append(row_dict)

        df_comb = pd.DataFrame(rows_data)
        df_merged = df_obs.merge(df_comb, on='project', how='inner')

        yearwise_mae = []
        yearwise_bias = []
        yearwise_r2 = []
        for yr in YEARS_LABELS:
            col_obs = f'rate_{yr}'
            col_hyb = f'rate_{yr}_hyb'
            mae_val = mae(df_merged[col_obs], df_merged[col_hyb])
            bias_val = bias(df_merged[col_obs], df_merged[col_hyb])
            r2_val = r2_identity(df_merged[col_obs], df_merged[col_hyb])
            if not pd.isna(mae_val):
                yearwise_mae.append(mae_val)
            if not pd.isna(bias_val):
                yearwise_bias.append(bias_val)
            if not pd.isna(r2_val):
                yearwise_r2.append(r2_val)
        avg_mae  = np.mean(yearwise_mae)  if len(yearwise_mae)  else np.nan
        avg_bias = np.mean(yearwise_bias) if len(yearwise_bias) else np.nan
        avg_r2 = np.mean(yearwise_r2) if len(yearwise_r2) else np.nan

        results_all.append({
            'proj_baseline': pb,
            'reg_baseline':  rb,
            'avg_mae':       avg_mae,
            'avg_bias':      avg_bias,
            'avg_r2':        avg_r2,
            'df_merged':     df_merged
        })

        combination_predictions[(pb, rb)] = df_merged

df_results = pd.DataFrame(results_all)
RESULTS_CSV = '/maps/epr26/placebo_evaluation_out/hybrid_results.csv'
df_results.to_csv(RESULTS_CSV, index = False)

# find best by average r2
results_sorted = sorted(results_all, key=lambda d: d['avg_r2'] if not pd.isna(d['avg_r2']) else -9999, reverse=True)
best = results_sorted[0]
best_pb = best['proj_baseline']
best_rb = best['reg_baseline']
best_r2 = best['avg_r2']
df_best = best['df_merged']

# save csv for best result
OUT_CSV = '/maps/jh2589/exante/csvs/hybrid_exante_rates.csv'
df_best.to_csv(OUT_CSV, index=False)
