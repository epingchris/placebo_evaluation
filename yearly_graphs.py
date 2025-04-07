#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# set up paths for csvs and plots
CSV_DIR = Path("/maps/jh2589/exante/csvs")
PLOT_DIR = Path("/maps/jh2589/exante/plots")
PLOT_DIR.mkdir(exist_ok=True)

fn_k        = CSV_DIR / "k_rates.csv"
fn_k_exante = CSV_DIR / "k_exante_rates.csv"
fn_s_exante = CSV_DIR / "s_exante_rates.csv"
fn_reg      = CSV_DIR / "regional_exante_rates.csv"
fn_s_expost = CSV_DIR / "s_expost_rates.csv"

out_csv      = CSV_DIR / "goodness_of_fit.csv"
out_png_r2   = PLOT_DIR / "goodness_of_fit_r2.png"
out_png_mae  = PLOT_DIR / "goodness_of_fit_mae.png"
out_png_bias = PLOT_DIR / "goodness_of_fit_bias.png"

# read all csvs
k_df        = pd.read_csv(fn_k)
k_exante_df = pd.read_csv(fn_k_exante)
s_exante_df = pd.read_csv(fn_s_exante)
reg_df      = pd.read_csv(fn_reg)
s_expost_df = pd.read_csv(fn_s_expost)

# merge on 'project' so all data line up
df_merged = k_df.merge(k_exante_df, on='project', suffixes=('', '_kex'))
df_merged = df_merged.merge(s_exante_df, on='project', suffixes=('', '_sex'))
df_merged = df_merged.merge(reg_df, on='project', suffixes=('', '_reg'))
df_merged = df_merged.merge(s_expost_df, on='project', suffixes=('', '_sexp'))

def r2_identity(observed, predicted):
    pairs = pd.DataFrame({'obs': observed, 'pred': predicted}).dropna()
    obs = pairs['obs']
    pred = pairs['pred']
    sse = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - sse / sst

def mae(observed, predicted):
    pairs = pd.DataFrame({'obs': observed, 'pred': predicted}).dropna()

    return np.mean(np.abs(pairs['obs'] - pairs['pred']))

def bias(observed, predicted):
    pairs = pd.DataFrame({'obs': observed, 'pred': predicted}).dropna()
    return np.mean(pairs['pred'] - pairs['obs'])

# compute r2, mae, and bias for each year and each method
methods = {
    'k_exante':         '_kex',
    's_exante':         '_sex',
    'regional_exante':  '_reg',
    's_expost':         '_sexp',
}

results = []
for year in range(2012, 2022):
    col_obs = f"rate_{year}"
    row = {'year': year}
    for method_key, suffix in methods.items():
        col_pred = f"rate_{year}{suffix}"
        r2_val = r2_identity(df_merged[col_obs], df_merged[col_pred])
        mae_val = mae(df_merged[col_obs], df_merged[col_pred])
        bias_val = bias(df_merged[col_obs], df_merged[col_pred])
        row[f"{method_key}_r2"] = r2_val
        row[f"{method_key}_mae"] = mae_val
        row[f"{method_key}_bias"] = bias_val
    results.append(row)

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(out_csv, index=False)
print(f"wrote r², mae, and bias to {out_csv}")

years = metrics_df['year']

# r2 plot
plt.figure(figsize=(6,6))
plt.plot(years, metrics_df['s_expost_r2'], label='Ex Post matching')
plt.plot(years, metrics_df['k_exante_r2'], label='Ex Ante project')
plt.plot(years, metrics_df['s_exante_r2'], label='Ex Ante time shifted')
plt.plot(years, metrics_df['regional_exante_r2'], label='Ex Ante regional')
plt.xlabel('Year')
plt.ylabel('R² (Identity Line)')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.tight_layout()
plt.savefig(out_png_r2, dpi=150, bbox_inches='tight')
plt.show()
print(f"saved r² plot to {out_png_r2}")

# mae plot
plt.figure(figsize=(6,6))
plt.plot(years, metrics_df['s_expost_r2'], label='Ex Post matching')
plt.plot(years, metrics_df['k_exante_r2'], label='Ex Ante project')
plt.plot(years, metrics_df['s_exante_r2'], label='Ex Ante time shifted')
plt.plot(years, metrics_df['regional_exante_r2'], label='Ex Ante regional')
plt.xlabel('Year')
plt.ylabel('MAE')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.tight_layout()
plt.savefig(out_png_mae, dpi=150, bbox_inches='tight')
plt.show()
print(f"saved mae plot to {out_png_mae}")

# bias plot
plt.figure(figsize=(6,6))
plt.plot(years, metrics_df['s_expost_r2'], label='Ex Post matching')
plt.plot(years, metrics_df['k_exante_r2'], label='Ex Ante project')
plt.plot(years, metrics_df['s_exante_r2'], label='Ex Ante time shifted')
plt.plot(years, metrics_df['regional_exante_r2'], label='Ex Ante regional')

# dashed horizontal line at 0
plt.axhline(0, color='gray', linestyle='--')

plt.xlabel('Year')
plt.ylabel('Bias')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.tight_layout()
plt.savefig(out_png_bias, dpi=150, bbox_inches='tight')
plt.show()
print(f"saved bias plot to {out_png_bias}")