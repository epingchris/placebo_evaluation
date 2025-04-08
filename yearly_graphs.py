#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# set up paths for csvs and plots
CSV_DIR = Path('/maps/jh2589/exante/csvs')
PLOT_DIR = Path('/maps/epr26/placebo_evaluation_out')
PLOT_DIR.mkdir(exist_ok = True)

fn_k        = CSV_DIR / 'k_rates.csv'
fn_k_exante = CSV_DIR / 'k_exante_rates.csv'
fn_s_exante = CSV_DIR / 's_exante_rates.csv'
fn_reg      = CSV_DIR / 'regional_exante_rates.csv'
fn_s_expost = CSV_DIR / 's_expost_rates.csv'

out_csv          = CSV_DIR / 'goodness_of_fit.csv'
out_png_mae      = PLOT_DIR / 'out_figure5a_mae_new.png'
out_png_bias     = PLOT_DIR / 'out_figure5b_bias_new.png'
out_png_r2       = PLOT_DIR / 'out_figure5c_r2_new.png'
out_png_combined = PLOT_DIR / 'out_figure5_combined.png'

# read all csvs
k_df        = pd.read_csv(fn_k)
k_exante_df = pd.read_csv(fn_k_exante)
s_exante_df = pd.read_csv(fn_s_exante)
reg_df      = pd.read_csv(fn_reg)
s_expost_df = pd.read_csv(fn_s_expost)

# merge on 'project' so all data line up
df_merged = k_df.merge(k_exante_df, on = 'project', suffixes = ('', '_kex'))
df_merged = df_merged.merge(s_exante_df, on = 'project', suffixes = ('', '_sex'))
df_merged = df_merged.merge(reg_df, on = 'project', suffixes = ('', '_reg'))
df_merged = df_merged.merge(s_expost_df, on = 'project', suffixes = ('', '_sexp'))

def mae(observed, predicted):
    pairs = pd.DataFrame({'obs': observed, 'pred': predicted}).dropna()

    return np.mean(np.abs(pairs['obs'] - pairs['pred']))

def bias(observed, predicted):
    pairs = pd.DataFrame({'obs': observed, 'pred': predicted}).dropna()
    return np.mean(pairs['pred'] - pairs['obs'])

def r2_identity(observed, predicted):
    pairs = pd.DataFrame({'obs': observed, 'pred': predicted}).dropna()
    obs = pairs['obs']
    pred = pairs['pred']
    sse = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - sse / sst

# compute mae, bias, and r2 for each year and each method
methods = {
    'k_exante':         '_kex',
    's_exante':         '_sex',
    'regional_exante':  '_reg',
    's_expost':         '_sexp',
}

results = []
for year in range(2012, 2022):
    col_obs = f'rate_{year}'
    row = {'year': year}
    for method_key, suffix in methods.items():
        col_pred = f'rate_{year}{suffix}'
        mae_val = mae(df_merged[col_obs], df_merged[col_pred])
        bias_val = bias(df_merged[col_obs], df_merged[col_pred])
        r2_val = r2_identity(df_merged[col_obs], df_merged[col_pred])
        row[f'{method_key}_mae'] = mae_val
        row[f'{method_key}_bias'] = bias_val
        row[f'{method_key}_r2'] = r2_val
    results.append(row)

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(out_csv, index=False)
print(f'wrote r², mae, and bias to {out_csv}')

years = metrics_df['year']

# Set graphic parameters
plt.rc('figure', figsize=(8, 4))
plt.rc('axes', titlesize = 16, labelsize = 14)
plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
plt.rc('legend', fontsize = 12)
plt.rc('savefig', dpi = 150)

# Set graphic labels
metrics = ['mae', 'bias', 'r2']
titles = ['A. Mean absolute error (MAE)', 'B. Mean bias', 'C. Goodness-of-fit']
ylabels = ['MAE', 'Bias', 'R² over the identity line']
out_paths = [out_png_mae, out_png_bias, out_png_r2]

n = len(metrics)
fig, axs = plt.subplots(n, 1, figsize=(8, 4 * n), sharex = True)

for i, metric in enumerate(metrics): # plot for each metric
    ax = axs[i]
    ax.plot(years, metrics_df[f'regional_exante_{metric}'], label = 'Ex ante regional', color = '#006CD1', linewidth = 1)
    ax.plot(years, metrics_df[f'k_exante_{metric}'], label = 'Ex ante project', color = '#40B0A6', linewidth = 1)
    ax.plot(years, metrics_df[f's_exante_{metric}'], label = 'Ex ante time-shifted', color = '#CDAC60', linewidth = 1)
    ax.plot(years, metrics_df[f's_expost_{metric}'], label = 'Ex post matching', color = '#C13C3C', linestyle = 'dashed', linewidth = 2)
    if metric == 'bias':
        ax.axhline(0, color='gray', linestyle='--') # dashed horizontal line at 0 for the bias plot
    ax.set_title(titles[i])
    ax.set_ylabel(ylabels[i])

axs[-1].set_xlabel('Year')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc = 'center left', bbox_to_anchor = (1.01, 0.5), ncol = 1)
fig.tight_layout()
fig.savefig(out_png_combined, bbox_inches = 'tight')
print(f'Saved combined plot to {out_png_combined}')