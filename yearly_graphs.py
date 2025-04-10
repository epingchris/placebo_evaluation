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
fn_hybrid   = CSV_DIR / 'hybrid_exante_rates.csv'

# Prompt to include hybrid approach
include_hybrid = input("Include hybrid ex ante approach in plots? (y/n, default=n): ").strip().lower()
include_hybrid = include_hybrid.startswith('y')

# Conditionally load hybrid rates
if include_hybrid:
    if fn_hybrid.exists():
        hybrid_df = pd.read_csv(fn_hybrid)
        print(f"Loaded hybrid ex ante rates: {len(hybrid_df)} rows")
    else:
        print(f"Warning: Hybrid file {fn_hybrid} not found. Continuing without hybrid approach.")
        include_hybrid = False

out_csv_suffix = '_with_hybrid' if include_hybrid else ''
fig_num = '10' if include_hybrid else '5'
out_csv          = CSV_DIR / f'goodness_of_fit{out_csv_suffix}.csv'
out_png_mae      = PLOT_DIR / f'out_figure{fig_num}a_mae.png'
out_png_bias     = PLOT_DIR / f'out_figure{fig_num}b_bias.png'
out_png_r2       = PLOT_DIR / f'out_figure{fig_num}c_r2.png'
out_png_combined = PLOT_DIR / f'out_figure{fig_num}_combined.png'

# read all csvs
print("Loading CSV files...")
k_df        = pd.read_csv(fn_k)
k_exante_df = pd.read_csv(fn_k_exante)
s_exante_df = pd.read_csv(fn_s_exante)
reg_df      = pd.read_csv(fn_reg)
s_expost_df = pd.read_csv(fn_s_expost)
hybrid_df   = pd.read_csv(fn_hybrid)

# merge on 'project' so all data line up
print("Merging dataframes...")
df_merged = k_df.merge(k_exante_df, on = 'project', suffixes = ('', '_kex'))
df_merged = df_merged.merge(s_exante_df, on = 'project', suffixes = ('', '_sex'))
df_merged = df_merged.merge(reg_df, on = 'project', suffixes = ('', '_reg'))
df_merged = df_merged.merge(s_expost_df, on = 'project', suffixes = ('', '_sexp'))
df_merged = df_merged.merge(hybrid_df, on = 'project', suffixes = ('', '_hybrid'))

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

if include_hybrid:
    methods['hybrid_exante'] = '_hyb'

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
metrics_df.to_csv(out_csv, index = False)
print(f'wrote r², mae, and bias to {out_csv}')

years = metrics_df['year']

# Set graphic parameters
plt.rc('figure', figsize = (8, 4))
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
fig, axs = plt.subplots(n, 1, figsize = (8, 4 * n), sharex = True)

# Define colors
colors = {
    'regional_exante': '#006CD1',  # Blue
    'k_exante': '#40B0A6',         # Teal
    's_exante': '#CDAC60',         # Gold
    's_expost': '#C13C3C',         # Red
    'hybrid_exante': '#9467BD'     # Purple for hybrid
}

for i, metric in enumerate(metrics): # plot for each metric
    ax = axs[i]
    ax.plot(years, metrics_df[f'regional_exante_{metric}'], label = '$\it{Ex\ ante}$ regional', color = colors['regional_exante'], linewidth = 1)
    ax.plot(years, metrics_df[f'k_exante_{metric}'], label = '$\it{Ex\ ante}$ project', color = colors['k_exante'], linewidth = 1)
    ax.plot(years, metrics_df[f's_exante_{metric}'], label = '$\it{Ex\ ante}$ time-shifted', color = colors['s_exante'], linewidth = 1)
    ax.plot(years, metrics_df[f's_expost_{metric}'], label = '$\it{Ex\ post}$ matching', color = colors['s_expost'], linestyle = 'dashed', linewidth = 2)
    
    # Add hybrid plot if requested
    if include_hybrid:
        ax.plot(years, metrics_df[f'hybrid_exante_{metric}'], label = '$\it{Ex\ ante}$ hybrid', color = colors['hybrid_exante'], linewidth = 2)
    
    if metric == 'bias':
        ax.axhline(0, color='gray', linestyle='--') # dashed horizontal line at 0 for the bias plot
    ax.set_title(titles[i])
    ax.set_ylabel(ylabels[i])

# Save combined plot
axs[-1].set_xlabel('Year')
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc = 'center left', bbox_to_anchor = (1.01, 0.5), ncol = 1)
fig.tight_layout()
fig.savefig(out_png_combined, bbox_inches = 'tight')
print(f'Saved combined plot to {out_png_combined}')

# Save individual plots
for i, metric in enumerate(metrics):
    plt.figure(figsize = (8, 4))

    plt.plot(years, metrics_df[f'regional_exante_{metric}'], label = '$\it{Ex\ ante}$ regional', color = colors['regional_exante'], linewidth = 1)
    plt.plot(years, metrics_df[f'k_exante_{metric}'], label = '$\it{Ex\ ante}$ project', color = colors['k_exante'], linewidth = 1)
    plt.plot(years, metrics_df[f's_exante_{metric}'], label = '$\it{Ex\ ante}$ time-shifted', color = colors['s_exante'], linewidth = 1)
    plt.plot(years, metrics_df[f's_expost_{metric}'], label = '$\it{Ex\ post}$ matching', color = colors['s_expost'], linestyle = 'dashed', linewidth = 2)

    if include_hybrid:
        plt.plot(years, metrics_df[f'hybrid_exante_{metric}'], label = '$\it{Ex\ ante}$ hybrid', color = colors['hybrid_exante'], linewidth = 2)

    if metric == 'bias':
        plt.axhline(0, color='gray', linestyle='--')

    plt.title(titles[i])
    plt.ylabel(ylabels[i])
    plt.xlabel('Year')
    plt.legend(loc = 'center left', bbox_to_anchor = (1.01, 0.5), ncol = 1)
    plt.tight_layout()
    plt.savefig(out_paths[i])
    print(f"Saved individual subplot for {metric} to {out_paths[i]}")