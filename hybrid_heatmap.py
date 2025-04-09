#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import seaborn as sns

# set up paths for csvs and plots
DIR = Path('/maps/epr26/placebo_evaluation_out')
csv_path = DIR / 'hybrid_results.csv'

# read csv containing complete results for hybrid ex ante methods
df = pd.read_csv(csv_path)
df = df[(df['proj_baseline'] != 2011) & (df['reg_baseline'] != 2011)]

df_heatmap = df.pivot(index = 'reg_baseline', columns = 'proj_baseline', values = 'avg_r2')

# Find the position of the max value
max_idx = df_heatmap.stack().idxmax() #stack() essentially converts the dataframe into long format, and idxmax() returns the index (row, column tuple) of the maximum value
row, col = max_idx

# Get the row and column indices for annotation
row_idx = list(df_heatmap.index).index(row)
col_idx = list(df_heatmap.columns).index(col)

# Set graphic parameters
plt.rc('figure', figsize = (8, 4))
plt.rc('axes', titlesize = 16, labelsize = 14)
plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
plt.rc('legend', fontsize = 12)
plt.rc('savefig', dpi = 150)

plt.figure(figsize = (10, 8))
ax = sns.heatmap(df_heatmap, annot = True, fmt = '.2f', cmap = 'coolwarm_r', cbar_kws = {'label': 'Goodness-of-fit (R² over the identity line)'}) #the "_r" suffix reverses the coolwarm palette

# Add a thick rectangle around the max cell
ax.add_patch(Rectangle(
    (col_idx, row_idx),     # (x, y)
    1, 1,                   # width, height
    fill = False, 
    edgecolor='black', 
    lw = 3                  # Line width
))

plt.title('Average goodness-of-fit of forecasts')
plt.xlabel('Start year of project deforestation rate interval')
plt.ylabel('Start year of regional deforestation rate interval')
plt.tight_layout()
plt.show()
png_path = DIR / 'out_figure8_hybrid_method_heatmap.png'
plt.savefig(png_path, bbox_inches = 'tight')
print(f'Saved heatmap to {png_path}')