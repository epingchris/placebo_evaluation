#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# set up paths for csvs and plots
DIR = Path('/maps/epr26/placebo_evaluation_out')
csv_path = DIR / 'hybrid_results.csv'

# read csv containing complete results for hybrid ex ante methods
df_hybrid = pd.read_csv(csv_path)

df_heatmap = df_hybrid.pivot(index='reg_baseline', columns='proj_baseline', values='avg_r2')

plt.figure(figsize = (10, 8))
sns.heatmap(df_heatmap, annot = True, fmt = ".2f", cmap = "coolwarm", cbar_kws = {'label': 'avg_r2'})
plt.title('Average R² of hybrid forecasts')
plt.xlabel('Project deforestation rate calculated for interval starting from')
plt.ylabel('Regional deforestation rate starting for interval starting from')
plt.tight_layout()
plt.show()
png_path = DIR / 'hybrid_method_heatmap.png'
plt.savefig(png_path, bbox_inches = 'tight')
print(f'Saved heatmap to {png_path}')