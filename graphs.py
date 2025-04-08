#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# paths
CSV_DIR = Path("/maps/jh2589/exante/csvs")
PLOT_DIR = Path("/maps/epr26/placebo_evaluation_out")
PLOT_DIR.mkdir(exist_ok=True)

INPUT_CSV = CSV_DIR / "project_rates.csv"

df = pd.read_csv(INPUT_CSV).dropna()

# titles
axis_labels = {
    "s_exante_rate": "Time-shifted",
    "k_exante_rate": "Project",
    "regional_exante_rate": "Regional",
    "s_expost_rate": ""
}

def make_scatter(counterfactual_cols, filename, observed_col="k_rate", colours=None):
    """
    create scatter subplots for each approach
    """
    n = len(counterfactual_cols)
    fig, axs = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

    for i, x_col in enumerate(counterfactual_cols):
        ax = axs[i] if n > 1 else axs
        x = df[x_col]
        y = df[observed_col]

        mae_val = np.mean(np.abs(y - x))
        bias_val = np.mean(x - y)
        prediction = 1 - np.sum((y - x)**2) / np.sum((y - y.mean())**2)

        ax.scatter(x, y, color=colours[i])
        mins = min(x.min(), y.min() + 0.5)
        maxs = max(x.max(), y.max() + 0.5)
        line = np.linspace(mins, maxs, 100)

        # identity line
        ax.plot(line, line, linestyle="--", color="gray")

        ax.set_xlim(mins, maxs)
        ax.set_ylim(mins, maxs)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(axis_labels.get(x_col, x_col), fontsize=16)
        ax.set_xlabel("Counterfactual deforestation rate (%)", fontsize=14)
        if i == 0:
            ax.set_ylabel("Observed deforestation rate (%)", fontsize=14)

        ax.text(
            mins + 0.02*(maxs - mins),
            maxs - 0.1*(maxs - mins),
            f"MAE: {mae_val:.3f}\nBias: {bias_val:.3f}\nGoodness-of-fit: {prediction:.2f}",
            fontsize=12,
            va="top"
        )

    plt.subplots_adjust(wspace=0.02)
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"saved {filename}")

# create plots for ex ante methods
make_scatter(
    ["regional_exante_rate", "k_exante_rate", "s_exante_rate"],
    "out_figure4a_exante.png",
    colours=["#006CD1", "#40B0A6", "#CDAC60"]
)

# create the plot for ex post method
make_scatter(
    ["s_expost_rate"],
    "out_figure4b_expost.png",
    colour="#C13C3C"
)