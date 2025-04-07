#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# paths
CSV_DIR = Path("/maps/jh2589/exante/csvs")
PLOT_DIR = Path("/maps/jh2589/exante/plots")
PLOT_DIR.mkdir(exist_ok=True)

INPUT_CSV = CSV_DIR / "project_rates.csv"

df = pd.read_csv(INPUT_CSV).dropna()

# titles
axis_labels = {
    "s_exante_rate": "Time-Shifted",
    "k_exante_rate": "Project",
    "regional_exante_rate": "Regional",
    "s_expost_rate": ""
}

def make_facet_scatter(counterfactual_cols, filename, observed_col="k_rate", colours=None):
    """
    create scatter subplots for each approach
    """
    n = len(counterfactual_cols)
    fig, axs = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

    all_x = df[counterfactual_cols].values.flatten()
    all_y = df[observed_col].values.flatten()
    all_vals = np.concatenate([all_x, all_y])
    min_val = np.floor(all_vals.min() / 5) * 5
    max_val = np.ceil(all_vals.max() / 5) * 5

    for i, x_col in enumerate(counterfactual_cols):
        ax = axs[i]
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
        ax.set_xlabel("Counterfactual Deforestation Rate (%)", fontsize=14)
        if i == 0:
            ax.set_ylabel("Observed Deforestation Rate (%)", fontsize=14)

        ax.text(
            mins + 0.02*(maxs - mins),
            maxs - 0.1*(maxs - mins),
            f"MAE: {mae_val:.3f}\nBias: {bias_val:.3f}\nGoodness-of-Fit: {prediction:.2f}",
            fontsize=12,
            va="top"
        )

    plt.subplots_adjust(wspace=0.02)
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"saved {filename}")

def make_single_scatter(x_col, filename, y_col="k_rate", colour="#1f77b4"):
    """
    create a single scatter plot for an approach
    """
    x = df[x_col]
    y = df[y_col]

    mae_val = np.mean(np.abs(y - x))
    bias_val = np.mean(x - y)
    prediction = 1 - np.sum((y - x)**2) / np.sum((y - y.mean())**2)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, color=colour)

    mins = min(x.min(), y.min() + 0.5)
    maxs = max(x.max(), y.max() + 0.5)
    line = np.linspace(mins, maxs, 100)

    # identity line
    plt.plot(line, line, linestyle="--", color="gray")

    plt.xlim(mins, maxs)
    plt.ylim(mins, maxs)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.xlabel("Counterfactual Deforestation Rate (%)", fontsize=14)
    plt.ylabel("Observed Deforestation Rate (%)", fontsize=14)
    plt.title(axis_labels.get(x_col, x_col), fontsize=16)

    plt.text(
        mins + 0.02*(maxs - mins),
        maxs - 0.1*(maxs - mins),
        f"MAE: {mae_val:.3f}\nBias: {bias_val:.3f}\nGoodness-of-Fit: {prediction:.2f}",
        fontsize=12,
        va="top"
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"saved {filename}")

# create a multi-facet scatter
make_facet_scatter(
    ["regional_exante_rate", "k_exante_rate", "s_exante_rate"],
    "exante_counterfactuals_vs_observed_rate.png",
    colours=["#006CD1", "#40B0A6", "#CDAC60"]
)

# create a single scatter
make_single_scatter(
    "s_expost_rate",
    "expost_counterfactual_vs_observed_rate.png",
    colour="#C13C3C"
)