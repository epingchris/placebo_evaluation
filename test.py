#!/usr/bin/env python3

import numpy as np
import pandas as pd
import time
from scipy.stats import lognorm
import matplotlib.pyplot as plt

# Load data of existing REDD+ projects
REDD_existing = pd.read_csv('IDRECCO_V5_20231201_project.csv')
REDD_existing = REDD_existing.rename(columns = REDD_existing.iloc[0])
REDD_existing = REDD_existing.drop(labels = [0, 1])

# Filter out projects that:
# (1) has ended as of September 2022
# (2) project type includes "REDD" but not "jurisdictional"
# (3) dominant activity type is "REDD"
# (4) forest type is "humid"
# (5) is not located in a protected area 
REDD_existing = REDD_existing[
    (REDD_existing['Status_2022'].str.lower() == 'ongoing') &
    (REDD_existing['project_type'].str.contains('REDD', case = False, na = False)) &
    (~REDD_existing['project_type'].str.contains('jurisdictional', case = False, na = False)) &
    (REDD_existing['dominant_type'] == 'REDD') &
    (REDD_existing['forest_type'].str.lower() == 'humid') &
    (REDD_existing['protected_area'].str.lower() == 'no')
]
REDD_existing['area'] = pd.to_numeric(REDD_existing['area'], errors = 'coerce')

plt.hist(REDD_existing['area'], bins = 50, edgecolor = 'black')
plt.savefig("test.png")

# # Obtain the top and bottom 5% area quantiles
# lower_quantile = REDD_existing['area'].quantile(0.05)
# upper_quantile = REDD_existing['area'].quantile(0.95)
# REDD_filtered = REDD_existing[(REDD_existing['area'] > lower_quantile) & (REDD_existing['area'] < upper_quantile)].copy()

# # Compute the radius in meters
# REDD_filtered['radius_m'] = np.sqrt(REDD_filtered['area'] * 10000 / np.pi)

# # Fit a log-normal distribution to the radius data
# # scipy's lognorm uses shape (sigma), loc, and scale (exp(mu))
# shape, loc, scale = lognorm.fit(REDD_filtered['radius_m'], floc = 0)

# # Generate synthetic data (lognormal with estimated parameters)
# log_radius_samples = np.random.normal(loc = np.log(scale), scale = shape, size = 100)
# REDD_fitted = pd.DataFrame({'log_radius': log_radius_samples})
# REDD_fitted['radius_m'] = np.exp(REDD_fitted['log_radius'])
# REDD_fitted['area'] = REDD_fitted['radius_m'] ** 2 * np.pi / 10000

# # Compute area quantiles
# lb_area = REDD_fitted['area'].quantile(0.05)
# ub_area = REDD_fitted['area'].quantile(0.095)

# # Input params (area parameters that are needed to create buffers in mask)
# #lb_area = 29962.86  # 5% quantile from existing project sample
# #ub_area = 793911.22  # 95% quantile from existing project sample

# lb_radius = (lb_area * 10000 / np.pi) ** 0.5
# ub_radius = (ub_area * 10000 / np.pi) ** 0.5