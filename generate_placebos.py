#!/usr/bin/env python3

import os
import ee
import random
import numpy as np
import pandas as pd
import time
from scipy.stats import lognorm

%matplotlib inline
import matplotlib.pyplot as plt

#Connect to Google Earth Engine API; run ee.Authenticate() for first-time users
#ee.Authenticate()
ee.Initialize(project='ex-ante-forecast')
user_path = 'users/epingchris'
date = time.strftime("%b%d").lower()

# Select a region
region = 'afrotropics' # neotropics, afrotropics, asian_tropics

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
REDD_existing['area'] = pd.to_numeric(REDD_existing['area'], errors = 'coerce') # Convert area to numeric
REDD_existing['radius_m'] = np.sqrt(REDD_existing['area'] * 10000 / np.pi) # Compute the radius in meters

# Filter the radius to the 5% and 95% quantiles
lb_radius = REDD_existing['radius_m'].quantile(0.05)
ub_radius = REDD_existing['radius_m'].quantile(0.05)
REDD_filtered = REDD_existing[(REDD_existing['radius_m'] > lb_radius) & (REDD_existing['radius_m'] < ub_radius)].copy()

# Fit a log-normal distribution to the radius data
# scipy's lognorm uses shape (sigma), loc, and scale (exp(mu))
shape, loc, scale = lognorm.fit(REDD_filtered['radius_m'], floc = 0)

# Generate synthetic data (lognormal with estimated parameters)
log_radius_samples = np.random.normal(loc = np.log(scale), scale = shape, size = 200)
REDD_fitted = pd.DataFrame({'log_radius': log_radius_samples})
REDD_fitted['radius_m'] = np.exp(REDD_fitted['log_radius'])
REDD_fitted['area'] = REDD_fitted['radius_m'] ** 2 * np.pi / 10000

plt.hist(REDD_existing['area'], bins = 50, color = 'darkgray', edgecolor = 'black')
plt.hist(REDD_fitted['area'], bins = 50, color = 'red', edgecolor = 'grey', alpha = 0.5)
plt.savefig("hist_area.png")
plt.clf()  # Clear the figure

plt.hist(REDD_existing['radius_m'], bins = 50, color = 'darkgray', edgecolor = 'black')
plt.hist(REDD_fitted['radius_m'], bins = 50, color = 'green', edgecolor = 'grey', alpha = 0.5)
plt.savefig("hist_radius.png")
plt.clf()  # Clear the figure

# Compute radius and area quantiles
lb_radius = REDD_fitted['radius_m'].quantile(0.05)
ub_radius = REDD_fitted['radius_m'].quantile(0.95)

lb_area = REDD_fitted['area'].quantile(0.05)
ub_area = REDD_fitted['area'].quantile(0.95)

# Customised input params (area quantiles that are needed to create buffers in mask)
#lb_area = 29962.86  # 5% quantile from existing project sample
#ub_area = 793911.22  # 95% quantile from existing project sample

match region:
    case 'neotropics':
        coords = [
            [-112.870, -41.410],
            [-33.033, -41.410],
            [-33.033, 33.665],
            [-112.870, 33.665],
            [-112.870, -41.410] 
        ]
    case 'afrotropics':
        coords = [
            [-17.881, -31.772],
            [57.958, -31.772],
            [57.958, 18.743],
            [-17.881, 18.743],
            [-17.881, -31.772]
        ]
    case 'asian_tropics':
        coords = [
            [69.118, -37.058],
            [178.893, -37.058],
            [178.893, 42.841],
            [69.118, 42.841],
            [69.118, -37.058] 
        ]
    case _:
        print('need to pick a region')

# Load masks if already uploaded - CHANGE AS APPROPRIATE 

#forest_mask = ee.Image(user_path + "/asian_tropics_forest_mask_sep12").unmask(0).rename('mask')
#forest_mask = ee.Image(user_path + "/afrotropics_forest_mask_sep11").unmask(0).rename('mask')
#forest_mask = ee.Image(user_path + "/neotropics_forest_mask_sep10").unmask(0).rename('mask')
#combined_mask = ee.Image(user_path + "/asian_tropics_combined_mask_jun24").unmask(0)

# Generate masks if not already uploaded

# Create an EE geometry polygon from the coordinates
bbox = ee.Geometry.Polygon(coords)

# 1. Mask to JRC (undisturbed + degraded + reforested)
jrc = ee.ImageCollection('projects/JRC/TMF/v1_2022/AnnualChanges').mosaic()

forest_mask = jrc.select('Dec2022').eq(1) \
    .Or(jrc.select('Dec2022').eq(2)) \
    .Or(jrc.select('Dec2022').eq(4)) \
    .unmask(0) \
    .clip(bbox)

# Export mask 
task = ee.batch.Export.image.toAsset(image=forest_mask, 
                              description=region+' forest mask', 
                              assetId=user_path+'/'+region+'_forest_mask_'+date,
                              scale=30,
                              region = bbox,
                              maxPixels=1e12)
task.start()
print('Forest mask export started')

# 2. Mask to access

access = ee.Image("projects/ex-ante-forecast/assets/202001_Global_Walking_Only_Travel_Time_To_Healthcare_2019").select('b1').clip(bbox)

# Masking
lb_access = 48.50452 # 5% quantile from existing project sample
ub_access = 1418.10620 # 95% quantile from existing project sample
access_mask = access.gte(lb_access).And(access.lte(ub_access))

# 3. Mask to existing REDD+ projects (BUFFERED)

polygons = ee.FeatureCollection("projects/ex-ante-forecast/assets/REDD_polygon_shapefiles_V2").filterBounds(bbox)
def add_buffer(feature):
    buffered_geometry = feature.geometry().buffer(ub_radius) # buffer width = radius of maximum circle in m
    return ee.Feature(buffered_geometry, feature.toDictionary())
buffered_polygons = polygons.map(add_buffer)
shapefile_img = ee.Image.constant(1)
shapefile_mask_buffer = shapefile_img.clip(buffered_polygons).mask().Not().clip(bbox)

# 4. Mask to land (BUFFERED)

countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filterBounds(bbox)
land_buffered = countries.geometry().buffer(-ub_radius) # removed .dissolve() here to keep country border buffers
land_img = ee.Image.constant(1)
land_mask_buffer = land_img.clip(land_buffered).clip(bbox) # SELECT AS APPROPRIATE

# 5. Create combined mask

combined_mask = forest_mask.And(access_mask).And(shapefile_mask_buffer).And(land_mask_buffer).rename('mask')

# Export mask
task = ee.batch.Export.image.toAsset(image=access_mask, 
                              description=region+' access mask', 
                              assetId=user_path+'/'+region+'_access_mask_'+date,
                              scale=30,
                              region = bbox,
                              maxPixels=1e12)
task.start()

task = ee.batch.Export.image.toAsset(image=shapefile_mask_buffer, 
                              description=region+' shapefile mask buffer', 
                              assetId=user_path+'/'+region+'_shapefile_mask_buffer_'+date,
                              scale=30,
                              region = bbox,
                              maxPixels=1e12)
task.start()

task = ee.batch.Export.image.toAsset(image=land_mask_buffer, 
                              description=region+' land mask buffer', 
                              assetId=user_path+'/'+region+'_land_mask_buffer_'+date,
                              scale=30,
                              region = bbox,
                              maxPixels=1e12)
task.start()

task = ee.batch.Export.image.toAsset(image=combined_mask, 
                              description=region+' combined mask', 
                              assetId=user_path+'/'+region+'_combined_mask_'+date,
                              scale=30,
                              region = bbox,
                              maxPixels=1e12)
task.start()
print('High resolution combined mask -- export started')

# 6. Generate random circles and calculate forest cover
num_points = 100

points = combined_mask.stratifiedSample(
    numPoints=num_points,
    classBand='mask',
    scale=30, # CHANGE AS APPROPRIATE
    geometries=True
).filter(ee.Filter.eq('mask', 1))

random_radius = np.random.uniform(low = lb_radius, high = ub_radius, size = num_points)
random_radius_list = ee.List(list(random_radius))

# Function to add a buffer around each point using each random_radius value as radius
def buffer_with_radius(feature):
    # Get the index strong of the current feature and parse it to an integer
    feature_index_str = ee.Feature(feature).get('system:index')
    feature_index = ee.Number.parse(feature_index_str)
    
    # Get the corresponding radius value from the random_radius list
    radius_value = ee.Number(random_radius_list.get(feature_index))
    maxError = radius_value.multiply(0.05)
    buffered_feature = feature.buffer(radius_value, maxError) # 5% error
    
    # Add the radius value as a property called 'radius'
    return buffered_feature

circles = points.map(buffer_with_radius)

# Sample points from within the circle
num_points_forest = 1000 # CHANGE AS APPROPRIATE
def sample_points_in_circle(feature):
    # Get the geometry of the feature
    geometry = feature.geometry()
    
    samples = ee.FeatureCollection.randomPoints(
        region = geometry,
        points = num_points_forest)
    sampled_values = forest_mask.sampleRegions(
        collection=samples,
        properties=[],
        scale=30 
    )
    
    ones = sampled_values.filter(ee.Filter.eq('mask', 1)).size()
    #print('Calculated ones:', ones.getInfo())
    total = sampled_values.size()
    #print('Calculated total', total.getInfo())
    proportion_of_ones = ones.divide(total)
    
    return feature.set('forest_cover', proportion_of_ones)

circles_with_forest_cover = circles.map(sample_points_in_circle)


task = ee.batch.Export.table.toAsset(collection=ee.FeatureCollection(circles_with_forest_cover),
                                     description=region+' circles',
                                     assetId=user_path+'/'+region+'_controls')
task.start()
print('-- Export started --')