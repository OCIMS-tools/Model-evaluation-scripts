#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:46:11 2020

@author: matty
"""
"""
This script is used to examine specific regions of interest by comparing area-average regions between model output and observations. 
Scripts to perform the pre-processing are include in the tool-box named: 

INPUT:
    - two netcdf file containing the model output and observations
    - the title and co-oridantes of the region of interest
    - the path and name of the two netcdf files 
    - the name of the variables to be compared (! as written in netcdf file !)
    - the name of the lon and lat dimesion of the model data (! as written in netcdf file !)

OUTPUT:
    - A map of the region of interest 
    - A timeseries model vs observations (including correlation and significance)
    - A timeseries of the daily climatology (including correlation and significance)
    - A timeseries of the daily anomalies (including correlation and significance)
    - A scatter plot of the daily anomalies with a colormap of indicating time of year in seasons (including correlation and significance)

REQUIRMENTS:
    -The netcdf files need to be formatted so that the variabled used for comparsion are a 3 dimesional matrix of (time, longitude, latitude)
    -The netcdf files MUST be regridded to identical grids 
    -The netcdf files MUST be sampled at a daily frequency and the time dimesion should have the same length (there can not be missing days in the data) 

The inputs to the script should be changed where nessaccary is the section below called USER INPUTS
"""

# %% USER INPUTS

#Name of the region (used for graph title and file saving)
position_name='Cape Caludron'

#Cooridates of the region of interest 
west=15
east=16
south=-35
north=-34

#Path to where the netcdf data is stored 
path_to_model = "E:\\Model_evaluations\\"
path_to_satelite = "E:\\Model_evaluations\\"

#The name of the netcdf file
model_dataset_name = "model_dataset.nc"
satelite_dataset_name = "satelite_dataset.nc"

#The variables to be analysed as written in netcdf files 
model_variable = "temp"
satelite_variable= "analysed_sst"

#The lat and lon of the model netcdf file  
model_lon_name = "lon_rho"
model_lat_name = "lat_rho"

#Path to where the figures should be stored 
savepath="E:\\Model_evaluations\\"
#Name of figure (png file) which shows the position of the area of interest 
savename_fig1 = 'Position_area_of_interest_'+str(position_name)
#Name of figure (png file) which shows the full timeseries  
savename_fig2 = 'Full_timerseries_'+str(position_name)
#Name of figure (png file) which shows the timeseries of the daily climatology for the region of interest
savename_fig3 = 'Daily_climatology_'+str(position_name)
#Name of figure (png file) which shows the timeseries of the daily anomaly for the region of interest
savename_fig4 = 'Daily_anomaly_'+str(position_name)
#Name of figure (png file) which shows the scatter graph of the daily anomaly for the region of interest
savename_fig5 = 'Scatter_Daily_anomaly_'+str(position_name)
ext = '.png'


# %% Importing packages 

import numpy as np
import xarray as xr 
import pylab as plt
import scipy.stats as stat
import cartopy.crs as ccrs
import cartopy

# %% The function used in the script below 

#This function calculates the find a values in a array closest to the specificed element
#The script uses the function to find closest lat/lon points in the dataset so users do not need the exact lat/lon coordinates
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#This function calculates the mean for each timestep of a 3d dimesional matrix, producing a array the same length of the matrix.
#The spript uses this function to average selected subset into arrays for statistical comparisions 
def mean_matrix_3d(matrix):
    matrix_array = np.zeros(len(matrix))
    for x in np.arange(len(matrix[:,1,1])):
        array_tmp = np.nanmean(matrix[x,:,:])
        matrix_array[x] = array_tmp
        
    return matrix_array

# %% Loading netcdf files using xarray 

model_dataset = xr.open_dataset(path_to_model+model_dataset_name)
satelite_dataset = xr.open_dataset(path_to_satelite+satelite_dataset_name)

#loading lon and lat 
longitude = model_dataset[model_lon_name].values[0,:]
latitude = model_dataset[model_lat_name].values[:,0]

#Loading time
time = satelite_dataset.time.values

# Time grouped by seasonal climatolgies for the time dimesion of scatter plot 
time_cat = np.zeros(len(time))

seasonal_model_idxs=model_dataset.groupby('time.season').groups

time_cat[seasonal_model_idxs['DJF']] = 0
time_cat[seasonal_model_idxs['MAM']] = 1
time_cat[seasonal_model_idxs['JJA']] = 2
time_cat[seasonal_model_idxs['SON']] = 3

#Finding the exact cordinates within the model and satelite grids for subsetting 
west_co = find_nearest(longitude,west)
east_co = find_nearest(longitude,east)
south_co = find_nearest(latitude,south)
north_co = find_nearest(latitude,north)

#Subsetting the datasets 
model_subset = model_dataset.where((model_dataset.lat_rho > south_co) &
                (model_dataset.lat_rho < north_co) & (model_dataset.lon_rho > west_co) &
                 (model_dataset.lon_rho < east_co), drop=True)

satelite_subset = satelite_dataset.where((satelite_dataset.lat > south_co) &
                (satelite_dataset.lat < north_co) & (satelite_dataset.lon > west_co) &
                 (satelite_dataset.lon < east_co), drop=True)


# %% Calculating the daily anomalies and daily climatologies using xr grouby functions

#Calculating the daily anomalies in the model and satelite data using xarray groupby methods 
model_daily_ano = model_subset.groupby("time.dayofyear") - model_subset.groupby("time.dayofyear").mean("time")
sat_daily_ano = satelite_subset.groupby("time.dayofyear") - satelite_subset.groupby("time.dayofyear").mean("time")

#Calculating the daily climatology of satelite and model datasets 
sat_daily_clim = satelite_subset.groupby("time.dayofyear").mean("time")
model_daily_clim = model_subset.groupby("time.dayofyear").mean("time")

# %% Selecting subset area

#Finding the correct indices of the above cooridates
west_index = np.where(longitude == find_nearest(longitude,west))[0][0]
east_index = np.where(longitude == find_nearest(longitude,east))[0][0]
south_index = np.where(latitude == find_nearest(latitude,south))[0][0]
north_index = np.where(latitude == find_nearest(latitude,north))[0][0]


# %% Loading the variables from the dataset

#Converting zeros to nans in the model data (land is denoted as zeros instead of nans which affects averaging)
model_subset[model_variable].values[np.isnan(satelite_subset[satelite_variable].values)] = np.nan
model_daily_ano[model_variable].values[np.isnan(sat_daily_ano[satelite_variable].values)] = np.nan
model_daily_clim[model_variable].values[np.isnan(sat_daily_clim[satelite_variable].values)] = np.nan

# %% #Finding the mean of the subset for each timestep using the function mean_matrix_array. (Converts a 3d matrix into area averaged array)

#Finding the mean for full timseries 
model_array = mean_matrix_3d(model_subset[model_variable].values)
satelite_array = mean_matrix_3d(satelite_subset[satelite_variable].values)

#Finding the mean of the daily anomalies 
model_ano_array = mean_matrix_3d(model_daily_ano[model_variable].values)
satelite_ano_array = mean_matrix_3d(sat_daily_ano[satelite_variable].values)

#Finding the mean of the daily climatology
model_clim_array = mean_matrix_3d(model_daily_clim[model_variable].values)
satelite_clim_array = mean_matrix_3d(sat_daily_clim[satelite_variable].values)


# %% Plotting postion of the selected area
    
fig = plt.figure(figsize=(10, 5),facecolor='white')
ax = plt.axes(projection=ccrs.PlateCarree())
cs = plt.contourf(longitude,latitude,model_dataset[model_variable].values[1,:,:],60, transform=ccrs.PlateCarree());
ln0 = plt.plot([longitude[west_index], longitude[east_index]], [latitude[south_index], latitude[south_index]], '-k')
ln1 = plt.plot([longitude[west_index], longitude[east_index]], [latitude[north_index], latitude[north_index]], '-k')
ln2 = plt.plot([longitude[east_index], longitude[east_index]], [latitude[north_index], latitude[south_index]], '-k')
ln3 = plt.plot([longitude[west_index], longitude[west_index]], [latitude[north_index], latitude[south_index]], '-k')
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=0)

plt.savefig(savepath+savename_fig1+ext)

# %% Ploting timeseries

#Finding correlation of the satelite and model 
corr_tmp = stat.pearsonr(model_array,satelite_array)
corr = corr_tmp[0]
sig = corr_tmp[1]

fig,ax = plt.subplots()
line_sat = plt.plot(time,satelite_array,label="REMSS");
line_roms = plt.plot(time,model_array,label="ROMS");
plt.xlabel('Date in years')
plt.ylabel('SST ($^\circ$C)')
plt.title('SST ROMS vs REMSS\n'+str(position_name))
plt.legend()

#Correlation label
textstr = 'corr - '+str(np.round(corr,2))
textstr2 = 'sig - '+str(np.round(sig,2))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.02, 1.2, textstr+'\n'+textstr2 , transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(savepath+savename_fig2+ext,bbox_inches='tight', pad_inches=0.2)


# %% Ploting timeseries daily clim

#Finding correlation of the satelite and model daily clim
corr_tmp = stat.pearsonr(model_clim_array,satelite_clim_array)
corr = corr_tmp[0]
sig = corr_tmp[1]

fig,ax = plt.subplots()
line_sat = plt.plot(satelite_clim_array,label="REMSS");
line_roms = plt.plot(model_clim_array,label="ROMS");
plt.xlabel('Day of year')
plt.ylabel('SST ($^\circ$C)')
plt.title('SST ROMS vs REMSS\n'+str(position_name))
plt.legend()

#Correlation label
textstr = 'corr - '+str(np.round(corr,2))
textstr2 = 'sig - '+str(np.round(sig,2))

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.02, 1.2, textstr+'\n'+textstr2 , transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(savepath+savename_fig3+ext,bbox_inches='tight', pad_inches=0.2)


# %% Ploting timeseries of daily anomaly

corr_tmp = stat.pearsonr(model_ano_array,satelite_ano_array)
corr = corr_tmp[0]
sig = corr_tmp[1]

fig,ax = plt.subplots()
line_sat = plt.plot(time,satelite_ano_array,label="REMSS");
line_roms = plt.plot(time,model_ano_array,label="ROMS");
plt.xlabel('Date in years')
plt.ylabel('SST ($^\circ$C)')
plt.title('SST ROMS vs REMSS\n'+str(position_name))
plt.legend()

#Correlation label
textstr = 'corr - '+str(np.round(corr,2))
textstr2 = 'sig - '+str(np.round(sig,2))

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.02, 1.2, textstr+'\n'+textstr2 , transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(savepath+savename_fig4+ext,bbox_inches='tight', pad_inches=0.2)

# %% Scatter plot

corr_tmp = stat.pearsonr(model_ano_array,satelite_ano_array)

corr = corr_tmp[0]
sig = corr_tmp[1]

fig,ax = plt.subplots()
cs = plt.scatter(model_ano_array,satelite_ano_array,c=time_cat)
loc = np.linspace(0, 4, 5)
cbar = plt.colorbar(cs, boundaries=np.linspace(0, 4, 5))
cbar.set_ticks(loc +.5)
cbar.set_ticklabels(['DJF','MAM','JJA','SON'])
#cbar.set_ticks([0,1,2,3,],update_ticks=True)

plt.axis('equal')
plt.ylabel('SST REMSS ($^\circ$C)')
plt.xlabel('SST ROMS ($^\circ$C)')
plt.title('SST ROMS vs REMSS\n'+str(position_name))

#Correlation label
textstr = 'corr - '+str(np.round(corr,2))
textstr2 = 'sig - '+str(np.round(sig,2))

# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='white', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.02, 1.2, textstr+'\n'+textstr2 , transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(savepath+savename_fig5+ext,bbox_inches='tight', pad_inches=0.2)


