#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:50:15 2020

@author: matty
"""
"""
This script is used to calculate the correlation between model output and observations for a given region.
The correlation is performed for each grid point for the full time period provided. The script calculates 
the correlation on the variable provided as well as the daily anomalies of the variable.
  
The inputs to the script should be changed where nessaccary is the section below called USER INPUTS

INPUT:
    - two netcdf file containing the model output and observations
    - the path and name of the two netcdf files 
    - the path and name of the figures to be saved
    - the name of the variables to be compared (! as written in netcdf file !)
    - the name of the lon and lat dimesion of the model data (! as written in netcdf file !)
    - the colormap plus the min and max values for the colorbar of the figure

OUTPUT:
    - Figure 1: Map of correlations for the daily anomalies of the chosen variable (! areas which aren't significant are masked !)
    - Figure 2: Map of correlations for the chosen variable (! areas which aren't significant are masked !)
    
REQUIRMENTS:
    -The netcdf files need to be formatted so that the variabled used for comparsion is a 3 dimesional matrix of (time, latitude, longitude)
    -The netcdf files MUST be regridded to identical grids 
    -The netcdf files MUST be sampled at a daily frequency and the time dimesion should have the same length (there can not be missing days in the data) 

Scripts to perform the pre-processing are include in the tool-box. 

"""

# %% USER INPUTS

#Path to where the netcdf data is stored 
path_to_model = '/media/matthew/Seagate_Expansion_Drive/Model_output/' 
path_to_satelite = '/media/matthew/Seagate_Expansion_Drive/Model_output/' 
#The name of the netcdf file
model_dataset_name = "model_dataset.nc"
satelite_dataset_name = "satelite_dataset.nc"

#The variables to be analysed as written in netcdf files 
model_variable = "temp"
satelite_variable= "analysed_sst"

#The lat and lon of the model netcdf file  
model_lon_name = "lon_rho"
model_lat_name = "lat_rho"

#The colormap of the figures
cmap = 'viridis' 

#The max and min values of the colorbar plus the number of ticks on the colorbar (tick number will change depending on max and min values)
cmin=0
cmax=1
cint=0.2

#Path to where the figures should be stored 
savepath="E:\\Model_evaluations\\"
#Name of figure (png file) for the correlation between model and satelite daily anomalies 
savename_fig1 = 'Correlation_daily_ano_map'
#Name of figure (png file) for the correlation between model and satelite 
savename_fig2 = 'Correlation_map'
ext = '.png'

# %% Importing neccassary packages 

import numpy as np
import xarray as xr 
import pylab as plt
import scipy.stats as stat
import cartopy.crs as ccrs
import cartopy

# %% Function need in the script

#Funtion which calculates the correlation between a 3d matrix [time,lat,lon]. The two matrices must be the same shape.
# The function loops through each grid cell to extract an array from each matrix, the correlation between the two arrays,
# is then calculated and the resulant correlation and sigfincance in stored in a new matrix
# The correlation is calculated using the stat.pearsonr function from scipy (https://www.scipy.org/)
def corr_matrix (model,satelite):
    #Getting the number of rows and columns to loop through each grid cell
    row_length = np.shape(model[0,:,:])[0];rows=np.arange(start=0, stop=row_length, step=1);
    column_length = np.shape(model[0,:,:])[1];columns=np.arange(start=0, stop=column_length, step=1);

    #Creating empty variable to fill within the loop
    corr_matrix = np.zeros(np.shape(model[0,:,:]))
    sig_matrix = np.zeros(np.shape(model[0,:,:]))
    #Loop through each grid cell and find the correlation between the two arrays 
    for x in rows:
            for y in columns:
                model_tmp_array=model[:,x,y];satelite_tmp_array=satelite[:,x,y]
                if np.isnan(model_tmp_array).all():
                    corr_matrix[x,y] =  np.nan
                    sig_matrix[x,y] =  np.nan
                elif np.isnan(satelite_tmp_array).all():
                    corr_matrix[x,y] =  np.nan   
                    sig_matrix[x,y] =  np.nan                      
                else:
                    tmp_corr_array = stat.pearsonr(model_tmp_array,satelite_tmp_array)
                    corr_matrix[x,y] =  tmp_corr_array[0]
                    sig_matrix[x,y] =  tmp_corr_array[1]
                
    return corr_matrix, sig_matrix

# %% Loading netcdf files using xarray 

model_dataset = xr.open_dataset(path_to_model+model_dataset_name)
satelite_dataset = xr.open_dataset(path_to_satelite+satelite_dataset_name)

#loading lon and lat 
longitude = model_dataset[model_lon_name].values[0,:]
latitude = model_dataset[model_lat_name].values[:,0]

#Loading time
time = satelite_dataset.time.values
            
# %% Calculating the daily anomalies 

#Calculating the daily anomalies in the model and satelite data using xarray groupby methods 
model_daily_ano = model_dataset.groupby("time.dayofyear") - model_dataset.groupby("time.dayofyear").mean("time")
sat_daily_ano = satelite_dataset.groupby("time.dayofyear") - satelite_dataset.groupby("time.dayofyear").mean("time")

#sat_daily_clim = sat_regrid_ds.groupby("time.dayofyear").mean("time")

#Loading sst values from roms and satelite dataset
sst_model = model_dataset[model_variable].values
sst_satelite = satelite_dataset[satelite_variable].values

#Loading daily anomalies values from roms and satelite dataset
sst_ano_model = model_daily_ano[model_variable].values
sst_ano_satelite = sat_daily_ano[satelite_variable].values

# %% Find the correlations between model and satelite SST at each grid cell

#Using the def 
[correlation_ano_matrix, sigfincance_ano_matrix] = corr_matrix(sst_ano_model,sst_ano_satelite)

#Mask the correlation values that are not sigfincant 
correlation_ano_mask = np.copy(correlation_ano_matrix)
correlation_ano_mask[np.where(sigfincance_ano_matrix > 0.05)] = np.nan 

#Using the def 
[correlation_matrix, sigfincance_matrix] = corr_matrix(sst_model,sst_satelite)

#Mask the correlation values that are not sigfincant 
correlation_mask = np.copy(correlation_matrix)
correlation_mask[np.where(sigfincance_matrix > 0.05)] = np.nan 

# %% Plotting map showing the correlations of the daily anomalies  

fig = plt.figure(figsize=(10, 5),facecolor='white')
ax = plt.axes(projection=ccrs.PlateCarree())
cs = plt.contourf(longitude,latitude,correlation_ano_mask,60,vmin=cmin,vmax=cmax,cmap = cmap,transform=ccrs.PlateCarree());
cbar = plt.colorbar(cs)
cbar.set_label('Correlation (R)')
plt.title('Correlation of daily anomalies between\nREMSS and ROMS SST') 
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=0)
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

gl = ax.gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.right_labels=False
gl.xlines = True
gl.ylines = True
#gl.xlabel_style = {'size': 15, 'color': 'gray'}
#gl.xlabel_style = {'color': 'black', 'weight': 'bold'}

plt.savefig(savepath+savename_fig1+ext,dpi=700)

# %% Plotting map showing the correlations 

fig = plt.figure(figsize=(10, 5),facecolor='white')
ax = plt.axes(projection=ccrs.PlateCarree())
cs = plt.contourf(longitude,latitude,correlation_mask,60,vmin=cmin,vmax=cmax,cmap = cmap, transform=ccrs.PlateCarree());
cbar = plt.colorbar(cs)
cbar.set_label('Correlation (R)')
plt.title('Correlation between\nREMSS and ROMS SST') 
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=0)
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)

gl = ax.gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.right_labels=False
gl.xlines = True
gl.ylines = True
#gl.xlabel_style = {'size': 15, 'color': 'gray'}
#gl.xlabel_style = {'color': 'black', 'weight': 'bold'}

plt.savefig(savepath+savename_fig2+ext,dpi=700)
