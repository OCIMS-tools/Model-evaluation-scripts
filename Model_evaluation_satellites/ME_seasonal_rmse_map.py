#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:27:29 2020

@author: matty
"""
"""
This script is used to calculate the root square mean error (RMSE) between model output and observations for a given region.
The RMSE is performed for each grid point per season. The seasons are defined as Summer (December, January, February), 
Autumn (March, April, May), Winter (June, July, August) and Spring (September, October, November). The script calculates 
the RMSE on the variable provided as well as the daily anomalies of the variable.
  
The inputs to the script should be changed where nessaccary is the section below called USER INPUTS

INPUT:
    - two netcdf file containing the model output and observations
    - the path and name of the two netcdf files 
    - the path and name of the figures to be saved
    - the name of the variables to be compared (! as written in netcdf file !)
    - the name of the lon and lat dimesion of the model data (! as written in netcdf file !)
    - the colormap plus the min and max values for the colorbar of the figure

OUTPUT:
    - Figure 1: Map of RMSE for the chosen variable the subplots (2x2) show the correlations
      for winter, autumn, spring and summer respectively
    
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

#The colormap of the figures
cmap = 'viridis' 

#The max and min values of the colorbar plus the number of ticks on the colorbar (tick number will change depending on max and min values)
cmin=0
cmax=3
cint=0.2

#The lat and lon of the model netcdf file  
model_lon_name = "lon_rho"
model_lat_name = "lat_rho"

#Path to where the figures should be stored 
savepath="E:\\Model_evaluations\\"
savename = 'RMSE_seasonal_map'
ext = '.png'

# %% Import packages 

import numpy as np
import xarray as xr 
import pylab as plt
import cartopy.crs as ccrs
import cartopy
from sklearn.metrics import mean_squared_error
from math import sqrt

# %% Fnction used in the script 

#Funtion which calculates the RMSE of a 3d matrix [time,lat,lon]. The two matrices must be the same shape.
# The function loops through each grid cell to extract an array from each matrix, the RMSE between the two arrays,
# is then calculated and the resulant RMSE values is stored in a new matrix
# The RMSE is calculated using the mean_squared_error function from sklearn (https://scikit-learn.org/stable/)

def rmse_matrix (observations, model):
    #Getting the number of rows and columns to loop through each grid cell
    row_length = np.shape(observations[0,:,:])[0];rows=np.arange(start=0, stop=row_length, step=1);
    column_length = np.shape(observations[0,:,:])[1];columns=np.arange(start=0, stop=column_length, step=1);

    #Creating empty variable to fill within the loop
    rmse_matrix = np.zeros(np.shape(observations[0,:,:]))
    #Loop through each grid cell and find the correlation between the two arrays 
    for x in rows:
            for y in columns:
                roms_tmp_array=model[:,x,y];satelite_tmp_array=observations[:,x,y]
                if np.isnan(roms_tmp_array).all():
                    rmse_matrix[x,y] =  np.nan
                elif np.isnan(satelite_tmp_array).all():
                    rmse_matrix[x,y] =  np.nan   
                else:
                    tmp_rmse = sqrt(mean_squared_error(satelite_tmp_array, roms_tmp_array))
                    rmse_matrix[x,y] =  tmp_rmse
                
    return rmse_matrix 


# %%Loading the matrix from the dataset


model_dataset = xr.open_dataset(path_to_model+model_dataset_name)
satelite_dataset = xr.open_dataset(path_to_satelite+satelite_dataset_name)

#Loading sst values from roms and satelite dataset
sst_model = model_dataset[model_variable].values
sst_satelite = satelite_dataset[satelite_variable].values

#loading lon and lat 
longitude = model_dataset[model_lon_name].values[0,:]
latitude = model_dataset[model_lat_name].values[:,0]

#Loading time
time = satelite_dataset.time.values

# %% Seperating using seasonality of the months 

# Use .groupby('time.month') to organize the data into months
# then use .groups to extract the indices for each month
seasonal_model_idxs=model_dataset.groupby('time.season').groups

# Extract the time indices corresponding to all the Januarys 
summer_idxs=seasonal_model_idxs['DJF']
winter_idxs=seasonal_model_idxs['JJA']
autumn_idxs=seasonal_model_idxs['MAM']
spring_idxs=seasonal_model_idxs['SON']

# Extract the january months by selecting 
# the relevant indices
summer_model = model_dataset.isel(time=summer_idxs)
winter_model = model_dataset.isel(time=winter_idxs)
autumn_model = model_dataset.isel(time=autumn_idxs)
spring_model = model_dataset.isel(time=spring_idxs)

# Grouping satelite sst by seasons 'DJF' 'JJA' 'MAM' 'SON'
seasonal_sat_idxs=satelite_dataset.groupby('time.season').groups

# Extract the time indices corresponding to all the Januarys 
summer_idxs=seasonal_sat_idxs['DJF']
winter_idxs=seasonal_sat_idxs['JJA']
autumn_idxs=seasonal_sat_idxs['MAM']
spring_idxs=seasonal_sat_idxs['SON']

# Extract the january months by selecting 
# the relevant indices
summer_sat = satelite_dataset.isel(time=summer_idxs)
winter_sat = satelite_dataset.isel(time=winter_idxs)
autumn_sat = satelite_dataset.isel(time=autumn_idxs)
spring_sat = satelite_dataset.isel(time=spring_idxs)

# %% Loading the SST values as matrices for RMSE calculations 

#Loading model data 
summer_model = summer_model[model_variable].values
winter_model = winter_model[model_variable].values
autumn_model = autumn_model[model_variable].values
spring_model = spring_model[model_variable].values

#Loading satelite data
summer_sat = summer_sat[satelite_variable].values
winter_sat = winter_sat[satelite_variable].values
autumn_sat = autumn_sat[satelite_variable].values
spring_sat = spring_sat[satelite_variable].values

# %% Calculating the RMSE using the function defined in the beginning of the script

#Caculates the RMSE of two matrix equal size with observes first then model data [time,lat,lon] 
summer_rmse = rmse_matrix(summer_sat,summer_model)
spring_rmse = rmse_matrix(spring_sat,spring_model)
autumn_rmse = rmse_matrix(autumn_sat,autumn_model)
winter_rmse = rmse_matrix(winter_sat,winter_model)
 
# %%

# Label for the colormap
cb_label='Temperature [$^{o}$C]'

fig, ax = plt.subplots(nrows=2, ncols=2,
      subplot_kw=dict(projection=ccrs.PlateCarree()))

fig.suptitle('Seasonal RMSE')

cs = ax[0,0].contourf(longitude,latitude,summer_rmse,60,vmin=cmin,vmax=cmax,cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[0,0].title.set_text('Summer') 
ax[0,0].coastlines()
ax[0,0].add_feature(cartopy.feature.LAND, zorder=0)
ax[0,0].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[0,0].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=False
gl.xlines = True
gl.ylines = True

cs = ax[0,1].contourf(longitude,latitude,autumn_rmse,60,vmin=cmin,vmax=cmax,cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[0,1].title.set_text('Autumn') 
ax[0,1].coastlines()
ax[0,1].add_feature(cartopy.feature.LAND, zorder=0)
ax[0,1].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[0,1].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=False
gl.xlines = True
gl.ylines = True

cs = ax[1,0].contourf(longitude,latitude,winter_rmse,60,vmin=cmin,vmax=cmax,cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[1,0].title.set_text('Winter') 
ax[1,0].coastlines()
ax[1,0].add_feature(cartopy.feature.LAND, zorder=0)
ax[1,0].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[1,0].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=True
gl.xlines = True
gl.ylines = True

cs = ax[1,1].contourf(longitude,latitude,spring_rmse,60,vmin=cmin,vmax=cmax,cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[1,1].title.set_text('Spring') 
ax[1,1].coastlines()
ax[1,1].add_feature(cartopy.feature.LAND, zorder=0)
ax[1,1].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[1,1].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=True
gl.xlines = True
gl.ylines = True

pos1 = ax[1,0].get_position()
pos2 = ax[1,1].get_position()

#### Add a colourbar at the bottom
cax = fig.add_axes([pos1.y0,pos2.x0-pos2.y1-0.04, pos2.x0+0.23, 0.02])
cticks=np.arange(cmin,cmax+(cint*2),cint*2)
cb = plt.colorbar(cs, orientation = 'horizontal', cax=cax, ticks=cticks, extendrect='True', drawedges=True)
cb.ax.tick_params(axis='x',direction='in', length=7, width=1, colors='k', top='on')
cb.set_label(cb_label)
cb.outline.set_edgecolor('black')
cb.outline.set_linewidth(1)
cb.dividers.set_edgecolor('black')
cb.dividers.set_linewidth(0)

fig.subplots_adjust(hspace=0.3, wspace=0.2)


plt.savefig(savepath+savename+ext,dpi=700)
