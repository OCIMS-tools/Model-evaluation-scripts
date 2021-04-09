#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:00:02 2020

@author: matty
"""
"""
This script is used to calculate the standard deviation (SD) between model output and observations for a given region.
The SD is performed for each grid point per season. The seasons are defined as Summer (December, January, February), 
Autumn (March, April, May), Winter (June, July, August) and Spring (September, October, November). The script calculates 
the SD on the variable provided for both the model and observations.
  
The inputs to the script should be changed where nessaccary is the section below called USER INPUTS

INPUT:
    - two netcdf file containing the model output and observations
    - the path and name of the two netcdf files 
    - the path and name of the figures to be saved
    - the name of the variables to be compared (! as written in netcdf file !)
    - the name of the lon and lat dimesion of the model data (! as written in netcdf file !)
    - the colormap plus the min and max values for the colorbar of the figure


OUTPUT:
    - Figure 1: Map of the SD (3x4) for the chosen variable of both the model and observations as well as the bias
      bias the SD (model - observations) for each of the seasons.
    
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
cmax=2
cint=0.2

#The colormap of the BIAS figures
bmap = 'RdBu' 

#The max and min values of the colorbar (BIAS) plus the number of ticks on the colorbar (tick number will change depending on max and min values)
bmin=-1
bmax=1
bint=0.2

#Path to where the figures should be stored 
savepath="E:\\Model_evaluations\\"
savename = 'Std_dev_seasonal_map'
ext = '.png'

# %% Import packages 

import numpy as np
import xarray as xr 
import pylab as plt
import scipy.stats as stat
import cartopy.crs as ccrs
import cartopy

# %% Function used in the script

#Funtion which calculates the standard deviation (SD) between a 3d matrix [time,lat,lon]. The two matrices must be the same shape.
# The function loops through each grid cell to extract an array the matrix, the SD of the arrays,
# is then calculated and in stored in a new matrix. The correlation is calculated 
# using the stat.tstd function from scipy (https://www.scipy.org/)
def std_matrix (data_matrix):
    #Getting the number of rows and columns to loop through each grid cell
    row_length = np.shape(data_matrix[0,:,:])[0];rows=np.arange(start=0, stop=row_length, step=1);
    column_length = np.shape(data_matrix[0,:,:])[1];columns=np.arange(start=0, stop=column_length, step=1);

    #Creating empty variable to fill within the loop
    out_matrix = np.zeros(np.shape(data_matrix[0,:,:]))
    #Loop through each grid cell and find the standard deviation of the array at each grid
    for x in rows:
            for y in columns:
                tmp_array=data_matrix[:,x,y];
                if np.isnan(tmp_array).all():
                    out_matrix[x,y] =  np.nan                     
                else:
                    tmp_std = stat.tstd(tmp_array)
                    out_matrix[x,y] =  tmp_std
                
    return out_matrix

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

# %% Loading the SST values as matrices for correlations

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

# %% Find the standard deviation of a matrix at each grid cell [time,lat,lon]

#Finding the standard deviation of the model output 
summer_model_std = std_matrix(summer_model)
winter_model_std = std_matrix(winter_model)
autumn_model_std = std_matrix(autumn_model)
spring_model_std = std_matrix(spring_model)

#Finding the standard deviation of the satelite data 
summer_satelite_std = std_matrix(summer_sat)
winter_satelite_std = std_matrix(winter_sat)
autumn_satelite_std = std_matrix(autumn_sat)
spring_satelite_std = std_matrix(spring_sat)

#Finding the standard deviation of the bias between the model and satelite data 
summer_bias_std = summer_model_std - summer_satelite_std
winter_bias_std = winter_model_std - winter_satelite_std
autumn_bias_std = autumn_model_std - autumn_satelite_std
spring_bias_std = spring_model_std - spring_satelite_std

# %% #Plotting std dev for roms data 

cb_label='Std dev [SD]'

fig, ax = plt.subplots(nrows=4, ncols=3,
      figsize = (8, 10) ,subplot_kw=dict(projection=ccrs.PlateCarree()))

fig.suptitle('Seasonal standard deviations')

cs1 = ax[0,0].contourf(longitude,latitude,summer_satelite_std,60,vmin=cmin, vmax=cmax, cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[0,0].title.set_text('Summer REMSS') 
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

cs2 = ax[0,1].contourf(longitude,latitude,summer_model_std,60,vmin=cmin, vmax=cmax, cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[0,1].title.set_text('Summer ROMS') 
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

cs3 = ax[0,2].contourf(longitude,latitude,summer_bias_std,60,vmin=bmin, vmax=bmax, cmap = bmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[0,2].title.set_text('Summer ROMS-REMSS') 
ax[0,2].coastlines()
ax[0,2].add_feature(cartopy.feature.LAND, zorder=0)
ax[0,2].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[0,2].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=False
gl.xlines = True
gl.ylines = True


cs4 = ax[1,0].contourf(longitude,latitude,autumn_satelite_std,60,vmin=cmin, vmax=cmax, cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[1,0].title.set_text('Autumn REMSS') 
ax[1,0].coastlines()
ax[1,0].add_feature(cartopy.feature.LAND, zorder=0)
ax[1,0].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[1,0].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.bottom_labels=False
gl.right_labels=False
gl.xlines = True
gl.ylines = True

cs5 = ax[1,1].contourf(longitude,latitude,autumn_model_std,60,vmin=cmin, vmax=cmax,cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[1,1].title.set_text('Autumn ROMS') 
ax[1,1].coastlines()
ax[1,1].add_feature(cartopy.feature.LAND, zorder=0)
ax[1,1].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[1,1].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.bottom_labels=False
gl.right_labels=False
gl.xlines = True
gl.ylines = True

cs6 = ax[1,2].contourf(longitude,latitude,autumn_bias_std,60,vmin=bmin, vmax=bmax, cmap = bmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[1,2].title.set_text('Autumn ROMS-REMSS') 
ax[1,2].coastlines()
ax[1,2].add_feature(cartopy.feature.LAND, zorder=0)
ax[1,2].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[1,2].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.bottom_labels=False
gl.right_labels=False
gl.xlines = True
gl.ylines = True



cs7 = ax[2,0].contourf(longitude,latitude,winter_satelite_std,60,vmin=cmin, vmax=cmax, cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[2,0].title.set_text('Winter REMSS') 
ax[2,0].coastlines()
ax[2,0].add_feature(cartopy.feature.LAND, zorder=0)
ax[2,0].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[2,0].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=False
gl.xlines = True
gl.ylines = True

cs8 = ax[2,1].contourf(longitude,latitude,winter_model_std,60,vmin=cmin, vmax=cmax, cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[2,1].title.set_text('Winter ROMS') 
ax[2,1].coastlines()
ax[2,1].add_feature(cartopy.feature.LAND, zorder=0)
ax[2,1].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[2,1].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=False
gl.xlines = True
gl.ylines = True

cs9 = ax[2,2].contourf(longitude,latitude,winter_bias_std,60,vmin=bmin, vmax=bmax, cmap = bmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[2,2].title.set_text('Winter bias') 
ax[2,2].coastlines()
ax[2,2].add_feature(cartopy.feature.LAND, zorder=0)
ax[2,2].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[2,2].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=False
gl.xlines = True
gl.ylines = True


cs10 = ax[3,0].contourf(longitude,latitude,spring_satelite_std,60,vmin=cmin, vmax=cmax, cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[3,0].title.set_text('Spring REMSS') 
ax[3,0].coastlines()
ax[3,0].add_feature(cartopy.feature.LAND, zorder=0)
ax[3,0].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[3,0].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = True
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=True
gl.xlines = True
gl.ylines = True

cs = ax[3,1].contourf(longitude,latitude,spring_model_std,60,vmin=cmin, vmax=cmax, cmap = cmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[3,1].title.set_text('Spring ROMS') 
ax[3,1].coastlines()
ax[3,1].add_feature(cartopy.feature.LAND, zorder=0)
ax[3,1].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[3,1].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=True
gl.xlines = True
gl.ylines = True

cs = ax[3,2].contourf(longitude,latitude,spring_bias_std,60,vmin=bmin, vmax=bmax, cmap = bmap, transform=ccrs.PlateCarree());
#cbar = plt.colorbar(cs)
#cbar.set_label('Correlation (R)')
ax[3,2].title.set_text('Spring ROMS-REMSS') 
ax[3,2].coastlines()
ax[3,2].add_feature(cartopy.feature.LAND, zorder=0)
ax[3,2].set_aspect('auto')
#ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl = ax[3,2].gridlines(crs=ccrs.PlateCarree(),linewidth=0, color='black', draw_labels=True)
gl.left_labels = False
gl.top_labels = False
gl.right_labels=False
gl.bottom_labels=True
gl.xlines = True
gl.ylines = True

fig.subplots_adjust(hspace=0.2, wspace=0.1)


cticks=np.arange(cmin,cmax+(cint*1),cint*1)
cb = fig.colorbar(cs1, ax=list(ax[3,0:2]),orientation='horizontal',ticks=cticks ,pad=0.1)
cb.ax.tick_params(axis='x',direction='in', length=7, width=1, colors='k', top='on')
cb.ax.locator_params(nbins=7)
cb.set_label(cb_label)
cb.outline.set_edgecolor('black')
cb.outline.set_linewidth(1)

bticks=np.arange(bmin,bmax+(bint*1),bint*1)
cb1 = fig.colorbar(cs3, ax=ax[3,2],orientation='horizontal',ticks=bticks, aspect=10, pad=0.1)
cb1.ax.tick_params(axis='x',direction='in', length=7, width=1, colors='k', top='on')
cb1.ax.locator_params(nbins=3)
cb1.set_label(cb_label)
cb1.outline.set_edgecolor('black')
cb1.outline.set_linewidth(1)

#Saving figure
plt.savefig(savepath+savename+ext,dpi=700)

