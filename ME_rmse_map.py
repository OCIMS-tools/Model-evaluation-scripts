#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:13:48 2020

@author: matty
"""
"""
This script is used to calculate the root square mean error (RMSE) between model output and observations for a given region.
The RMSE is performed for each grid point for the full time period provide. The script calculates 
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
    - Figure 1: Map of RMSE for the chosen variable
    - Figure 2: Map of RMSE for the daily anomalies of the chosen variable
    
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

#Path to where the figures should be stored 
savepath="E:\\Model_evaluations\\"
savename_fig1 = 'RMSE_map.png'
savename_fig2 = 'RMSE_ano_map'
ext = '.png'

# %% Import packages 

import numpy as np
import xarray as xr 
import pylab as plt
import cartopy.crs as ccrs
import cartopy
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.cm as cm
  
# %% Functions used in the script    

#Funtion which calculates the RMSE of a 3d matrix [time,lat,lon]. The two matrices must be the same shape.
# The function loops through each grid cell to extract an array from each matrix, the RMSE between the two arrays,
# is then calculated and the resulant RMSE values is stored in a new matrix
# The RMSE is calculated using the mean_squared_error function from sklearn (https://scikit-learn.org/stable/)

def rmse_matrix (observations, model):
    #Getting the number of rows and columns to loop through each grid cell
    row_length = np.shape(model[0,:,:])[0];rows=np.arange(start=0, stop=row_length, step=1);
    column_length = np.shape(model[0,:,:])[1];columns=np.arange(start=0, stop=column_length, step=1);

    #Creating empty variable to fill within the loop
    rmse_matrix = np.zeros(np.shape(model[0,:,:]))
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

#Calculating the daily anomalies in the model and satelite data using xarray groupby methods 
model_daily_ano = model_dataset.groupby("time.dayofyear") - model_dataset.groupby("time.dayofyear").mean("time")
sat_daily_ano = satelite_dataset.groupby("time.dayofyear") - satelite_dataset.groupby("time.dayofyear").mean("time")

#Loading daily anomalies values from roms and satelite dataset
sst_ano_model = model_daily_ano[model_variable].values
sst_ano_satelite = sat_daily_ano[satelite_variable].values

#loading lon and lat 
longitude = model_dataset[model_lon_name].values[0,:]
latitude = model_dataset[model_lat_name].values[:,0]

#Loading time
time = satelite_dataset.time.values

# %% Calucalting RMSE 

#Caculates the RMSE of two matrix equal size with observes first then model data [time,lat,lon] 
rmse = rmse_matrix(sst_satelite,sst_model)

#Caculates the RMSE of the daily anomaly
rmse_ano = rmse_matrix(sst_ano_satelite,sst_ano_model)

# %% #Plotting ROMS for roms data 

fig = plt.figure(figsize=(10, 5),facecolor='white')
ax = plt.axes(projection=ccrs.PlateCarree())
cs = plt.contourf(longitude,latitude,rmse,60,vmin = cmin,vmax = cmax, cmap = cmap, transform=ccrs.PlateCarree());
m = plt.cm.ScalarMappable(cmap=cm.viridis)
m.set_array(rmse)
m.set_clim(cmin, cmax)
cbar = plt.colorbar(m, boundaries=np.linspace(cmin, cmax, 7))

cbar.set_label('RMSE')
plt.title('RMSE ROMS vs REMSS SST') 
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


plt.savefig(savepath+savename_fig1,dpi=700)

# %% #Plotting RMSE ano for roms data 

fig = plt.figure(figsize=(10, 5),facecolor='white')
ax = plt.axes(projection=ccrs.PlateCarree())
cs = plt.contourf(longitude,latitude,rmse_ano,60,vmin = cmin,vmax = cmax,cmap = cmap, transform=ccrs.PlateCarree());
m = plt.cm.ScalarMappable(cmap=cm.viridis)
m.set_array(rmse)
m.set_clim(cmin, cmax)
cbar = plt.colorbar(m, boundaries=np.linspace(cmin, cmax, 7))

cbar.set_label('RMSE')
plt.title('RMSE ROMS vs REMSS SST (Daily anomalies)') 
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


