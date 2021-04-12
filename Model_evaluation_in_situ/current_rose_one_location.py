# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:10:18 2021

@author: cristinarusso
"""
"""
This script is used to evaluate and compare model and in situ current direction and strength by plotting current roses. 
The script compares in situ observations to the model output at a certain location.  
Scripts to perform the pre-processing are include in the tool-box. 

INPUT:
    - the model and in situ dataset name
    - the path to the model and in situ netcdf files 
    - the netcdf file containing the model output
    - the netcdf file containing the in situ observations
    - the mooring name
    - the start and end date of the time series
    - the resample type (daily, hourly, etc)
    - the depth level at which to evaluate the data
    - the desired file extention type with which to save figures

OUTPUT:
    - A figure with two current roses, one for the in situ data and the other for the model output 

REQUIRMENTS:
    -The netcdf files need to be formatted so that the data used for comparsion are at the sepcific mooring locations with the dimensions (depth,time)

The inputs to the script should be changed where nessaccary is the section below called USER INPUTS
"""

# %% USER INPUTS

#Name of the region (used for graph title and file saving)
model_name = 'HYCOM'
insitu_data_name = 'SAMBA'

#Path to where the netcdf data is stored 
path_to_model =  'C:/sharedfolder/SAMBA/HYCOM/hycom_M3.nc' 
path_to_insitu = 'C:/sharedfolder/SAMBA/SAMBAM3.nc' 

#Mooring Name
mooring_name = 'M3'

#Dates on which to conduct evaluation
start_date = '2014-07-10T00:00'
end_date = '2014-12-09T00:00'

#Reseampling of data to daily/hourly
resample_type = '1D'

#Path to where the figures should be stored 
path_out = 'C:/sharedfolder/SAMBA/HYCOM/figure/'

#Depth at which to evaluate
depth_int = 240

#Creating string of depth level to be used in labelling 
depth_level = str(depth_int)+' m' #NO INPUT NEEDED

#Figure extention
ext = '.png'

#Name of figure (png file) which shows the position of the area of interest 
savename_fig = 'current_roses_' + mooring_name+'_at_'+str(depth_int)+'m' #NO INPUT NEEDED

#Figure Title 
titlename_fig = 'Mooring ' + mooring_name+' at '+depth_level #NO INPUT NEEDED

# %% PREPPARING OUTPUT

## Importing Modules
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import windrose

##Defining find nearest function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return array[idx],idx
 
##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu)
samba_daily = samba_grid.resample(time=resample_type).mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date, end_date))

##Finding nearest depth to intended evaluation depth
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model)
uvdata = uvdata.sortby('time')

##Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date, end_date))

##Reading in variables
exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m3 = np.copy(samba_dir)
samba_mag_m3 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values*100')
nan_matrix_model = np.isnan(model_dir)
model_direction_m3 = np.copy(model_dir)
model_mag_m3 = np.copy(model_mag)

#Making sure that the data is nanned in the same locations
model_direction_m3[nan_matrix_samba==True] = np.nan 
model_mag_m3[nan_matrix_samba==True] = np.nan
samba_direction_m3[nan_matrix_model==True] = np.nan 
samba_mag_m3[nan_matrix_model==True] = np.nan

model_direction_m3[nan_matrix_model==True] = np.nan
model_mag_m3[nan_matrix_model==True] = np.nan
samba_direction_m3[nan_matrix_samba==True] = np.nan 
samba_mag_m3[nan_matrix_samba==True] = np.nan

#%% Plotting Current Rose

## Setting Figure and Axes
fig = plt.figure(figsize=[6,7])
ax1 = fig.add_subplot(2,1,1, projection="win drose")
ax2 = fig.add_subplot(2,1,2, projection="windrose")

## Plotting SAMBA MOORING ROSES
ax1.bar(samba_direction_m3, samba_mag_m3, bins=np.arange(0, 90, 10), normed = True,cmap=cm.jet)

## plotting MODEL ROSES
ax2.bar(model_direction_m3, model_mag_m3, bins=np.arange(0, 90, 10), normed = True,cmap=cm.jet)

## Adding Velocity Legend
ax2.legend(bbox_to_anchor=(1, 0.72))
fig.text(0.94,0.58,'Velocity (cm.s$^{-1}$)',fontweight='bold',fontsize=10,rotation=90)

## Adjusting Space between Subplots
plt.subplots_adjust(hspace=0.5,wspace=0.4)

## Adding Mooring Labels
fig.text(0.50,0.5,mooring_name,fontweight='bold',fontsize=14)

## Adding Model and SAMBA labels
fig.text(0.75,0.77,insitu_data_name,fontweight='bold',fontsize=14,rotation=90)
fig.text(0.75,0.3,model_name,fontweight='bold',fontsize=14,rotation=90)

## Setting Figure Title
fig.suptitle(titlename_fig,y=1, fontweight='bold',fontsize=16)

## Saving Figure
plt.savefig(path_out+savename_fig+ext)

