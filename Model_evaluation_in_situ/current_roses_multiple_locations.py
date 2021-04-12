# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:10:18 2021

@author: cristinarusso
"""
"""
This script is used to evaluate and compare model and in situ current direction and strength by plotting current roses. 
The script compares in situ observations to the model output at multiple locations.  
Scripts to perform the pre-processing are include in the tool-box. 

INPUT:
    - the model and in situ dataset name
    - the paths to the model and in situ netcdf files 
    - the netcdf files containing the model output
    - the netcdf files containing the in situ observations
    - the mooring names
    - the start and end date of the time series
    - the resample type (daily, hourly, etc)
    - the depth level at which to evaluate the data
    - the desired file extention type with which to save figures

OUTPUT:
    - A figure with multiple current roses, six for the in situ data and the other six for the model output 

REQUIRMENTS:
    -The netcdf files need to be formatted so that the data used for comparsion are at the sepcific mooring locations with the dimensions (depth,time)

The inputs to the script should be changed where nessaccary is the section below called USER INPUTS
"""

# %% USER INPUTS

#Name of the region (used for graph title and file saving)
model_name = 'HYCOM'
insitu_data_name = 'SAMBA'

#Path to where the netcdf data is stored 
path_to_model =  ['C:/sharedfolder/SAMBA/HYCOM/hycom_M3.nc',
                  'C:/sharedfolder/SAMBA/HYCOM/hycom_M4.nc',
                  'C:/sharedfolder/SAMBA/HYCOM/hycom_M7.nc',
                  'C:/sharedfolder/SAMBA/HYCOM/hycom_M8.nc',
                  'C:/sharedfolder/SAMBA/HYCOM/hycom_M9.nc',
                  'C:/sharedfolder/SAMBA/HYCOM/hycom_M10.nc']
path_to_insitu = ['C:/sharedfolder/SAMBA/SAMBAM3.nc',
                  'C:/sharedfolder/SAMBA/SAMBAM4.nc',
                  'C:/sharedfolder/SAMBA/SAMBAM7.nc',
                  'C:/sharedfolder/SAMBA/SAMBAM8.nc',
                  'C:/sharedfolder/SAMBA/SAMBAM9.nc',
                  'C:/sharedfolder/SAMBA/SAMBAM10.nc']

#Mooring Name
mooring_name = ['M3',
                'M4',
                'M7',
                'M8',
                'M9',
                'M10']

#Dates on which to conduct evaluation
start_date = '2014-09-20T00:00'
end_date = '2014-12-09T00:00'

#Reseampling of data to daily/hourly
resample_type = '1D'

#Path to where the figures should be stored 
path_out = 'C:/sharedfolder/SAMBA/HYCOM/figure/'

#Depth at which to evaluate
depth_int = 60

#Creating string of depth level to be used in labelling 
depth_level = str(depth_int)+' m' #NO INPUT NEEDED

#Figure extention
ext = '.png'

#Name of figure (png file) which shows the position of the area of interest 
savename_fig = 'current_roses_' +'_at_'+str(depth_int)+'m' #NO INPUT NEEDED

#Figure Title 
titlename_fig = 'Current Roses at '+depth_level #NO INPUT NEEDED

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
samba_grid = xr.open_dataset(path_to_insitu[0])
samba_daily = samba_grid.resample(time=resample_type).mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date, end_date))

##Finding nearest depth to intended evaluation depth
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[0])
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

##Making sure that the data is nanned in the same locations
model_direction_m3[nan_matrix_samba==True] = np.nan 
model_mag_m3[nan_matrix_samba==True] = np.nan
samba_direction_m3[nan_matrix_model==True] = np.nan 
samba_mag_m3[nan_matrix_model==True] = np.nan

model_direction_m3[nan_matrix_model==True] = np.nan
model_mag_m3[nan_matrix_model==True] = np.nan
samba_direction_m3[nan_matrix_samba==True] = np.nan 
samba_mag_m3[nan_matrix_samba==True] = np.nan

##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[1])
samba_daily = samba_grid.resample(time='1D').mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date, end_date))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[1])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date, end_date))

##Reading in variables
exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m4 = np.copy(samba_dir)
samba_mag_m4 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values*100')
nan_matrix_model = np.isnan(model_dir)
model_direction_m4 = np.copy(model_dir)
model_mag_m4 = np.copy(model_mag)

model_direction_m4[nan_matrix_samba==True] = np.nan 
model_mag_m4[nan_matrix_samba==True] = np.nan
samba_direction_m4[nan_matrix_model==True] = np.nan 
samba_mag_m4[nan_matrix_model==True] = np.nan

model_direction_m4[nan_matrix_model==True] = np.nan
model_mag_m4[nan_matrix_model==True] = np.nan
samba_direction_m4[nan_matrix_samba==True] = np.nan 
samba_mag_m4[nan_matrix_samba==True] = np.nan

##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[2])
samba_daily = samba_grid.resample(time=resample_type).mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date, end_date))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[2])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date, end_date))

## getting values
exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m7 = np.copy(samba_dir)
samba_mag_m7 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values*100')
nan_matrix_model = np.isnan(model_dir)
model_direction_m7 = np.copy(model_dir)
model_mag_m7 = np.copy(model_mag)

model_direction_m7[nan_matrix_samba==True] = np.nan 
model_mag_m7[nan_matrix_samba==True] = np.nan
samba_direction_m7[nan_matrix_model==True] = np.nan 
samba_mag_m7[nan_matrix_model==True] = np.nan

model_direction_m7[nan_matrix_model==True] = np.nan
model_mag_m7[nan_matrix_model==True] = np.nan
samba_direction_m7[nan_matrix_samba==True] = np.nan 
samba_mag_m7[nan_matrix_samba==True] = np.nan

##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[3])
samba_daily = samba_grid.resample(time=resample_type).mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date, end_date))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[3])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date, end_date))

## getting values

exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m8 = np.copy(samba_dir)
samba_mag_m8 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values*100')
nan_matrix_model = np.isnan(model_dir)
model_direction_m8 = np.copy(model_dir)
model_mag_m8 = np.copy(model_mag)

model_direction_m8[nan_matrix_samba==True] = np.nan 
model_mag_m8[nan_matrix_samba==True] = np.nan
samba_direction_m8[nan_matrix_model==True] = np.nan 
samba_mag_m8[nan_matrix_model==True] = np.nan

model_direction_m8[nan_matrix_model==True] = np.nan
model_mag_m8[nan_matrix_model==True] = np.nan
samba_direction_m8[nan_matrix_samba==True] = np.nan 
samba_mag_m8[nan_matrix_samba==True] = np.nan

##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[4])
samba_daily = samba_grid.resample(time=resample_type).mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date, end_date))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[4])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date, end_date))

## getting values

exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m9 = np.copy(samba_dir)
samba_mag_m9 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values*100')
nan_matrix_model = np.isnan(model_dir)
model_direction_m9 = np.copy(model_dir)
model_mag_m9 = np.copy(model_mag)

model_direction_m9[nan_matrix_samba==True] = np.nan 
model_mag_m9[nan_matrix_samba==True] = np.nan
samba_direction_m9[nan_matrix_model==True] = np.nan 
samba_mag_m9[nan_matrix_model==True] = np.nan

model_direction_m9[nan_matrix_model==True] = np.nan
model_mag_m9[nan_matrix_model==True] = np.nan
samba_direction_m9[nan_matrix_samba==True] = np.nan 
samba_mag_m9[nan_matrix_samba==True] = np.nan

##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[5])
samba_daily = samba_grid.resample(time=resample_type).mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date, end_date))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[5])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date, end_date))

## getting values
exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m10 = np.copy(samba_dir)
samba_mag_m10 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values*100')
nan_matrix_model = np.isnan(model_dir)
model_direction_m10 = np.copy(model_dir)
model_mag_m10 = np.copy(model_mag)

model_direction_m10[nan_matrix_samba==True] = np.nan 
model_mag_m10[nan_matrix_samba==True] = np.nan
samba_direction_m10[nan_matrix_model==True] = np.nan 
samba_mag_m10[nan_matrix_model==True] = np.nan

model_direction_m10[nan_matrix_model==True] = np.nan
model_mag_m10[nan_matrix_model==True] = np.nan
samba_direction_m10[nan_matrix_samba==True] = np.nan 
samba_mag_m10[nan_matrix_samba==True] = np.nan


#%% Plotting Current Rose

# Setting Figure and Axes
fig = plt.figure(figsize=[17,7])

ax1 = fig.add_subplot(2,6,1, projection="windrose")
ax2 = fig.add_subplot(2,6,2, projection="windrose") 
ax3 = fig.add_subplot(2,6,3, projection="windrose")
ax4 = fig.add_subplot(2,6,4, projection="windrose")
ax5 = fig.add_subplot(2,6,5, projection="windrose")  
ax6 = fig.add_subplot(2,6,6, projection="windrose")
ax7 = fig.add_subplot(2,6,7, projection="windrose")
ax8 = fig.add_subplot(2,6,8, projection="windrose")
ax9 = fig.add_subplot(2,6,9, projection="windrose")
ax10 = fig.add_subplot(2,6,10, projection="windrose")  
ax11 = fig.add_subplot(2,6,11, projection="windrose")
ax12 = fig.add_subplot(2,6,12, projection="windrose")  

## plotting SAMBA MOORING ROSES
ax1.bar(samba_direction_m10, samba_mag_m10, bins=np.arange(0, 90, 10),normed = True, cmap=cm.jet)
ax2.bar(samba_direction_m9, samba_mag_m9, bins=np.arange(0, 90, 10),normed = True, cmap=cm.jet)
ax3.bar(samba_direction_m8, samba_mag_m8, bins=np.arange(0, 90, 10),normed = True, cmap=cm.jet)
ax4.bar(samba_direction_m7, samba_mag_m7, bins=np.arange(0, 90, 10), normed = True,cmap=cm.jet)
ax5.bar(samba_direction_m4, samba_mag_m4, bins=np.arange(0, 90, 10), normed = True,cmap=cm.jet)
ax6.bar(samba_direction_m3, samba_mag_m3, bins=np.arange(0, 90, 10), normed = True,cmap=cm.jet)

## plotting MODEL ROSES
ax7.bar(model_direction_m10, model_mag_m10, bins=np.arange(0, 90, 10),normed = True, cmap=cm.jet)
ax8.bar(model_direction_m9, model_mag_m9, bins=np.arange(0, 90, 10),normed = True, cmap=cm.jet)
ax9.bar(model_direction_m8, model_mag_m8, bins=np.arange(0, 90, 10),normed = True, cmap=cm.jet)
ax10.bar(model_direction_m7, model_mag_m7, bins=np.arange(0, 90, 10), normed = True,cmap=cm.jet)
ax11.bar(model_direction_m4, model_mag_m4, bins=np.arange(0, 90, 10), normed = True,cmap=cm.jet)
ax12.bar(model_direction_m3, model_mag_m3, bins=np.arange(0, 90, 10), normed = True,cmap=cm.jet)

## Adding Velocity Legend
ax12.legend(bbox_to_anchor=(1, 0.72))
fig.text(0.99,0.58,'Velocity (cm.s$^{-1}$)',fontweight='bold',fontsize=10,rotation=90)

## Adjusting Space between Subplots
plt.subplots_adjust(hspace=0.00003,wspace=0.4)

## Adding Mooring Labels
fig.text(0.16,0.5,'M10',fontweight='bold',fontsize=14)
fig.text(0.30,0.5,'M9',fontweight='bold',fontsize=14)
fig.text(0.435,0.5,'M8',fontweight='bold',fontsize=14)
fig.text(0.57,0.5,'M7',fontweight='bold',fontsize=14)
fig.text(0.71,0.5,'M4',fontweight='bold',fontsize=14)
fig.text(0.845,0.5,'M3',fontweight='bold',fontsize=14)

## Adding Model and SAMBA labels
fig.text(0.925,0.75,insitu_data_name,fontweight='bold',fontsize=14,rotation=90)
fig.text(0.925,0.35,model_name,fontweight='bold',fontsize=14,rotation=90)


## Setting Figure Title
fig.suptitle(titlename_fig,y=1, fontweight='bold',fontsize=16)

## Saving Figure
plt.savefig(path_out+savename_fig+ext)

