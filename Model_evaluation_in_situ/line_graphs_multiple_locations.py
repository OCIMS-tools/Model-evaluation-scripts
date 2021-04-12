# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:10:18 2021

@author: cristinarusso
"""
"""
This script is used to evaluate and compare model and in situ current strength by plotting time series-long line graphs. 
The script compares in situ observations to the model output at multiple locations.  
Scripts to perform the pre-processing are included in the tool-box. 

INPUT:
    - the model and in situ dataset names
    - the paths to the model and in situ netcdf files 
    - the netcdf files containing the model output
    - the netcdf files containing the in situ observations
    - the mooring names
    - the start and end date of the time series for the insitu and model data
    - the resample type (daily, hourly, etc)
    - the depth level at which to evaluate the data
    - the desired file extention type with which to save figures
    - the desired title for the figure

OUTPUT:
    - A figure with multiple line graphs comparing the in situ data and model output 

REQUIRMENTS:
    -The netcdf files need to be formatted so that the data used for comparsion are at the sepcific mooring locations with the dimensions (depth,time)

The inputs to the script should be changed where nessaccary is the section below called USER INPUTS
"""
# %% USER INPUTS

#Name of the region (used for graph title and file saving)
model_name = 'BRAN'
insitu_data_name = 'SAMBA'

#Path to where the netcdf data is stored 
path_to_model =  ['C:/sharedfolder/SAMBA/BRAN/bran_M3.nc',
                  'C:/sharedfolder/SAMBA/BRAN/bran_M4.nc',
                  'C:/sharedfolder/SAMBA/BRAN/bran_M7.nc',
                  'C:/sharedfolder/SAMBA/BRAN/bran_M8.nc',
                  'C:/sharedfolder/SAMBA/BRAN/bran_M9.nc',
                  'C:/sharedfolder/SAMBA/BRAN/bran_M10.nc']
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
#If you want to evaluate moorings with different lengths of timeseries, 
#you would just need to add to the startdate and enddate lists and tweak the code
start_date = ['2014-09-20T00:00', #insitu date
              '2014-09-20T12:00']#model date NOTE: different in time
end_date = ['2014-12-09T00:00',#insitu date
            '2014-12-09T12:00']#model date NOTE: different in time

#Reseampling of data to daily/hourly
resample_type = '1D'

#Path to where the figures should be stored 
path_out = 'C:/sharedfolder/SAMBA/BRAN/figure/'

#Depth at which to evaluate
depth_int = 60

#Creating string of depth level to be used in labelling 
depth_level = str(depth_int)+' m' #NO INPUT NEEDED

#Figure extention
ext = '.png'

#Figure Title
figure_title='Time Series of Magnitude at '+depth_level+' Depth '

#Name of figure (png file) which shows the position of the area of interest 
savename_fig = path_out+model_name+depth_level #NO INPUT NEEDED

#%% PREPARING THE OUTPUT

## Importing Modules
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import scipy.stats as stats
import astropy.stats as ast
from astropy import units as u

def find_nearest(array, value):
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return array[idx],idx
##MOORING M3
##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[0])
samba_daily = samba_grid.resample(time='1D').mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date[0],end_date[0]))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[0])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date[1],end_date[1]))

## getting values
exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values/100')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m3 = np.copy(samba_dir)
samba_mag_m3 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values')
nan_matrix_model = np.isnan(model_dir)
model_direction_m3 = np.copy(model_dir)
model_mag_m3 = np.copy(model_mag)

model_direction_m3[nan_matrix_samba==True] = np.nan 
model_mag_m3[nan_matrix_samba==True] = np.nan
samba_direction_m3[nan_matrix_model==True] = np.nan 
samba_mag_m3[nan_matrix_model==True] = np.nan

model_direction_m3[nan_matrix_model==True] = np.nan
model_mag_m3[nan_matrix_model==True] = np.nan
samba_direction_m3[nan_matrix_samba==True] = np.nan 
samba_mag_m3[nan_matrix_samba==True] = np.nan

nas = np.logical_or(np.isnan(samba_mag_m3), np.isnan(model_mag_m3))

r_mag_m3,p_mag_m3 = stats.pearsonr(samba_mag_m3[~nas],model_mag_m3[~nas])
r_dir_m3 = ast.circcorrcoef(samba_direction_m3[~nas]*u.deg,model_direction_m3[~nas]*u.deg)

##MOORING M4

##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[1])
samba_daily = samba_grid.resample(time='1D').mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date[0],end_date[0]))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[1])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date[1],end_date[1]))

## getting values
exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values/100')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m4 = np.copy(samba_dir)
samba_mag_m4 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values')
nan_matrix_model = np.isnan(model_dir)
model_direction_m4 = np.copy(model_dir)
model_mag_m4 = np.copy(model_mag)

model_direction_m4[nan_matrix_model==True] = np.nan
model_mag_m4[nan_matrix_model==True] = np.nan
samba_direction_m4[nan_matrix_samba==True] = np.nan 
samba_mag_m4[nan_matrix_samba==True] = np.nan

model_direction_m4[nan_matrix_samba==True] = np.nan 
model_mag_m4[nan_matrix_samba==True] = np.nan
samba_direction_m4[nan_matrix_model==True] = np.nan 
samba_mag_m4[nan_matrix_model==True] = np.nan

nas = np.logical_or(np.isnan(samba_mag_m4), np.isnan(model_mag_m4))

r_mag_m4,p_mag_m4 = stats.pearsonr(samba_mag_m4[~nas],model_mag_m4[~nas])
r_dir_m4 = ast.circcorrcoef(samba_direction_m4[~nas]*u.deg,model_direction_m4[~nas]*u.deg)

##MOORING M7

##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[2])
samba_daily = samba_grid.resample(time='1D').mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date[0],end_date[0]))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[2])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date[1],end_date[1]))

## getting values

exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values/100')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m7 = np.copy(samba_dir)
samba_mag_m7 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values')
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

nas = np.logical_or(np.isnan(samba_mag_m7), np.isnan(model_mag_m7))

r_mag_m7,p_mag_m7 = stats.pearsonr(samba_mag_m7[~nas],model_mag_m7[~nas])
r_dir_m7 = ast.circcorrcoef(samba_direction_m7[~nas]*u.deg,model_direction_m7[~nas]*u.deg)

##MOORING M8
##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[3])
samba_daily = samba_grid.resample(time='1D').mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date[0],end_date[0]))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[3])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date[1],end_date[1]))

## getting values
exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values/100')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m8 = np.copy(samba_dir)
samba_mag_m8 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values')
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

nas = np.logical_or(np.isnan(samba_mag_m8), np.isnan(model_mag_m8))
r_mag_m8,p_mag_m8 = stats.pearsonr(samba_mag_m8[~nas],model_mag_m8[~nas])
r_dir_m8 = ast.circcorrcoef(samba_direction_m8[~nas]*u.deg,model_direction_m8[~nas]*u.deg)

##MOORING M9
##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[4])
samba_daily = samba_grid.resample(time='1D').mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date[0],end_date[0]))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[4])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date[1],end_date[1]))

## getting values

exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values/100')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m9 = np.copy(samba_dir)
samba_mag_m9 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values')
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

nas = np.logical_or(np.isnan(samba_mag_m9), np.isnan(model_mag_m9))
r_mag_m9,p_mag_m9 = stats.pearsonr(samba_mag_m9[~nas],model_mag_m9[~nas])
r_dir_m9= ast.circcorrcoef(samba_direction_m9[~nas]*u.deg,model_direction_m9[~nas]*u.deg)

## MOORING M10
##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu[5])
samba_daily = samba_grid.resample(time='1D').mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date[0],end_date[0]))
d,di = find_nearest(samba_mgrid.depth,depth_int)
depth_index = '[:,'+str(di)+']' 

##Importing Model Data
uvdata = xr.open_dataset(path_to_model[5])
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date[1],end_date[1]))

## getting values

exec('samba_dir = samba_mgrid.direction'+depth_index+'.values')
exec('samba_mag = samba_mgrid.magnitude'+depth_index+'.values/100')
nan_matrix_samba = np.isnan(samba_dir)
samba_direction_m10 = np.copy(samba_dir)
samba_mag_m10 = np.copy(samba_mag)

exec('model_dir = uvdata_samba_grid.direction'+depth_index+'.values')
exec('model_mag = uvdata_samba_grid.magnitude'+depth_index+'.values')
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

nas = np.logical_or(np.isnan(samba_mag_m10), np.isnan(model_mag_m10))
r_mag_m10,p_mag_m10 = stats.pearsonr(samba_mag_m10[~nas],model_mag_m10[~nas])
r_dir_m10 = ast.circcorrcoef(samba_direction_m10[~nas]*u.deg,model_direction_m10[~nas]*u.deg)

#%% PLOTTING OUTPUT

fig, axes = plt.subplots(3, 2,figsize=[12,8])

ax1 = axes[0,0]
ax2 = axes[0,1] 
ax3 = axes[1,0]  
ax4 = axes[1,1]
ax5 = axes[2,0]  
ax6 = axes[2,1] 

ax1.set_title(mooring_name[0],fontsize=14,fontweight='bold')
ax1.plot(samba_mag_m3,'r',label=insitu_data_name);ax1.plot(model_mag_m3,'b',label=model_name);
ax1.text(0,0,'R${^2}$='+str(np.round(r_mag_m3,3))+' p='+str(np.round(p_mag_m3,3)))
ax1.legend(loc='upper right')

ax2.set_title(mooring_name[1],fontsize=14,fontweight='bold')
ax2.plot(samba_mag_m4,'r',label=insitu_data_name);ax2.plot(model_mag_m4,'b',label=model_name);
ax2.text(0,0,'R${^2}$='+str(np.round(r_mag_m4,3))+' p='+str(np.round(p_mag_m4,3)))

ax3.set_title(mooring_name[2],fontsize=14,fontweight='bold')
ax3.plot(samba_mag_m7,'r',label=insitu_data_name);ax3.plot(model_mag_m7,'b',label=model_name);
ax3.text(0,0,'R${^2}$='+str(np.round(r_mag_m7,3))+' p='+str(np.round(p_mag_m7,3)))

ax4.set_title(mooring_name[3],fontsize=14,fontweight='bold')
ax4.plot(samba_mag_m8,'r',label=insitu_data_name);ax4.plot(model_mag_m8,'b',label=model_name);
ax4.text(0,0,'R${^2}$='+str(np.round(r_mag_m8,3))+' p='+str(np.round(p_mag_m8,3)))

ax5.set_title(mooring_name[4],fontsize=14,fontweight='bold')
ax5.plot(samba_mag_m9,'r',label=insitu_data_name);ax5.plot(model_mag_m9,'b',label=model_name);
ax5.text(0,0,'R${^2}$='+str(np.round(r_mag_m9,3))+' p='+str(np.round(p_mag_m9,3)))

ax6.set_title(mooring_name[5],fontsize=14,fontweight='bold')
ax6.plot(samba_mag_m10,'r',label=insitu_data_name);ax6.plot(model_mag_m10,'b',label=model_name);
ax6.text(0,0,'R${^2}$='+str(np.round(r_mag_m10,3))+' p='+str(np.round(p_mag_m10,3)))

plt.subplots_adjust(hspace=0.4,wspace=0.1)

## Setting Figure Title
fig.suptitle(figure_title,y=0.95, fontweight='bold',fontsize=16)


## Saving Figure
plt.savefig(savename_fig+ext)