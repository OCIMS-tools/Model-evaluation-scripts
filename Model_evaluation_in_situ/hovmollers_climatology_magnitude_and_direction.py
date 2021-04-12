# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:10:18 2021

@author: cristinarusso
"""
"""
This script is used to evaluate and compare model and in situ current direction and strength by plotting hovmoller diagrams, standard deviation plots, and correlation plots. 
The script compares in situ observations to the model output at a specific location.  
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
model_name = 'BRAN'
insitu_data_name = 'SAMBA'

#Path to where the netcdf data is stored 
path_to_model =  'C:/sharedfolder/SAMBA/BRAN/bran_M7.nc'
path_to_insitu = 'C:/sharedfolder/SAMBA/SAMBAM7.nc'

#Mooring Name
mooring_name = 'M7'

#Dates on which to conduct evaluation
start_date = '2014-09-18T00:00'
end_date = '2016-08-31T00:00'

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

#Setting maximum and minimum values for figures
mag_min= 0
mag_max= 0.7
mag_bias_min = -0.5
mag_bias_max = 0.5

dir_min= 0
dir_max= 360
dir_bias_min= 0
dir_bias_max= 180

#Name of figure (png file) which shows the position of the area of interest 
savename_fig = path_out+model_name+mooring_name #NO INPUT NEEDED

#%%  PREPARING OUTPUT

## Importing Modules
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import astropy.stats as ast
from astropy import units as u

## Define directional bias function
def direction_bias(insitu_direction, model_direction):
    ''' This function tells you how different the in situ direction is from the model direction. 
    The output describes whether the directions are in completely opposite directions (180), 
    perpendicular to each other (90) or whether they are the same (0) a are different '''
    bias = insitu_direction-model_direction
    proportional_difference = np.absolute((bias/360))

    difference = np.zeros(np.shape(proportional_difference))
    for i in range(len(proportional_difference[:,1])):
        for j in range(len(proportional_difference[1,:])):
            if proportional_difference[i,j]>0.5:
                difference[i,j] = (1-proportional_difference[i,j])*360
            else:
                difference[i,j] = proportional_difference[i,j]*360
    return difference

##Import Mooring data
samba_grid = xr.open_dataset(path_to_insitu)
samba_daily = samba_grid.resample(time=resample_type).mean()
samba_depth = samba_daily.depth.values
samba_grid_depth = [round(float(i), 2) for i in samba_depth]
samba_mgrid = samba_daily.sel(time=slice(start_date, end_date))


##Importing Model Data
uvdata = xr.open_dataset(path_to_model)
uvdata = uvdata.sortby('time')
## Interpolating model data onto insitu depths
uvdata_samba_grid = uvdata.interp(depth=samba_grid_depth, method='linear')
uvdata_samba_grid= uvdata_samba_grid.sel(time=slice(start_date, end_date))

## getting values
samba_dir = samba_mgrid.direction.groupby('time.month').mean('time')
samba_mag = samba_mgrid.magnitude.groupby('time.month').mean('time')/100

nan_matrix_samba = np.isnan(samba_dir.values)
samba_direction_m3 = np.copy(samba_dir.values)
samba_mag_m3 = np.copy(samba_mag.values)

model_dir = uvdata_samba_grid.direction.groupby('time.month').mean('time')
model_mag = uvdata_samba_grid.magnitude.groupby('time.month').mean('time')

nan_matrix_model = np.isnan(model_dir.values)
model_direction_m3 = np.copy(model_dir.values)
model_mag_m3 = np.copy(model_mag.values)

model_direction_m3[nan_matrix_samba==True] = np.nan 
model_mag_m3[nan_matrix_samba==True] = np.nan
samba_direction_m3[nan_matrix_model==True] = np.nan 
samba_mag_m3[nan_matrix_model==True] = np.nan

model_direction_m3[nan_matrix_model==True] = np.nan
model_mag_m3[nan_matrix_model==True] = np.nan
samba_direction_m3[nan_matrix_samba==True] = np.nan 
samba_mag_m3[nan_matrix_samba==True] = np.nan

t = np.concatenate([samba_dir.month,np.ones(1,)*13]) 
d = np.copy(samba_dir.depth)
depth, time = np.meshgrid(d,t)

# Correlations
corr_mag=[]
coeff_mag=[]
corr_dir=[]

for i in range(len(samba_grid_depth)):    
#    x, y = frame3.iloc[ :,i].values, control['CONTROL'].values
    nas = np.logical_or(np.isnan(samba_mag_m3[:,i]), np.isnan(model_mag_m3[:,i]))
    corr,coeff = stats.pearsonr(samba_mag_m3[:,i][~nas], model_mag_m3[:,i][~nas])
    corr_mag.append(corr)
    coeff_mag.append(coeff)
    corr = ast.circcorrcoef(samba_direction_m3[:,i][~nas]*u.deg, model_direction_m3[:,i][~nas]*u.deg)
    corr_dir.append(corr)

corr_mag[coeff_mag>0.05]=np.nan

mag_m3=np.concatenate([samba_mag_m3,np.zeros([1,len(samba_dir.depth.values)])],0) 
mag_m3_model=np.concatenate([model_mag_m3,np.zeros([1,len(samba_dir.depth.values)])],0)
dir_m3=np.concatenate([samba_direction_m3,np.zeros([1,len(samba_dir.depth.values)])],0) 
dir_m3_model=np.concatenate([model_direction_m3,np.zeros([1,len(samba_dir.depth.values)])],0) 

 
## FIGURE MAGNITUDE HOVMOLLER
fig = plt.figure(figsize = [12,8])  
fig.suptitle('Mooring '+mooring_name,y=0.95, fontweight='bold',fontsize=16)
grid = plt.GridSpec(2, 3, wspace=0.2, hspace=0.3)
ax1 = plt.subplot(grid[0, 0:2])
h1 = ax1.pcolor(time,-depth,mag_m3,cmap='jet',vmax=mag_max,vmin=mag_min)
plt.setp(ax1.xaxis.get_majorticklabels(), ha='center')
ax1.set_xticklabels('')
ax1.set_title('a) '+insitu_data_name+' Magnitude (m.s$^{-1}$)',fontweight='bold')


ax3 = plt.subplot(grid[1, 0:2])
h3 = ax3.pcolor(time,-depth,mag_m3_model,cmap='jet',vmax=mag_max,vmin=mag_min)
ax3.set_xticklabels(["Jan","Feb","Apr","Jun","Aug","Oct","Dec"],horizontalalignment='left')
ax3.xaxis.set_ticks_position('none')
ax3.set_title('b) '+model_name+' Magnitude (m.s$^{-1}$)',fontweight='bold')
cbar_ax = fig.add_axes([0.25, 0.515, 0.25,0.015])
fig.colorbar(h3,cax = cbar_ax,orientation = 'horizontal')

ax6 = plt.subplot(grid[0:, 2:3])
h = ax6.plot(corr_mag,-depth[0,:],'k')
ax6.set_ylim([-depth[0,-1],-depth[0,0]])
ax6.set_title('c) Correlation (p<0.05)',fontweight='bold')
ax6.plot(np.zeros(len(depth[0,:])),-depth[0,:],'k:')

plt.savefig(savename_fig+'hovmoller_magnitude_climatology'+ext)

## FIGURE MAGNITUDE HOVMOLLER
# Creating Legend for directions
bounds = [0,45,135,225,315,360]
colors = ["r", "m", "b","c","r"]
from matplotlib.patches import Patch
legend_elements = [Patch([0], [0], color='r', label='North'),
                   Patch([0], [0], color='m', label='East'),
                   Patch([0], [0], color='b', label='South'),
                   Patch([0], [0], color='c', label='West'),]


cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, len(colors))
umax= 360
umin= 0

fig = plt.figure(figsize = [12,8])  
fig.suptitle('Mooring '+mooring_name,y=0.95, fontweight='bold',fontsize=16)
grid = plt.GridSpec(2, 3, wspace=0.2, hspace=0.3)
ax1 = plt.subplot(grid[0, 0:2])
h1 = ax1.pcolor(time,-depth,dir_m3,cmap=cmap,norm=norm,vmax=dir_max,vmin=dir_min)
ax1.set_xticklabels('')
ax1.set_title('a) '+insitu_data_name+' Direction ($^{o}$)',fontweight='bold')
ax1.legend(handles=legend_elements,bbox_to_anchor=(0.15, 0.01, 1., .102), loc='lower left',
           ncol=4, borderaxespad=0.)

ax3 = plt.subplot(grid[1, 0:2])
h3 = ax3.pcolor(time,-depth,dir_m3_model,cmap=cmap,norm=norm,vmax=dir_max,vmin=dir_min)
ax3.set_xticklabels(["Jan","Feb","Apr","Jun","Aug","Oct","Dec"],horizontalalignment='left')
ax3.xaxis.set_ticks_position('none')
ax3.set_title('b) '+model_name+' Direction ($^{o}$)',fontweight='bold')
cbar_ax = fig.add_axes([0.25, 0.515, 0.25,0.015])
fig.colorbar(h3,cax = cbar_ax,orientation = 'horizontal')
    
ax6 = plt.subplot(grid[0:, 2:3])
h = ax6.plot(corr_dir,-depth[0,:],'k')
ax6.set_ylim([-depth[0,-1],-depth[0,0]])
ax6.set_title('c) Correlation',fontweight='bold')
ax6.plot(np.zeros(len(depth[0,:])),-depth[0,:],'k:')
plt.savefig(savename_fig+'hovmoller_direction_climatology'+ext)