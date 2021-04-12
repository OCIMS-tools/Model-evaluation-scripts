# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:10:18 2021

@author: cristinarusso
"""
"""
This script is used to evaluate and compare model and in situ variables from sectional data. 
The script compares in situ observations to the model output from a vertical section in the form of Taylor Diagram.  
Scripts to perform the pre-processing are include in the tool-box. 

INPUT:
    - the model and in situ dataset name
    - the path to the model and in situ netcdf files 
    - the netcdf file containing the model output
    - the netcdf file containing the in situ observations
    - the start and end date of the time series
    - the resample type (daily, hourly, etc)
    - the depth level at which to evaluate the data
    - the distance at which to evaluate the data
    - the desired file extention type with which to save figures
    - the desired variable which is velocity
    
OUTPUT:
    - Taylor Diagram and Line Graph for time series of chosen variable
    - Taylor Diagram and Line Graph for Monthly Climatology of chosen variable
      

REQUIRMENTS:
    -The netcdf files need to be formatted so that the data used for comparsion are at the sepcific mooring locations with the dimensions (depth,time)

The inputs to the script should be changed where necessary is the section below called USER INPUTS
"""
#%% USER INPUTS

#Name of the region (used for graph title and file saving)
model_name = 'GLORYS'
insitu_data_name = 'ASCA'

#Path to where the netcdf data is stored 
path_to_model = '/media/sf_sharedfolder/ASCA/glorys/glorys_ASCA_section_data_test.nc'
path_to_insitu = '/media/sf_sharedfolder/ASCA/ASCA_corrected_dimensions2.nc'

#Date range on which to conduct evaluation
start_date = ['2016-04-19T00:00', #insitu date
              '2016-04-19T12:00']#model date NOTE: different in time
end_date = ['2018-05-31T00:00',#insitu date
            '2018-05-31T12:00']#model date NOTE: different in time

#Distance range on which to conduct evaluation
start_distance = 240000
end_distance = 251000
dist_label_title = str(start_distance/1000)


#Depth range on which to conduct evaluation
start_depth = 90
end_depth = 100
depth_label_title = str(end_depth)

#Variables to evaluate
var = 'v'
var_label = 'Velocity ($m.s^{-1}$)'
var_label_title = 'Velocity'

#Reseampling of data to daily/hourly
resample_type = '1D'

#Figure Title
fig_title= insitu_data_name+' vs '+model_name+' '+var_label_title+' '+'('+dist_label_title+' km offshore, '+depth_label_title+' m depth)'

#Path to where the figures should be stored 
path_out = '/media/sf_sharedfolder/ASCA/bluelink/figure/new/'

#Figure extention
ext = '.png'

#Name of figure (png file) which shows the position of the area of interest 
savename_fig = path_out+model_name+'vs'+insitu_data_name #NO INPUT NEEDED


#%% PREPARING DATA FOR OUTPUT
## Defining rotation function for cross and along track velocities

def uv_rotate(u, v, angle):
    """
    USAGE:
        u_rotate = u*np.cos(angle) - v*np.sin(angle)
        v_rotate = u*np.sin(angle) + v*np.cos(angle)
        
        Rotate UV components by angle
    Input: 
        U, V, angle in degrees
    Outputs: 
        u_rotate, v_rotate
    """
    angle_rad = angle * np.pi / 180
    u_rot = u * np.cos(angle_rad) - v * np.sin(angle_rad)
    v_rot = u * np.sin(angle_rad) + v * np.cos(angle_rad)
    return u_rot, v_rot

## Importing Python Modules

import xarray as xr
import scipy.stats as stats
import math
import matplotlib.pylab as plt #new
from taylordiagram import TaylorDiagram
import numpy as np

## Importing model data

model_data = xr.open_dataset(path_to_model)
model_data = model_data.sortby('time') #sort the data chronological order
model_data = model_data.sel(time=slice(start_date[1],end_date[1])) #select data in insitu range
m_depth = model_data.depth.values
model_depth = [round(float(i), 2) for i in m_depth]
model_dist = model_data.section_distance.values
print('model imported')

## Importing insitu data

insitu_data = xr.open_dataset(path_to_insitu)
insitu_data_daily = insitu_data.resample(time=resample_type).mean()
insitu_data_daily = insitu_data_daily.sel(time=slice(start_date[0],end_date[0]))

insitu_data_model_grid = insitu_data_daily.interp(depth=model_depth,distance=model_dist, method='linear') #fit asca onto model grid
print('insitu imported')

## Calculating along and cross track velocities

u_data = model_data.u_vel
v_data = model_data.v_vel
dy = model_data.Lat[-1].values - model_data.Lat[0]
dx = model_data.Lon[-1].values - model_data.Lon[0]
along_track, cross_track = uv_rotate(u_data, v_data,
                                     -math.degrees(math.atan2(dy, dx)))
print('cross-track velocity calculated')

#%% Calculating Offshore Means

#Insitu Offshore
insitu_mean = insitu_data_model_grid.loc[dict(distance = slice(start_distance,end_distance),depth=slice(start_depth,end_depth))].mean(dim=('depth','distance'),skipna=True)
insitu_std = insitu_data_model_grid.loc[dict(distance = slice(start_distance,end_distance),depth=slice(start_depth,end_depth))].std(dim=('time','depth','distance'),skipna=True)
insitu_mean = insitu_mean.fillna(99999)

#Model Offshore
model_mean = cross_track.loc[dict(section_distance = slice(start_distance,end_distance),depth=slice(start_depth,end_depth))].mean(dim=('depth','section_distance'),skipna=True)
model_std = cross_track.loc[dict(section_distance = slice(start_distance,end_distance),depth=slice(start_depth,end_depth))].std(dim=('time','depth','section_distance'),skipna=True)
model_mean = model_mean.fillna(99999)

print('means calculated')

#%% Plotting Total Velocity Offshore Taylor Diagram

# Setting up Taylor Diagram values
corr, coeff = stats.pearsonr(insitu_mean[var].values,model_mean.values)
correlation = 'Correlation: ' + str(round(corr,3)) + ' ' +'(p='+str(round(coeff,3))+')'

# Reference std
stdref = insitu_std[var].values

# Samples std,rho (correlation?),name
samples = [[model_std.values, corr, model_name]]

fig = plt.figure(figsize=[10,7])

dia = TaylorDiagram(stdref, fig=fig, label=insitu_data_name, extend=True)
dia.samplePoints[0].set_color('r')  # Mark reference point as a red star

# Add models to Taylor diagram
for i, (stddev, corrcoef, name) in enumerate(samples):
    dia.add_sample(stddev, corrcoef,marker='o', ms=10, mfc='k', mec='k', label=name)

contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels in grey
plt.clabel(contours, inline=1, fontsize=10, fmt='%.3f')

dia.add_grid()                                  # Add grid
dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

# Add a figure legend and title
fig.legend(dia.samplePoints,
            [ p.get_label() for p in dia.samplePoints ],
            numpoints=1, prop=dict(size='small'), loc=[0.32,0.8])

ax = fig.add_axes([1,.2,1,.6]) 
ax.plot(insitu_mean.time,insitu_mean[var],label=insitu_data_name,linewidth=2,c='r')
ax.plot(model_mean.time,model_mean,label=model_name,linewidth=2,c='k')
ax.legend(loc='upper left')
ax.set_ylabel(var_label,fontweight='bold')
ax.text(model_mean.time[0],17.4,correlation,fontweight='bold')
ax.set_title(fig_title, size='x-large',fontweight='bold')  # Figure title
fig.savefig(savename_fig+'_taylordiagram_'+var_label_title+'_'+dist_label_title+'offshore_'+depth_label_title+ext,bbox_inches='tight')

print('Taylor Diagram plotted')

#%% Calculating Monthly Climatology for Offshore Temperature

#Insitu Total Velocity Offshore Climatology
insitu_months = insitu_data_model_grid.loc[dict(distance = slice(start_distance,end_distance),depth=slice(start_depth,end_depth))].groupby('time.month').mean('time')
insitu_months_1 = insitu_months.mean(dim=('depth','distance'),skipna=True)
insitu_months_1_std = insitu_months.std(dim=('month','depth','distance'),skipna=True)
insitu_months_1 = insitu_months_1.fillna(99999)

#Model Total Velocity Offshore Climatology
model_months = cross_track.loc[dict(section_distance = slice(start_distance,end_distance),depth=slice(start_depth,end_depth))].groupby('time.month').mean('time')
model_months_1 = model_months.mean(dim=('depth','section_distance'),skipna=True)
model_months_1_std = model_months.std(dim=('month','depth','section_distance'),skipna=True)
model_months_1 = model_months_1.fillna(99999)

#%% Plotting Taylor Diagram for Total Velocity Offshore Climatology

corr, coeff = stats.pearsonr(insitu_months_1[var].values,model_months_1.values)
correlation = 'Correlation: ' + str(round(corr,3)) + ' ' +'(p='+str(round(coeff,3))+')'

# Reference std
stdref = insitu_months_1_std[var].values

# Samples std,rho (correlation?),name
samples = [[model_months_1_std.values, corr, model_name]]

fig = plt.figure(figsize=[10,7])

dia = TaylorDiagram(stdref, fig=fig, label=insitu_data_name,extend=True)
dia.samplePoints[0].set_color('r')  # Mark reference point as a red star

# Add models to Taylor diagram
for i, (stddev, corrcoef, name) in enumerate(samples):
    dia.add_sample(stddev, corrcoef,marker='o', ms=10, mfc='k', mec='k', label=name)

contours = dia.add_contours(levels=5, colors='0.5')  # 5 levels in grey
plt.clabel(contours, inline=1, fontsize=10, fmt='%.3f')
dia.add_grid()                                  # Add grid
dia._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

# Add a figure legend and title
fig.legend(dia.samplePoints,
            [ p.get_label() for p in dia.samplePoints ],
            numpoints=1, prop=dict(size='small'), loc=[0.32,0.8])

ax = fig.add_axes([1,.2,1,.6]) 

ax.plot(insitu_months_1.month,insitu_months_1,label=insitu_data_name,linewidth=2,c='r')
ax.plot(model_months_1.month,model_months_1,label=model_name,linewidth=2,c='k')
ax.legend(loc='upper left')
ax.set_ylabel(var_label,fontweight='bold')
ax.text(2,19,correlation,fontweight='bold')

ax.set_title('Monthly Climatology for '+fig_title, size='x-large',fontweight='bold')  # Figure title

fig.savefig(savename_fig+'_climatology'+ext,bbox_inches='tight')

print('plotted seasonality Taylor Diagram')