# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:10:18 2021

@author: cristinarusso
"""
"""
This script is used to evaluate and compare model and in situ velocity variables from sectional data. 
The script compares timeseries-averaged in situ observations to the model output along a vertical section.  
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
    - a line graph comparing insitu and model velocity
    - a figure with two average velocity vertical sections, one for insitu data and the other for model,
      two standard deviation vertical sections, one correlation vertical section and one RMSE vertical section
    - a scatter plot comparing insitu and model average velocity

REQUIRMENTS:
    -The netcdf files need to be formatted so that the data used for comparsion are at the sepcific mooring locations with the dimensions (depth,time)

The inputs to the script should be changed where nessaccary is the section below called USER INPUTS
"""
#%% USER INPUTS

#Name of the region (used for graph title and file saving)
model_name = 'BRAN'
insitu_data_name = 'ASCA'

#Path to where the netcdf data is stored 
path_to_model =  '/media/sf_sharedfolder/ASCA/bluelink/bran_ASCA_section_data_test.nc'
path_to_insitu = '/media/sf_sharedfolder/ASCA/ASCA_corrected_dimensions2.nc'

#Dates on which to conduct evaluation
start_date = ['2016-04-19T00:00', #insitu date
              '2016-04-19T12:00']#model date NOTE: different in time
end_date = ['2018-05-31T00:00',#insitu date
            '2018-05-31T12:00']#model date NOTE: different in time
 
#Reseampling of data to daily/hourly
resample_type = '1D'

#Path to where the figures should be stored 
path_out = '/media/sf_sharedfolder/ASCA/bluelink/figure/new/'

#Figure extention
ext = '.png'

#Setting maximum and minimum values for figures
vel_min  = -1.6
vel_max  = 0.1
std_min  = 0
std_max  = 0.8
bias_min = -0.5
bias_max = 0.5
corr_min = -1
corr_max = 1
rmse_min = 0
rmse_max = 1

#Figure Title
figure_title=''

#Name of figure (png file) which shows the position of the area of interest 
savename_fig = path_out+model_name #NO INPUT NEEDED


#%% PREPARING DATA FOR OUTPUT

## Importing Python Modules

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import math
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt

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

## Defining Correlation coefficient function
    
def r2(x,y):
    return stats.pearsonr(x,y)[0]**2
## Importing model data

model_data = xr.open_dataset(path_to_model)
model_data = model_data.sortby('time') #sort the data chronological order
model_data = model_data.sel(time=slice(start_date[1],end_date[1])) #select data in ACT range
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

## Calculating Timeseries-long mean

insitu_years_mean = insitu_data_model_grid.v.mean('time') #ASCA mean
model_years_mean = cross_track.mean('time') #GLORYS mean
print('mean calculated')

## Calculating Timeseries-long standard deviation

insitu_years_std = insitu_data_model_grid.v.std('time')
model_years_std = cross_track.std('time')
print('standard deviation calculated')

## Calculating Bias

bias_years = insitu_years_mean.values - model_years_mean.values
print('bias calculated')

## Quick renaming of variables

insitu_years = insitu_data_model_grid
model_years = cross_track
insitu_years_v = insitu_years.v.values

## Preparing variables for correlation and RMSE calculations, must remove nans and reshape

insitu_years_filled = insitu_years.fillna(99999)
model_years_filled = model_years.fillna(99999)
insitu_years_v = insitu_years_filled.v.transpose('depth','distance','time')
model_years_v = model_years_filled.transpose('depth','section_distance','time')

## Calculating Correlations and pvalues

corr_years = np.zeros([len(insitu_years_mean.depth.values),len(insitu_years_mean.distance.values)]);
coeff_years = np.zeros([len(insitu_years_mean.depth.values),len(insitu_years_mean.distance.values)]);
for i in range(len(insitu_years_mean.depth.values)):
    for j in range(len(insitu_years_mean.distance.values)):
        
        test_corr_2,coeff = stats.pearsonr(insitu_years_v[i,j,:],model_years_v.values[i,j,:])
        corr_years[i,j] = test_corr_2
        coeff_years[i,j] = coeff

correlation_years = np.copy(corr_years)
print('correlations calculated')

correlation_years[coeff_years >= 0.05] = np.nan
print('significant correlations (p>0.05) selected')

## Calculating RMSE

rmse_years = np.zeros([len(insitu_years_mean.depth.values),len(insitu_years_mean.distance.values)]);
for i in range(len(insitu_years_mean.depth.values)):
    for j in range(len(insitu_years_mean.distance.values)):
        
        mse = mean_squared_error(insitu_years_v[i,j,:],model_years_v.values[i,j,:])
        rmse_years[i,j] = sqrt(mse)

rmse_years[rmse_years == 0] = np.nan
print('RMSE calculated')


## Average velocity timeseries

insitu_ave_velocity = []
model_ave_velocity = []
for i in range(len(insitu_data_model_grid.time)):
    insitu_ave_velocity.append(np.nanmean(insitu_data_model_grid.v[i,:,:]))
    model_ave_velocity.append(np.nanmean(cross_track[i,:,:]))

print('Mean velocities calculated')


#%% PLOTTING THE OUTPUT FIGURES

## Plotting Average Velocity Line Graph
fig = plt.figure(figsize=[15,5])	
plt.plot(insitu_data_model_grid.time,insitu_ave_velocity, 'SteelBlue',lw=4,label=insitu_data_name);plt.plot(insitu_data_model_grid.time, model_ave_velocity,'DarkOrange',lw=4,label=model_name)
plt.ylabel('Velocity (m/s)',fontsize=13)
plt.text(insitu_data_model_grid.time[60],-0.07,'$R^{2}$ = '+str(round(r2(insitu_ave_velocity,model_ave_velocity),4)))
plt.legend()
fig.savefig(path_out+model_name+'vs'+insitu_data_name+'_velocitytimeseries_stats'+ext,bbox_inches='tight')
plt.close()

print('Timeseries Line Graph Figure plotted and saved')

##Plotting Average Sectional Figure

midnorm = mpl.colors.DivergingNorm(vcenter=0)

fig,axes = plt.subplots(4,2,figsize=[20,15])

ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]
ax5 = axes[2,0]
ax6 = axes[2,1]
ax7 = axes[3,0]
ax8 = axes[3,1]


fig.suptitle(insitu_data_name+' vs '+model_name+' Cross track velocity (m.s$^{-1}$)',y=0.93, fontweight='bold',fontsize=16)

ax1.set_title('a) '+insitu_data_name+' mean', fontweight='bold')
h1=ax1.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_years_mean.v,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h1,ax=ax1)
ax1.set_ylabel('Depth (m)', fontweight='bold')


ax2.set_title('b) '+model_name+' mean', fontweight='bold')
h2=ax2.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_years_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h2,ax=ax2)

ax3.set_title('c) '+insitu_data_name+' STD', fontweight='bold')
h3=ax3.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_years_std.v,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h3,ax=ax3)
ax3.set_ylabel('Depth (m)', fontweight='bold')


ax4.set_title('d) '+model_name+' STD',fontweight='bold')
h4=ax4.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_years_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h4,ax=ax4)

ax5.set_title('e) Bias ('+insitu_data_name+'-'+model_name+')',fontweight='bold')
h5=ax5.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           bias_years,
             norm=midnorm,
             cmap='BrBG',
             vmin=bias_min,vmax=bias_max)
plt.colorbar(h5,ax=ax5)
ax5.set_ylabel('Depth (m)', fontweight='bold')


ax6.set_title('f) Correlation (p <= 0.05)',fontweight='bold')
h6=ax6.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           correlation_years,
              cmap='Spectral',
              vmin=corr_min,vmax=corr_max)
plt.colorbar(h6,ax=ax6)
ax6.set_xlabel('Distance along transect (km)', fontweight='bold')

ax7.set_title('g) RMSE', fontweight='bold')
h7=ax7.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           rmse_years,vmin=rmse_min,vmax=rmse_max,
             cmap='bone_r')
plt.colorbar(h7,ax=ax7)
ax7.set_xlabel('Distance along transect (km)', fontweight='bold')
ax7.set_ylabel('Depth (m)', fontweight='bold')

ax8.axis('off')

plt.subplots_adjust(hspace=0.3,wspace = 0.03)

fig.savefig(path_out+model_name+'vs'+insitu_data_name+'_years_stats'+ext,bbox_inches='tight')
print('Sectional Figure plotted and saved')
plt.close()

## Plotting Average velocities scatter plot
fig = plt.figure(figsize=[6,6])	
plt.scatter(insitu_ave_velocity, model_ave_velocity,color='YellowGreen')
z = np.polyfit(insitu_ave_velocity, model_ave_velocity,1)
p = np.poly1d(z)
plt.plot(insitu_ave_velocity,p(insitu_ave_velocity),'k:')

plt.ylabel(insitu_data_name+' Velocity (m/s)',fontsize=13)
plt.xlabel(model_name+' Velocity (m/s)',fontsize=13)
plt.text(-0.43,-0.2,'$R^{2}$ = '+str(round(r2(insitu_ave_velocity,model_ave_velocity),4)))

fig.savefig(path_out+model_name+'vs'+insitu_data_name+'_velocitytimeseries_scatter_plot'+ext,bbox_inches='tight')
plt.close()
print('Scatter Plot Figure plotted and saved')