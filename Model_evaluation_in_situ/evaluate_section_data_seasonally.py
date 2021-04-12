# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:10:18 2021

@author: cristinarusso
"""
"""
This script is used to evaluate and compare model and in situ velocity variables from sectional data. 
The script compares seasonally-averaged in situ observations to the model output along a vertical section.  
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
    - four figures (Summer, Autumn, Winer and Spring) with two average velocity vertical sections, one for insitu data and the other for model,
      two standard deviation vertical sections, one correlation vertical section and one RMSE vertical section

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

#%% Calculating Seasonal Means

model_seasons_mean = cross_track.groupby('time.season').mean('time')
insitu_seasons_mean = insitu_data_model_grid.groupby('time.season').mean('time')
print('seasonal means calculated')

#%% Calculating Seasonal Standard Deviations

model_seasons_std = cross_track.groupby('time.season').std('time')
insitu_seasons_std = insitu_data_model_grid.groupby('time.season').std('time')
print('seasonal standard deviations calculated')

#%% Seasonal Mean Velocities and Standard deviations 

#SUMMER
insitu_summer_mean = insitu_seasons_mean.v.sel(season='DJF')
model_summer_mean = model_seasons_mean.sel(season='DJF')
insitu_summer_std = insitu_seasons_std.v.sel(season='DJF')
model_summer_std = model_seasons_std.sel(season='DJF')

#AUTUMN
insitu_autumn_mean = insitu_seasons_mean.v.sel(season='MAM')
model_autumn_mean = model_seasons_mean.sel(season='MAM')
insitu_autumn_std = insitu_seasons_std.v.sel(season='MAM')
model_autumn_std = model_seasons_std.sel(season='MAM')

#WINTER
insitu_winter_mean = insitu_seasons_mean.v.sel(season='JJA')
model_winter_mean = model_seasons_mean.sel(season='JJA')
insitu_winter_std = insitu_seasons_std.v.sel(season='JJA')
model_winter_std = model_seasons_std.sel(season='JJA')

#SPRING
insitu_spring_mean = insitu_seasons_mean.v.sel(season='SON')
model_spring_mean = model_seasons_mean.sel(season='SON')
insitu_spring_std = insitu_seasons_std.v.sel(season='SON')
model_spring_std = model_seasons_std.sel(season='SON')

#%% Calculating Seasonal Biases

summer_bias = np.zeros(np.shape(insitu_summer_mean))
autumn_bias = np.zeros(np.shape(insitu_summer_mean))
winter_bias = np.zeros(np.shape(insitu_summer_mean))
spring_bias = np.zeros(np.shape(insitu_summer_mean))

summer_bias[summer_bias == 0] = np.nan
autumn_bias[autumn_bias == 0] = np.nan
winter_bias[winter_bias == 0] = np.nan
spring_bias[spring_bias == 0] = np.nan

for i in range(len(insitu_data_model_grid.depth.values)):
    for j in range(len(insitu_data_model_grid.distance.values)):
        
        summer_bias[i,j] = insitu_summer_mean.values[i,j] - model_summer_mean.values[i,j]
        autumn_bias[i,j] = insitu_autumn_mean.values[i,j] - model_autumn_mean.values[i,j]
        winter_bias[i,j] = insitu_winter_mean.values[i,j] - model_winter_mean.values[i,j]
        spring_bias[i,j] = insitu_spring_mean.values[i,j] - model_spring_mean.values[i,j]

print('seasonal biases calculated')

#%% Calculating Correlations and pvalues

model_seasons = cross_track.groupby('time.season').fillna(99999) #removing nans to calculate correlation
model_seasons_index = cross_track['time'].dt.season

insitu_seasons = insitu_data_model_grid.groupby('time.season').fillna(99999) #removing nans to calculate correlation
insitu_seasons_index = insitu_data_model_grid['time'].dt.season

model_summer = model_seasons[model_seasons_index == 'DJF'].transpose('depth','section_distance','time')
model_autumn = model_seasons[model_seasons_index == 'MAM'].transpose('depth','section_distance','time')
model_winter = model_seasons[model_seasons_index == 'JJA'].transpose('depth','section_distance','time')
model_spring = model_seasons[model_seasons_index == 'SON'].transpose('depth','section_distance','time')

insitu_summer = insitu_seasons.v[insitu_seasons_index == 'DJF'].transpose('depth','distance','time')
insitu_autumn = insitu_seasons.v[insitu_seasons_index == 'MAM'].transpose('depth','distance','time')
insitu_winter = insitu_seasons.v[insitu_seasons_index == 'JJA'].transpose('depth','distance','time')
insitu_spring = insitu_seasons.v[insitu_seasons_index == 'SON'].transpose('depth','distance','time')

corr_summer = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);
coeff_summer = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);

corr_autumn = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);
coeff_autumn = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);

corr_winter= np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);
coeff_winter = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);

corr_spring = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);
coeff_spring = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);


for i in range(len(insitu_data_model_grid.depth.values)):
    for j in range(len(insitu_data_model_grid.distance.values)):
        #SUMMER
        test_corr_2,coeff = stats.pearsonr(insitu_summer.values[i,j,:],model_summer.values[i,j,:])
        corr_summer[i,j] = test_corr_2
        coeff_summer[i,j] = coeff
        #AUTUMN
        test_corr_2,coeff  = stats.pearsonr(insitu_autumn.values[i,j,:],model_autumn.values[i,j,:])
        corr_autumn[i,j] = test_corr_2
        coeff_autumn[i,j] = coeff
        #WINTER
        test_corr_2,coeff  = stats.pearsonr(insitu_winter.values[i,j,:],model_winter.values[i,j,:])
        corr_winter[i,j] = test_corr_2
        coeff_winter[i,j] = coeff
        #SPRING
        test_corr_2,coeff  = stats.pearsonr(insitu_spring.values[i,j,:],model_spring.values[i,j,:])
        corr_spring [i,j] = test_corr_2
        coeff_spring[i,j] = coeff

# Selecting on statistically significant correlations
correlation_summer = np.copy(corr_summer)
correlation_autumn = np.copy(corr_autumn)
correlation_winter = np.copy(corr_winter)
correlation_spring = np.copy(corr_spring)
print('correlations calculated')

correlation_summer[coeff_summer >= 0.05] = np.nan
correlation_autumn[coeff_autumn >= 0.05] = np.nan
correlation_winter[coeff_winter >= 0.05] = np.nan
correlation_spring[coeff_spring >= 0.05] = np.nan
print('significant correlations selected')


#%% Calculating RMSE

rmse_summer = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);
rmse_autumn = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);
rmse_winter = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);
rmse_spring = np.zeros([len(insitu_data_model_grid.depth.values),len(insitu_data_model_grid.distance.values)]);

for tt in range(len(insitu_data_model_grid.depth.values)):
    for qq in range(len(insitu_data_model_grid.distance.values)):
        #SUMMER
        mse = mean_squared_error(insitu_summer.values[tt,qq,:],model_summer.values[tt,qq,:])
        rmse_summer[tt,qq] = sqrt(mse)
        #AUTUMN
        mse = mean_squared_error(insitu_autumn.values[tt,qq,:],model_autumn.values[tt,qq,:])
        rmse_autumn[tt,qq] = sqrt(mse)
        #WINTER
        mse = mean_squared_error(insitu_winter.values[tt,qq,:],model_winter.values[tt,qq,:])
        rmse_winter[tt,qq] = sqrt(mse)
        #SPRING
        mse = mean_squared_error(insitu_spring.values[tt,qq,:],model_spring.values[tt,qq,:])
        rmse_spring[tt,qq] = sqrt(mse)

rmse_summer[rmse_summer == 0] = np.nan
rmse_autumn[rmse_autumn == 0] = np.nan
rmse_winter[rmse_winter == 0] = np.nan
rmse_spring[rmse_spring == 0] = np.nan
print('RMSE calculated')

#%% Plotting the Summer Figure

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


fig.suptitle('Summer (DJF) ASCA vs BRAN Cross track velocity (m.s$^{-1}$)',y=0.93, fontweight='bold',fontsize=16)

ax1.set_title('a) '+insitu_data_name+' mean', fontweight='bold')
h1=ax1.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_summer_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h1,ax=ax1)
ax1.set_ylabel('Depth (m)', fontweight='bold')


ax2.set_title('b) '+model_name+' mean', fontweight='bold')
h2=ax2.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_summer_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h2,ax=ax2)

ax3.set_title('c) '+insitu_data_name+' STD', fontweight='bold')
h3=ax3.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_summer_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h3,ax=ax3)
ax3.set_ylabel('Depth (m)', fontweight='bold')


ax4.set_title('d) '+model_name+' STD',fontweight='bold')
h4=ax4.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_summer_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h4,ax=ax4)

ax5.set_title('e) Bias ('+insitu_data_name+'-'+model_name+')',fontweight='bold')
h5=ax5.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           summer_bias,
             norm=midnorm,
             cmap='BrBG',
             vmin=bias_min,vmax=bias_max)
plt.colorbar(h5,ax=ax5)
ax5.set_ylabel('Depth (m)', fontweight='bold')


ax6.set_title('f) Correlation (p <= 0.05)',fontweight='bold')
h6=ax6.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           correlation_summer,
              cmap='Spectral',
              vmin= corr_min,vmax=corr_max)
plt.colorbar(h6,ax=ax6)
ax6.set_xlabel('Distance along transect (km)', fontweight='bold')

ax7.set_title('g) RMSE', fontweight='bold')
h7=ax7.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           rmse_summer,
              vmin=rmse_min, vmax=rmse_max,
             cmap='bone_r')
plt.colorbar(h7,ax=ax7)
ax7.set_xlabel('Distance along transect (km)', fontweight='bold')
ax7.set_ylabel('Depth (m)', fontweight='bold')

ax8.axis('off')

plt.subplots_adjust(hspace=0.3,wspace = 0.03)


fig.savefig(path_out+model_name+'vs'+insitu_data_name+'_summer_stats'+ext,bbox_inches='tight')
print('summer figure plotted and saved')

#%% Plotting Autumn Figure

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


fig.suptitle('Autumn (MAM) ASCA vs BRAN Cross track velocity (m.s$^{-1}$)',y=0.93, fontweight='bold',fontsize=16)

ax1.set_title('a) '+insitu_data_name+' mean', fontweight='bold')
h1=ax1.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_autumn_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h1,ax=ax1)
ax1.set_ylabel('Depth (m)', fontweight='bold')


ax2.set_title('b) '+model_name+' mean', fontweight='bold')
h2=ax2.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_autumn_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h2,ax=ax2)

ax3.set_title('c) '+insitu_data_name+' STD', fontweight='bold')
h3=ax3.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_autumn_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h3,ax=ax3)
ax3.set_ylabel('Depth (m)', fontweight='bold')


ax4.set_title('d) '+model_name+' STD',fontweight='bold')
h4=ax4.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_autumn_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h4,ax=ax4)

ax5.set_title('e) Bias ('+insitu_data_name+'-'+model_name+')',fontweight='bold')
h5=ax5.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           autumn_bias,
             norm=midnorm,
             cmap='BrBG',
             vmin=bias_min,vmax=bias_max)
plt.colorbar(h5,ax=ax5)
ax5.set_ylabel('Depth (m)', fontweight='bold')


ax6.set_title('f) Correlation (p <= 0.05)',fontweight='bold')
h6=ax6.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           correlation_autumn,
              cmap='Spectral',
              vmin= corr_min,vmax=corr_max)
plt.colorbar(h6,ax=ax6)
ax6.set_xlabel('Distance along transect (km)', fontweight='bold')

ax7.set_title('g) RMSE', fontweight='bold')
h7=ax7.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           rmse_autumn,
              vmin=rmse_min, vmax=rmse_max,
             cmap='bone_r')
plt.colorbar(h7,ax=ax7)
ax7.set_xlabel('Distance along transect (km)', fontweight='bold')
ax7.set_ylabel('Depth (m)', fontweight='bold')

ax8.axis('off')

plt.subplots_adjust(hspace=0.3,wspace = 0.03)


fig.savefig(path_out+model_name+'vs'+insitu_data_name+'_autumn_stats'+ext,bbox_inches='tight')
print('autumn figure plotted and saved')

#%% Plotting Winter Figure

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


fig.suptitle('Winter (JJA) ASCA vs BRAN Cross track velocity (m.s$^{-1}$)',y=0.93, fontweight='bold',fontsize=16)

ax1.set_title('a) '+insitu_data_name+' mean', fontweight='bold')
h1=ax1.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_winter_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h1,ax=ax1)
ax1.set_ylabel('Depth (m)', fontweight='bold')


ax2.set_title('b) '+model_name+' mean', fontweight='bold')
h2=ax2.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_winter_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h2,ax=ax2)

ax3.set_title('c) '+insitu_data_name+' STD', fontweight='bold')
h3=ax3.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_winter_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h3,ax=ax3)
ax3.set_ylabel('Depth (m)', fontweight='bold')


ax4.set_title('d) '+model_name+' STD',fontweight='bold')
h4=ax4.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_winter_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h4,ax=ax4)

ax5.set_title('e) Bias ('+insitu_data_name+'-'+model_name+')',fontweight='bold')
h5=ax5.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           winter_bias,
             norm=midnorm,
             cmap='BrBG',
             vmin=bias_min,vmax=bias_max)
plt.colorbar(h5,ax=ax5)
ax5.set_ylabel('Depth (m)', fontweight='bold')


ax6.set_title('f) Correlation (p <= 0.05)',fontweight='bold')
h6=ax6.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           correlation_winter,
              cmap='Spectral',
              vmin= corr_min,vmax=corr_max)
plt.colorbar(h6,ax=ax6)
ax6.set_xlabel('Distance along transect (km)', fontweight='bold')

ax7.set_title('g) RMSE', fontweight='bold')
h7=ax7.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           rmse_winter,
              vmin=rmse_min, vmax=rmse_max,
             cmap='bone_r')
plt.colorbar(h7,ax=ax7)
ax7.set_xlabel('Distance along transect (km)', fontweight='bold')
ax7.set_ylabel('Depth (m)', fontweight='bold')

ax8.axis('off')

plt.subplots_adjust(hspace=0.3,wspace = 0.03)

fig.savefig(path_out+model_name+'vs'+insitu_data_name+'_winter_stats'+ext,bbox_inches='tight')
print('winter figure plotted and saved')

#%% Plotting Spring Figure

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


fig.suptitle('Spring (SON) ASCA vs BRAN Cross track velocity (m.s$^{-1}$)',y=0.93, fontweight='bold',fontsize=16)

ax1.set_title('a) '+insitu_data_name+' mean', fontweight='bold')
h1=ax1.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_spring_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h1,ax=ax1)
ax1.set_ylabel('Depth (m)', fontweight='bold')


ax2.set_title('b) '+model_name+' mean', fontweight='bold')
h2=ax2.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_spring_mean,
             cmap='seismic_r',
             vmin=vel_min,vmax=vel_max,
             norm=midnorm)
plt.colorbar(h2,ax=ax2)

ax3.set_title('c) '+insitu_data_name+' STD', fontweight='bold')
h3=ax3.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           insitu_spring_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h3,ax=ax3)
ax3.set_ylabel('Depth (m)', fontweight='bold')


ax4.set_title('d) '+model_name+' STD',fontweight='bold')
h4=ax4.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           model_spring_std,
             cmap='afmhot_r',
             vmin=std_min,vmax=std_max)
plt.colorbar(h4,ax=ax4)

ax5.set_title('e) Bias ('+insitu_data_name+'-'+model_name+')',fontweight='bold')
h5=ax5.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           spring_bias,
             norm=midnorm,
             cmap='BrBG',
             vmin=bias_min,vmax=bias_max)
plt.colorbar(h5,ax=ax5)
ax5.set_ylabel('Depth (m)', fontweight='bold')


ax6.set_title('f) Correlation (p <= 0.05)',fontweight='bold')
h6=ax6.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           correlation_spring,
              cmap='Spectral',
              vmin= corr_min,vmax=corr_max)
plt.colorbar(h6,ax=ax6)
ax6.set_xlabel('Distance along transect (km)', fontweight='bold')

ax7.set_title('g) RMSE', fontweight='bold')
h7=ax7.pcolor(insitu_data_model_grid.distance / 1000,
           -insitu_data_model_grid.depth,
           rmse_spring,
              vmin=rmse_min, vmax=rmse_max,
             cmap='bone_r')
plt.colorbar(h7,ax=ax7)
ax7.set_xlabel('Distance along transect (km)', fontweight='bold')
ax7.set_ylabel('Depth (m)', fontweight='bold')

ax8.axis('off')

plt.subplots_adjust(hspace=0.3,wspace = 0.03)

fig.savefig(path_out+model_name+'vs'+insitu_data_name+'_spring_stats'+ext,bbox_inches='tight')
print('spring figure plotted and saved')