# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:10:18 2021

@author: cristinarusso
"""
"""
This script is used to evaluate and compare model and in situ velocity variables from sectional data. 
The script compares yearly-averaged in situ observations to the model output along a vertical section.  
Scripts to perform the pre-processing are include in the tool-box. 

INPUT:
    - the model and in situ dataset name
    - the path to the model and in situ netcdf files 
    - the netcdf file containing the model output
    - the netcdf file containing the in situ observations
    - the start and end date of the time series
    - the resample type (daily, hourly, etc)
    - the depth level at which to evaluate the data
    - the desired file extention type with which to save figures

OUTPUT:
    - figures (as many as needed) with two average velocity vertical sections, one for insitu data and the other for model,
      two standard deviation vertical sections, one correlation vertical section and one RMSE vertical section

REQUIRMENTS:
    -The netcdf files need to be formatted so that the data used for comparsion are at the sepcific mooring locations with the dimensions (depth,time)

The inputs to the script should be changed where necessary is the section below called USER INPUTS
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

#Years which need to be evaluated
years = [2016,
         2017,
         2018]

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

## Calculating Yearly Means

for yr in years: 
     
    insitu_mean = insitu_data_model_grid.sel(time = slice(str(yr)+'-01-01',str(yr)+'-12-31')).mean('time')
    model_mean = cross_track.sel(time = slice(str(yr)+'-01-01',str(yr)+'-12-31')).mean('time')
    
    insitu_std = insitu_data_model_grid.sel(time = slice(str(yr)+'-01-01',str(yr)+'-12-31')).std('time')
    model_std = cross_track.sel(time = slice(str(yr)+'-01-01',str(yr)+'-12-31')).std('time')
    print(str(yr)+' mean and std calculated')
    
    
## Calculating Bias
    
    bias_year = np.zeros(np.shape(model_mean))
    bias_year[bias_year == 0] = np.nan
    bias_year = insitu_mean.v.values - model_mean.values
    print(str(yr)+' bias done')
    
## Calculating Correlations and pvalues
    
    #2016
    insitu_year = insitu_data_model_grid.sel(time = slice(str(yr)+'-01-01',str(yr)+'-12-31'))
    model_year = cross_track.sel(time = slice(str(yr)+'-01-01',str(yr)+'-12-31'))
    
    insitu_year_filled = insitu_year.fillna(99999)
    model_year_filled = model_year.fillna(99999)
    insitu_year_filled = insitu_year_filled.transpose('depth','distance','time')
    insitu_year_v = insitu_year_filled.v.values
    model_year_v = model_year_filled.transpose('depth','section_distance','time')
    
    
    corr_year = np.zeros([len(insitu_mean.depth.values),len(insitu_mean.distance.values)]);
    coeff_year = np.zeros([len(insitu_mean.depth.values),len(insitu_mean.distance.values)]);
    
    for tt in range(len(insitu_mean.depth.values)):
        for qq in range(len(insitu_mean.distance.values)):               
            
            test_corr_2,coeff = stats.pearsonr(insitu_year_v[tt,qq,:],model_year_v.values[tt,qq,:])
            corr_year[tt,qq] = test_corr_2
            coeff_year[tt,qq] = coeff
        
    
    correlation_year = corr_year
    correlation_year[coeff_year >= 0.05] = np.nan
    print('significant correlations calculated and selected')
    
## Calculating RMSE
    
    rmse_year = np.zeros([len(insitu_mean.depth.values),len(insitu_mean.distance.values)]);
    
    for tt in range(len(insitu_mean.depth.values)):
        for qq in range(len(insitu_mean.distance.values)):
            mse = mean_squared_error(insitu_year_v[tt,qq,:],model_year_v.values[tt,qq,:])
            rmse_year[tt,qq] = sqrt(mse)
    
    
    rmse_year[rmse_year == 0] = np.nan
    print('RMSE calculated for '+ str(yr))
    
#%% Plotting Output Figures
    
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
    
    
    fig.suptitle(str(yr)+insitu_data_name+' vs '+model_name+' Cross track velocity (m.s$^{-1}$)',y=0.93, fontweight='bold',fontsize=16)
    
    ax1.set_title('a) '+insitu_data_name+' mean', fontweight='bold')
    h1=ax1.pcolor(insitu_data_model_grid.distance / 1000,
               -insitu_data_model_grid.depth,
               insitu_mean,
                 cmap='seismic_r',
                 vmin=vel_min,vmax=vel_max,
                 norm=midnorm)
    plt.colorbar(h1,ax=ax1)
    ax1.set_ylabel('Depth (m)', fontweight='bold')
    
    
    ax2.set_title('b) '+model_name+' mean', fontweight='bold')
    h2=ax2.pcolor(insitu_data_model_grid.distance / 1000,
               -insitu_data_model_grid.depth,
               model_mean,
                 cmap='seismic_r',
                 vmin=vel_min,vmax=vel_max,
                 norm=midnorm)
    plt.colorbar(h2,ax=ax2)
    
    ax3.set_title('c) '+insitu_data_name+' STD', fontweight='bold')
    h3=ax3.pcolor(insitu_data_model_grid.distance / 1000,
               -insitu_data_model_grid.depth,
               insitu_std,
                 cmap='afmhot_r',
                 vmin=std_min,vmax=std_max)
    plt.colorbar(h3,ax=ax3)
    ax3.set_ylabel('Depth (m)', fontweight='bold')
    
    
    ax4.set_title('d) '+model_name+' STD',fontweight='bold')
    h4=ax4.pcolor(insitu_data_model_grid.distance / 1000,
               -insitu_data_model_grid.depth,
               model_std,
                 cmap='afmhot_r',
                 vmin=std_min,vmax=std_max)
    plt.colorbar(h4,ax=ax4)
    
    ax5.set_title('e) Bias ('+insitu_data_name+'-'+model_name+')',fontweight='bold')
    h5=ax5.pcolor(insitu_data_model_grid.distance / 1000,
               -insitu_data_model_grid.depth,
               bias_year,
                 norm=midnorm,
                 cmap='BrBG',
                 vmin=bias_min,vmax=bias_max)
    plt.colorbar(h5,ax=ax5)
    ax5.set_ylabel('Depth (m)', fontweight='bold')
    
    
    ax6.set_title('f) Correlation (p <= 0.05)',fontweight='bold')
    h6=ax6.pcolor(insitu_data_model_grid.distance / 1000,
               -insitu_data_model_grid.depth,
               correlation_year,
                  cmap='Spectral',
                  vmin= corr_min,vmax=corr_max)
    plt.colorbar(h6,ax=ax6)
    ax6.set_xlabel('Distance along transect (km)', fontweight='bold')
    
    ax7.set_title('g) RMSE', fontweight='bold')
    h7=ax7.pcolor(insitu_data_model_grid.distance / 1000,
               -insitu_data_model_grid.depth,
               rmse_year,
                  vmin=rmse_min, vmax=rmse_max,
                 cmap='bone_r')
    plt.colorbar(h7,ax=ax7)
    ax7.set_xlabel('Distance along transect (km)', fontweight='bold')
    ax7.set_ylabel('Depth (m)', fontweight='bold')
    
    ax8.axis('off')
    
    plt.subplots_adjust(hspace=0.3,wspace = 0.03)
    
    
    fig.savefig(path_out+model_name+'vs'+insitu_data_name+'_'+yr+'_stats'+ext,bbox_inches='tight')
    print(str(yr)+' figure plotted and saved')