#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 11:46:11 2020

@author: matty
"""
"""
This script is used to examine specific regions of interest by comparing area-average regions between model output and observations. 
The script compares multiple observations to the model output, in order to account for potential variation between observations.  
Scripts to perform the pre-processing are include in the tool-box named: 

INPUT:
    - a netcdf file containing the model output a
    - the netcdf files containing the observations datasets
    - the title and co-oridantes of the region of interest
    - the path and name of the mutliple netcdf files 
    - the name of observation and model dataset for headings and legends 
    - the name of the variables to be compared (! as written in netcdf file !)
    - the name of the lon and lat dimesion of the model data (! as written in netcdf file !)

OUTPUT:
    - A map of the region of interest 
    - A timeseries of the daily climatology for each dataset
    - A timeseries of the daily anomalies for each dataset
    - A taylor diagram using the an array of the median for each of the datasets as the taylor diagram observations 
    - A taylor diagram of the daily anomalies using the an array of the median for each of the datasets as the taylor diagram observations 


REQUIRMENTS:
    -The netcdf files need to be formatted so that the variabled used for comparsion are a 3 dimesional matrix of (time, longitude, latitude)
    -The netcdf files MUST be regridded to identical grids 
    -The netcdf files MUST be sampled at a daily frequency and the time dimesion should have the same length (there can not be missing days in the data) 

The inputs to the script should be changed where nessaccary is the section below called USER INPUTS
"""

# %% USER INPUTS

#Name of the region (used for graph title and file saving)
position_name='Agulhas Bank '

#Cooridates of the region of interest 
west=20.1
east=21.1
south=-35.8
north=-34.8

#Path to where the netcdf data is stored 
path_to_model =  '/media/matthew/Seagate_Expansion_Drive/Model_output/' 
path_to_satelite = '/media/matthew/Seagate_Expansion_Drive/Model_output/' 

#The name of the netcdf file
model_dataset = "model_dataset_TD.nc"
satelite_dataset1  = "satelite_dataset_OSTIA.nc" #OSTIA
satelite_dataset2 = "satelite_dataset_ODYSSEA.nc" #ODYSSEA 
satelite_dataset3  = "satelite_dataset_REMSS.nc" #REMSS

#The name of observations for legends and headings 
model_name = "ROMS"
satelite_name1 = "OSTIA"
satelite_name2 = "ODYSSEA"
satelite_name3 = "REMSS"

#The variables to be analysed as written in netcdf files 
model_variable = "temp"
satelite_variable= "analysed_sst"

#The lat and lon of the model netcdf file  
model_lon_name = "lon_rho"
model_lat_name = "lat_rho"

#Path to where the figures should be stored 
savepath= '/media/matthew/Seagate_Expansion_Drive/Model_evaluations_figures/Taylor_diagrams/'

#Name of figure (png file) which shows the position of the area of interest 
savename_fig1 = 'Position_area_of_interest_'+str(position_name)
#Name of figure (png file) which shows the full timeseries  
savename_fig2 = 'Daily_climatology_'+str(position_name)
#Name of figure (png file) which shows the timeseries of the daily climatology for the region of interest
savename_fig3 = 'Daily_anomaly_'+str(position_name)
#Name of figure (png file) which shows the timeseries of the daily anomaly for the region of interest
savename_fig4 = 'Taylor_diagram'+str(position_name)
#Name of figure (png file) which shows the scatter graph of the daily anomaly for the region of interest
savename_fig5 = 'Taylor_diagram_daily_anomaly'+str(position_name)

#Extension
ext = '.png'


# %% Importing packages 

import numpy as np
import xarray as xr 
import pylab as plt
import scipy.stats as stat
import cartopy.crs as ccrs
import cartopy
import skill_metrics as sm

# %% The function used in the script below 

#This function calculates the find a values in a array closest to the specificed element
#The script uses the function to find closest lat/lon points in the dataset so users do not need the exact lat/lon coordinates
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#This function calculates the mean for each timestep of a 3d dimesional matrix, producing a array the same length of the matrix.
#The spript uses this function to average selected subset into arrays for statistical comparisions 
def mean_matrix_3d(matrix):
    matrix_array = np.zeros(len(matrix))
    for x in np.arange(len(matrix[:,1,1])):
        array_tmp = np.nanmean(matrix[x,:,:])
        matrix_array[x] = array_tmp
        
    return matrix_array

# %% Loading netcdf files using xarray 

model_dataset = xr.open_dataset(path_to_model+model_dataset)
satelite_dataset1 = xr.open_dataset(path_to_satelite+satelite_dataset1)
satelite_dataset2 = xr.open_dataset(path_to_satelite+satelite_dataset2)
satelite_dataset3  = xr.open_dataset(path_to_satelite+satelite_dataset3)

#loading lon and lat 
longitude = model_dataset[model_lon_name].values[0,:]
latitude = model_dataset[model_lat_name].values[:,0]

#Loading time
time = satelite_dataset3.time.values

# Time grouped by seasonal climatolgies for the time dimesion of scatter plot 
#time_cat = np.zeros(len(time))

#Finding the exact cordinates within the model and satelite grids for subsetting 
west_co = find_nearest(longitude,west)
east_co = find_nearest(longitude,east)
south_co = find_nearest(latitude,south)
north_co = find_nearest(latitude,north)

#Subsetting the datasets 
model_subset = model_dataset.where((model_dataset.lat_rho > south_co) &
                (model_dataset.lat_rho < north_co) & (model_dataset.lon_rho > west_co) &
                 (model_dataset.lon_rho < east_co), drop=True)

satelite_subset1 = satelite_dataset1.where((satelite_dataset1.lat > south_co) &
                (satelite_dataset1.lat < north_co) & (satelite_dataset1.lon > west_co) &
                 (satelite_dataset1.lon < east_co), drop=True)

satelite_subset2 = satelite_dataset2.where((satelite_dataset2.lat > south_co) &
                (satelite_dataset2.lat < north_co) & (satelite_dataset2.lon > west_co) &
                 (satelite_dataset2.lon < east_co), drop=True)

satelite_subset3 = satelite_dataset3.where((satelite_dataset3.lat > south_co) &
                (satelite_dataset3.lat < north_co) & (satelite_dataset3.lon > west_co) &
                 (satelite_dataset3.lon < east_co), drop=True)


# %% Calculating the daily anomalies and daily climatologies using xr grouby functions

#Calculating the daily anomalies in the model and satelite data using xarray groupby methods 
model_daily_ano = model_subset.groupby("time.dayofyear") - model_subset.groupby("time.dayofyear").mean("time")

sat_daily_ano1 = satelite_subset1.groupby("time.dayofyear") - satelite_subset1.groupby("time.dayofyear").mean("time")
sat_daily_ano2= satelite_subset2.groupby("time.dayofyear") - satelite_subset2.groupby("time.dayofyear").mean("time")
sat_daily_ano3 = satelite_subset3.groupby("time.dayofyear") - satelite_subset3.groupby("time.dayofyear").mean("time")


#Calculating the daily climatology of satelite and model datasets 
model_daily_clim = model_subset.groupby("time.dayofyear").mean("time")

sat_daily_clim1 = satelite_subset1.groupby("time.dayofyear").mean("time")
sat_daily_clim2 = satelite_subset2.groupby("time.dayofyear").mean("time")
sat_daily_clim3 = satelite_subset3.groupby("time.dayofyear").mean("time")

# %% Loading the variables from the dataset

#Converting zeros to nans in the model data (land is denoted as zeros instead of nans which affects averaging)
model_subset[model_variable].values[np.isnan(satelite_subset3[satelite_variable].values)] = np.nan
model_daily_ano[model_variable].values[np.isnan(sat_daily_ano3[satelite_variable].values)] = np.nan
model_daily_clim[model_variable].values[np.isnan(sat_daily_clim3[satelite_variable].values)] = np.nan


# %% #Finding the mean of the subset for each timestep using the function mean_matrix_array. (Converts a 3d matrix into area averaged array)

#Finding the mean for full timseries 
model_array = mean_matrix_3d(model_subset[model_variable].values)
satelite_array1 = mean_matrix_3d(satelite_subset1[satelite_variable].values)
satelite_array2 = mean_matrix_3d(satelite_subset2[satelite_variable].values)
satelite_array3 = mean_matrix_3d(satelite_subset3[satelite_variable].values)

#Finding the mean of the daily anomalies 
model_ano_array = mean_matrix_3d(model_daily_ano[model_variable].values)
satelite_ano1_array = mean_matrix_3d(sat_daily_ano1[satelite_variable].values)
satelite_ano2_array = mean_matrix_3d(sat_daily_ano2[satelite_variable].values)
satelite_ano3_array = mean_matrix_3d(sat_daily_ano3[satelite_variable].values)

#Finding the mean of the daily climatology
model_clim_array = mean_matrix_3d(model_daily_clim[model_variable].values)

satelite_clim1_array = mean_matrix_3d(sat_daily_clim1[satelite_variable].values)
satelite_clim2_array = mean_matrix_3d(sat_daily_clim2[satelite_variable].values)
satelite_clim3_array = mean_matrix_3d(sat_daily_clim3[satelite_variable].values)

# %% Calculating medium

#Median of timeseries  
combined_array = np.stack([model_array, satelite_array1, satelite_array2,satelite_array3]) 
median_array = np.median(combined_array, axis = 0)

#Median of daily anomalies timeseries  
combined_ano_array = np.stack([model_ano_array, satelite_ano1_array, satelite_ano2_array,satelite_ano3_array]) 
median_ano_array = np.median(combined_ano_array, axis = 0)

#Median of daily climatology timeseries  
combined_clim_array = np.stack([model_clim_array, satelite_clim1_array, satelite_clim2_array,satelite_clim3_array]) 
median_clim_array = np.median(combined_clim_array, axis = 0)

# %% Taylor iagram stats 

crmsd_model = sm.centered_rms_dev(model_array, median_array);
crmsd1 = sm.centered_rms_dev(satelite_array1, median_array)
crmsd2 = sm.centered_rms_dev(satelite_array2, median_array)
crmsd3 = sm.centered_rms_dev(satelite_array3, median_array)

r_model = stat.pearsonr(model_array, median_array)[0]
r1 = stat.pearsonr(satelite_array1, median_array)[0]
r2 = stat.pearsonr(satelite_array2, median_array)[0]
r3 = stat.pearsonr(satelite_array3, median_array)[0]

std_model = stat.tstd(model_array)
std1 = stat.tstd(satelite_array1)
std2 = stat.tstd(satelite_array2)
std3 = stat.tstd(satelite_array3)


# (e.g. taylor_stats1[1:]) are those for the predicted series.
taylor_stats1 = sm.taylor_statistics(model_array,median_array,'data')
taylor_stats2 = sm.taylor_statistics(satelite_array1,median_array,'data')
taylor_stats3 = sm.taylor_statistics(satelite_array2,median_array,'data')
taylor_stats4 = sm.taylor_statistics(satelite_array3,median_array,'data')

    
# Store statistics in arrays
sdev = np.array([taylor_stats1['sdev'][0], taylor_stats1['sdev'][1], 
                     taylor_stats2['sdev'][1], taylor_stats3['sdev'][1],taylor_stats4['sdev'][1]])
crmsd = np.array([taylor_stats1['crmsd'][0], taylor_stats1['crmsd'][1], 
                      taylor_stats2['crmsd'][1], taylor_stats3['crmsd'][1],taylor_stats4['crmsd'][1]])
ccoef = np.array([taylor_stats1['ccoef'][0], taylor_stats1['ccoef'][1], 
                      taylor_stats2['ccoef'][1], taylor_stats3['ccoef'][1], taylor_stats4['ccoef'][1]])

# (e.g. taylor_stats1[1:]) are those for the predicted series.
taylor_stats_ano1 = sm.taylor_statistics(model_ano_array,median_ano_array,'data')
taylor_stats_ano2 = sm.taylor_statistics(satelite_ano1_array,median_ano_array,'data')
taylor_stats_ano3 = sm.taylor_statistics(satelite_ano2_array,median_ano_array,'data')
taylor_stats_ano4 = sm.taylor_statistics(satelite_ano3_array,median_ano_array,'data')

# Store statistics in arrays
sdev_ano = np.array([taylor_stats_ano1['sdev'][0], taylor_stats_ano1['sdev'][1], 
                     taylor_stats_ano2['sdev'][1], taylor_stats_ano3['sdev'][1],taylor_stats_ano4['sdev'][1]])
crmsd_ano = np.array([taylor_stats_ano1['crmsd'][0], taylor_stats_ano1['crmsd'][1], 
                      taylor_stats_ano2['crmsd'][1], taylor_stats_ano3['crmsd'][1],taylor_stats_ano4['crmsd'][1]])
ccoef_ano = np.array([taylor_stats_ano1['ccoef'][0], taylor_stats_ano1['ccoef'][1], 
                      taylor_stats_ano2['ccoef'][1], taylor_stats_ano3['ccoef'][1], taylor_stats_ano4['ccoef'][1]])


# %% Selecting subset area

#Finding the correct indices of the above cooridates
west_index = np.where(longitude == find_nearest(longitude,west))[0][0]
east_index = np.where(longitude == find_nearest(longitude,east))[0][0]
south_index = np.where(latitude == find_nearest(latitude,south))[0][0]
north_index = np.where(latitude == find_nearest(latitude,north))[0][0]

# %% Plotting postion of the selected area
    
fig = plt.figure(figsize=(10, 5),facecolor='white')
ax = plt.axes(projection=ccrs.PlateCarree())
cs = plt.contourf(longitude,latitude,model_dataset[model_variable].values[1,:,:],60, transform=ccrs.PlateCarree());
ln0 = plt.plot([longitude[west_index], longitude[east_index]], [latitude[south_index], latitude[south_index]], '-k')
ln1 = plt.plot([longitude[west_index], longitude[east_index]], [latitude[north_index], latitude[north_index]], '-k')
ln2 = plt.plot([longitude[east_index], longitude[east_index]], [latitude[north_index], latitude[south_index]], '-k')
ln3 = plt.plot([longitude[west_index], longitude[west_index]], [latitude[north_index], latitude[south_index]], '-k')
ax.coastlines()
ax.add_feature(cartopy.feature.LAND, zorder=0)

plt.savefig(savepath+savename_fig1+ext)


# %% Ploting timeseries daily clim

fig,ax = plt.subplots()
line_sat1 = plt.plot(satelite_clim1_array,label= satelite_name1);
line_sat2 = plt.plot(satelite_clim2_array,label= satelite_name2);
line_sat3 = plt.plot(satelite_clim3_array,label= satelite_name3);
line_roms = plt.plot(model_clim_array,label = model_name);
plt.xlabel('Day of year')
plt.ylabel('SST ($^\circ$C)')
plt.title(str(position_name))
plt.legend()

plt.savefig(savepath+savename_fig2+ext,bbox_inches='tight', pad_inches=0.2)


# %% Ploting timeseries of daily anomaly

fig,ax = plt.subplots()
line_sat1 = plt.plot(time,satelite_ano1_array,label = satelite_name1); 
line_sat2 = plt.plot(time,satelite_ano2_array,label = satelite_name2);
line_sat3 = plt.plot(time,satelite_ano3_array,label = satelite_name3);
line_roms = plt.plot(time,model_ano_array,label = model_name);
plt.xlabel('Date in years')
plt.ylabel('SST ($^\circ$C)')
plt.title(str(position_name))
plt.legend()

plt.savefig(savepath+savename_fig3+ext,bbox_inches='tight', pad_inches=0.2)

# %% Taylor diagram 

label = ['Non-Dimensional Observation', model_name, satelite_name1 , satelite_name2, satelite_name3]

fig,ax = plt.subplots()
plt.title(str(position_name))
sm.taylor_diagram(sdev,crmsd,ccoef,markerLabel = label,markerLegend = 'on', styleOBS = '-', colOBS = 'k', markerobs = 'o',titleOBS = 'Median')

plt.savefig(savepath+savename_fig4)

# %% Taylor diagram daily anomalies

label = ['Non-Dimensional Observation', model_name, satelite_name1 , satelite_name2, satelite_name3]

crmsd_ano = np.round(crmsd_ano,2)

fig,ax = plt.subplots()
plt.title(str(position_name)+"_daily_anomalies")
sm.taylor_diagram(sdev_ano,crmsd_ano,ccoef_ano, rmsLabelFormat= '0:.2f' , markerLabel = label,markerLegend = 'on', styleOBS = '-', colOBS = 'k', markerobs = 'o',titleOBS = 'Median')

plt.savefig(savepath+savename_fig5)

