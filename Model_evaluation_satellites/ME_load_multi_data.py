#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 15:28:53 2020

@author: matthew
"""
"""
This script is used to proccess the model and multiple observation data. The script is designed to
concatnate mutliple netcdf files to produce single netcdf files for the model and observerations
respectively. The script regrids the observations to fit on the same grid as the model output.
The script includes a function to dectect whether the datasets have the same time dimesions 
and remove missing day from the model output if the satelite dataset includes missing days.
This is vital as analysis requires the time sampling to be identical the script will not
write the netcdf files if the time dimesion of the model and observations do not match.

INPUT:
    - Netcdf files containing the model output and observations
    - the path to the folders containing the netcdf files 
    - the path to folder where the output netcdf files should be stored
    - the name of the netcdf file to be stored 
    - a wildcard name that can be used to identify all the input files in a paricular folder   
    - the name of the variables to be compared (! as written in netcdf file !)
    - the name of depth dimesion in the model output files (! as written in netcdf file !)
    - the name of the longitude and latitiude dimensions in the model and observation  files (! as written in netcdf file !)
    - the time of the model initialization

OUTPUT:
    - four nedcdf files containing the model output and observations with identical 
      grid size and time sampling.
    
REQUIRMENTS:
    -The netcdf files need to be formatted so that the variabled used is a 3 dimesional matrix of (time, longitude, latitude) 
    -The netcdf files must be sampled at a daily frequency 

The inputs to the script should be changed where necessary is the section below called USER INPUTS

"""

# %% Import packages 

import numpy as np
import xarray as xr 
from glob import glob
import netCDF4 as nc
import pandas as pd 
from natsort import natsorted
from datetime import date, timedelta


# %% Setting paths 

#Adding the path to the model data 
path_to_model_data = '/media/matthew/Seagate_Expansion_Drive/Model_output/' 

#Adding the path to satelite data 
path_to_satelite_data1 =  '/media/matthew/Data/High_res_SST/OSTIA_NRT/'
satelite_name_1 = "OSTIA"

#Adding the path to satelite data 
path_to_satelite_data2 =  '/media/matthew/Elements/SST_data/Medium_res_SST/ODYSSEA_MUR/'
satelite_name_2 = "ODYSSEA"

#Adding the path to satelite data 
path_to_satelite_data3  =  '/media/matthew/Data/Medium_res_SST/REMSS_MW_IR/'
satelite_name_3 = "REMSS"

#The variables to be analysed as written in netcdf files 
model_variable = "temp"
satelite_variable= "analysed_sst"

#The depth dimesion of the model data used to extract surface level data 
depth_dim = 's_rho'

#The longitude and latitude dimesions of the model output of named in netcdf file
lat_model = 'lat_rho'
lon_model = 'lon_rho'

#The longitude and latitude dimesions of the observations of named in netcdf file
lat_satelite = 'lat'
lon_satelite = 'lon'

#Adding path to save the model netcdf file 
save_path_model = '/media/matthew/Seagate_Expansion_Drive/Model_output/' 
fname_model = "model_dataset_TD" +".nc"

#Adding path to save the satelite netcdf file 
save_path_satelite = '/media/matthew/Seagate_Expansion_Drive/Model_output/' 
fname_satelite1 = "satelite_dataset_OSTIA" +".nc"
fname_satelite2 = "satelite_dataset_ODYSSEA" +".nc"
fname_satelite3 = "satelite_dataset_REMSS" +".nc"

# %% Loading model data 


#Getting a directory of file names natsorted is include to ensure files are in the correct order 
modelFiles = natsorted(glob(path_to_model_data+"*Y*.nc"))

#Loading reference dataset to find surface level index
ds = xr.open_dataset(modelFiles[0])

#Finding surface level of model output
level = ds[depth_dim].size-1

# Importing data through loop as seprate dataframe (sel only the surface and temperature)
list_of_arrays = []
for file in modelFiles:
    ds = xr.open_dataset(file)
    #Smaller domain
    ds = ds[model_variable].isel(s_rho=level)
    #ds = ds[vname].isel(s_rho=level)
    list_of_arrays.append(ds)

#Concatenating mutliple dataset to single dataset with a single variable and cooridates  
model_ds = xr.concat(list_of_arrays[0:len(list_of_arrays)],dim='time')

# Converting the time of dataset from seconds since intilation to datetime
a=model_ds.coords['time'].values
local_time = nc.num2date(a, units='seconds since 1990-01-01 00:00:00') 
## then copy the decoded back into the dataset
lt=xr.DataArray(local_time,dims='time')
#ds_ROMS['time'].values=lt.values
model_ds = model_ds.assign_coords(time=('time',lt))

#Converting time from cftime.DatetimeGregorian to datetime
datetimeindex = model_ds.indexes['time'].to_datetimeindex()
model_ds['time'] = datetimeindex

# loading lon and lat 
lon_roms = model_ds.lon_rho.values[0,:]
lat_roms = model_ds.lat_rho.values[:,0]

#Time subsetting 
start_time = np.datetime64('2011-01-01T12:00:00')
end_time = np.datetime64('2018-12-30T12:00:00')

#Subset model data by time
model_ds =  model_ds.sel(time=slice(start_time,end_time))

#Loading time variable from model  
model_time = model_ds.time.values

# %% Loading ostia satelite data  

# Domain of satelite data from the limits of the model data 
west=lon_roms.min()
east=lon_roms.max()
south=lat_roms.min()
north=lat_roms.max()

#Timesubset of the model data 
start_time = model_time[0]
end_time = model_time[len(model_time)-1]


# Selecting files in folder
sstFiles = natsorted(glob(path_to_satelite_data1 +'*.nc'))

# Importing data through loop saving memory 
list_of_arrays = []
for file in sstFiles:
    ds = xr.open_dataset(file)
    ds = ds[satelite_variable].sel(lat=slice(south, north), lon=slice(west, east))
    list_of_arrays.append(ds)

sat_ds = xr.concat(list_of_arrays[0:len(list_of_arrays)],dim='time')

#Slicing the dataset to the correct time 
sat_ds =  sat_ds.sel(time=slice(start_time,end_time))

# loading lon and lat 
lon_sst = sat_ds.lon.values
lat_sst = sat_ds.lat.values

#Loading time from sst 
time_sst1 = sat_ds.time.values 

#Kelvin conversion 
sat_ds = sat_ds-273.15

#Regrid
sat_regrid_ds_1 = sat_ds.interp(lon=lon_roms, lat=lat_roms)

#Remove ungridded dataset
del sat_ds

# %% Loading satelite data  

# Domain of satelite data from the limits of the model data 
west=lon_roms.min()
east=lon_roms.max()
south=lat_roms.min()
north=lat_roms.max()

#Timesubset of the model data 
start_time = np.datetime64('2011-01-01T00:00:00')
end_time = np.datetime64('2018-11-30T00:00:00')

# Selecting files in folder
sstFiles = natsorted(glob(path_to_satelite_data2 +'*.nc'))

# Importing data through loop saving memory 
list_of_arrays = []
for file in sstFiles:
    ds = xr.open_dataset(file)
    ds = ds[satelite_variable].sel(lat=slice(south, north), lon=slice(west, east))
    list_of_arrays.append(ds)

sat_ds = xr.concat(list_of_arrays[0:len(list_of_arrays)],dim='time')

#Slicing the dataset to the correct time 
sat_ds =  sat_ds.sel(time=slice(start_time,end_time))

# loading lon and lat 
lon_sst = sat_ds.lon.values
lat_sst = sat_ds.lat.values

#Loading time from sst 
time_sst2 = sat_ds.time.values 

#Kelvin conversion 
sat_ds = sat_ds-273.15

#Regrid
sat_regrid_ds_2 = sat_ds.interp(lon=lon_roms, lat=lat_roms)

#Remove ungridded dataset
sat_ds


# %% Loading satelite data  

# Domain of satelite data from the limits of the model data 
west=lon_roms.min()
east=lon_roms.max()
south=lat_roms.min()
north=lat_roms.max()

#Timesubset of the model data 
start_time = model_time[0]
end_time = model_time[len(model_time)-1]

# Selecting files in folder
sstFiles = natsorted(glob(path_to_satelite_data3 +'*.nc'))

# Importing data through loop saving memory 
list_of_arrays = []
for file in sstFiles:
    ds = xr.open_dataset(file)
    ds = ds[satelite_variable].sel(lat=slice(south, north), lon=slice(west, east))
    list_of_arrays.append(ds)

sat_ds = xr.concat(list_of_arrays[0:len(list_of_arrays)],dim='time')

#Slicing the dataset to the correct time 
sat_ds =  sat_ds.sel(time=slice(start_time,end_time))

# loading lon and lat 
lon_sst = sat_ds.lon.values
lat_sst = sat_ds.lat.values

#Loading time from sst 
time_sst3 = sat_ds.time.values 

#Kelvin conversion 
sat_ds = sat_ds-273.15

#Regrid
sat_regrid_ds_3 = sat_ds.interp(lon=lon_roms, lat=lat_roms)

#Remove ungridded dataset
del sat_ds

# %% Find the shortess period of time (most missing days)

def missing_days(sat_time):
    #Convert time to list with time in correct format
    sat_time_list = []
    for i in np.arange(len(sat_time)):
        time_ordinal_tmp = pd.to_datetime(str(sat_time[i])[0:10]).toordinal()
        time_tmp = date.fromordinal(time_ordinal_tmp)
        sat_time_list.append(time_tmp)  

    #Finding the missing values 
    date_set = set(sat_time_list[0] + timedelta(x) for x in range((sat_time_list[-1] - sat_time_list[0]).days))
    missing = sorted(date_set - set(sat_time_list))
    return missing, sat_time_list

def missing_index(missing_con,time_list):
    index=[]
    for x in np.arange(len(missing_con)):
        tmp_index = np.where(np.asarray(time_list)==missing_con[x])[0]
        if len(tmp_index>0):
            index.append(tmp_index[0])
    return index

#Find missing days in SST dataset
missing_days1, time_list1 = missing_days(time_sst1)
missing_days2, time_list2 = missing_days(time_sst2)
missing_days3, time_list3 = missing_days(time_sst3)

#Combine missing days in one list 
missing_con = missing_days1 + missing_days2 + missing_days3

#Finding missing index
index1 = missing_index(missing_con,time_list1)
index2 = missing_index(missing_con,time_list2)
index3 = missing_index(missing_con,time_list3)

#Converting ROMS time variable to readable variables (converting from datetime to ordinal)
model_time_list = []
model_time_list_ordinal = []

for i in np.arange(len(model_time)-1):
    time_ordinal_tmp = pd.to_datetime(str(model_time[i])[0:10]).toordinal()
    time_tmp = date.fromordinal(time_ordinal_tmp)
    model_time_list.append(time_tmp)
    model_time_list_ordinal.append(time_ordinal_tmp)

# Finding when the missing day     
model_index = missing_index(missing_con,model_time_list)

#Dropping the missing days in the observations from the model dataset
model_ds = model_ds.drop(model_time[model_index], dim='time') 

sat_regrid_ds_1 = sat_regrid_ds_1.drop(time_sst1[index1], dim='time') 
sat_regrid_ds_2 = sat_regrid_ds_2.drop(time_sst2[index2], dim='time') 
sat_regrid_ds_3 = sat_regrid_ds_3.drop(time_sst3[index3], dim='time') 

    

# %% Saving the netcdf file

model_ds.to_netcdf(save_path_model+fname_model)

sat_regrid_ds_1.to_netcdf(save_path_satelite+fname_satelite1)

sat_regrid_ds_2.to_netcdf(save_path_satelite+fname_satelite2)

sat_regrid_ds_3.to_netcdf(save_path_satelite+fname_satelite3)