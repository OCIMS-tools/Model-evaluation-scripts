#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:36:26 2020

@author: matty
"""
"""
This script is used to proccess the model and observation data. The script is designed to
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
    - Two nedcdf files containing the model output and observations with identical 
      grid size and time sampling.
    
REQUIRMENTS:
    -The netcdf files need to be formatted so that the variabled used is a 3 dimesional matrix of (time, longitude, latitude) 
    -The netcdf files must be sampled at a daily frequency 

The inputs to the script should be changed where necessary is the section below called USER INPUTS
"""
# %% USER INPUTS

#Adding the path to the model data 
path_to_model_data = 'E:\\Model_output\\'
path_to_satelite_data  =  'E:\\SST_downloads\\REMSS_MW_IR\\'

#Adding path to save the model netcdf file 
save_path_model = 'E:\\Model_evaluations\\'
fname_model = "model_dataset" +".nc"

#Adding path to save the satelite netcdf file 
save_path_satelite = 'E:\\Model_evaluations\\'
fname_satelite = "satelite_dataset" +".nc"

#The variables to be analysed as written in netcdf files 
model_variable = "temp"
satelite_variable= "analysed_sst"

#Wildcard name (Shortcut used to identify files e.g. all the netcdf files with the letter Y)
wildcard = "*Y*.nc"

#The longitude and latitude dimesions of the model output of named in netcdf file
lat_model = 'lat_rho'
lon_model = 'lon_rho'

#The longitude and latitude dimesions of the observations of named in netcdf file
lat_satelite = 'lat'
lon_satelite = 'lon'

#The depth dimesion of the model data used to extract surface level data 
depth_dim = 's_rho'

#Time since initialization of the model
int_time = 'seconds since 1990-01-01 00:00:00'

# %% Import packages 

import numpy as np
import xarray as xr 
from glob import glob
import netCDF4 as nc
import pandas as pd 
from natsort import natsorted
from datetime import date, timedelta

# %% Loading model data 

#Getting a directory of file names natsorted is include to ensure files are in the correct order 
modelFiles = natsorted(glob(path_to_model_data+wildcard))

#Loading reference dataset to find surface level index
ds = xr.open_dataset(modelFiles[0])

#Finding surface level of model output
level = ds[depth_dim].size-1

# Importing data through loop as seprate dataframe (sel only the surface and temperature)
list_of_arrays = []
for file in modelFiles:
    ds = xr.open_dataset(file)
    ds = ds[model_variable].isel(s_rho=level)
    list_of_arrays.append(ds)

#Concatenating mutliple dataset to single dataset with a single variable and cooridates  
model_ds = xr.concat(list_of_arrays[0:len(list_of_arrays)],dim='time')

# Converting the time of dataset from seconds since intilation to datetime
a=model_ds.coords['time'].values
local_time = nc.num2date(a, units=int_time) 
## then copy the decoded back into the dataset
lt=xr.DataArray(local_time,dims='time')
#ds_ROMS['time'].values=lt.values
model_ds = model_ds.assign_coords(time=('time',lt))

#Converting time from cftime.DatetimeGregorian to datetime
datetimeindex = model_ds.indexes['time'].to_datetimeindex()
model_ds['time'] = datetimeindex

# loading lon and lat 
lon_model = model_ds[lon_model].values[0,:]
lat_model = model_ds[lat_model].values[:,0]

#Loading time variable from model  
model_time = model_ds.time.values

# %% Loading satelite data  

# Domain of satelite data from the limits of the model data 
west=lon_model.min()
east=lon_model.max()
south=lat_model.min()
north=lat_model.max()

#Timesubset of the model data 
start_time = model_time[0]
end_time = model_time[len(model_time)-1]

# Selecting files in folder
sstFiles = natsorted(glob(path_to_satelite_data+'*.nc'))

# Importing data through loop saving memory 
list_of_arrays = []
for file in sstFiles:#[2769:6024]:
    ds = xr.open_dataset(file)
    ds = ds['analysed_sst'].sel(lat=slice(south, north), lon=slice(west, east))
    list_of_arrays.append(ds)

#Concatenating the data in one dataset 
sat_ds = xr.concat(list_of_arrays[0:len(list_of_arrays)],dim='time')

#Slicing the dataset to the correct time 
sat_ds =  sat_ds.sel(time=slice(start_time,end_time))

# loading lon and lat 
lon_sst = sat_ds[lon_satelite].values
lat_sst = sat_ds[lat_satelite].values

#Loading time from sst 
time_sst = sat_ds.time.values 

#Kelvin conversion 
sat_ds = sat_ds-273.15

# %% Regrid satelite data to the same grid of the model data 

sat_regrid_ds = sat_ds.interp(lon=lon_model, lat=lat_model)

# %% Check if the time dimesions are the same length if not,identifing and
# dropping the missing days for the mode output 

#Loading time variable from the satelite day    
sat_time = sat_ds.time.values

if len(sat_time) != len(model_time):
    #Converting satelite time variable to readable variables (converting from datetime to ordinal)
    sat_time_list = []
    for i in np.arange(len(sat_time)):
        time_ordinal_tmp = pd.to_datetime(str(sat_time[i])[0:10]).toordinal()
        time_tmp = date.fromordinal(time_ordinal_tmp)
        sat_time_list.append(time_tmp)  

    #d = sat_time_list[:]
    #Finding the missing values 
    date_set = set(sat_time_list[0] + timedelta(x) for x in range((sat_time_list[-1] - sat_time_list[0]).days))
    missing = sorted(date_set - set(sat_time_list))

    #Converting ROMS time variable to readable variables (converting from datetime to ordinal)
    model_time_list = []
    model_time_list_ordinal = []

    for i in np.arange(len(model_time)-1):
        time_ordinal_tmp = pd.to_datetime(str(model_time[i])[0:10]).toordinal()
        time_tmp = date.fromordinal(time_ordinal_tmp)
        model_time_list.append(time_tmp)
        model_time_list_ordinal.append(time_ordinal_tmp)

    # Finding when the missing day 
    missing_index = np.where(np.asarray(model_time_list)==missing[0])
    
    #Dropping the missing days in the observations from the model dataset
    model_ds = model_ds.drop(model_time[missing_index], dim='time') 

# %% Remove missing days from the model dataset to make the dataset time dimesions the same        
# The files are only saved if the time dimesions match! 

satelite_time_check = sat_regrid_ds.time.values         
model_time_check = model_ds.time.values

if len(satelite_time_check) == len(model_time_check):
    # Saving the netcdf file
    model_ds.to_netcdf(save_path_model+fname_model)
    
    sat_regrid_ds.to_netcdf(save_path_satelite+fname_satelite)

else:
    print("Length of the time dimesion does not match! File not saved")
       
        

