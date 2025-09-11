# -*- coding: utf-8 -*-
"""
Created on 2023-07-14

@author: ivespe

Intro script for Exercise 2 ("Load analysis to evaluate the need for flexibility") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""

# %% Dependencies

import pandapower as pp
import pandapower.plotting as pp_plotting
import pandas as pd
import os
import load_scenarios as ls
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         = 'CINELDI_MV_reference_system_v_2023-03-06/'

filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')

# Subset of load buses to consider in the grid area, considering the area at the end of the main radial in the grid
bus_i_subset = [90, 91, 92, 96]

# Assumed power flow limit in MW that limit the load demand in the grid area (through line 85-86)
P_lim = 0.637 

# Maximum load demand of new load being added to the system
P_max_new = 0.4

# Which time series from the load data set that should represent the new load
i_time_series_new_load = 90


# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)

# %% Extract hourly load time series for a full year for all the load points in the CINELDI reference system
# (this code is made available for solving task 3)

load_profiles = lp.load_profiles(filename_load_data_fullpath)


# Get all the days of the year
repr_days = list(range(1,366))

# Get normalized load profiles for representative days mapped to buses of the CINELDI reference grid;
# the column index is the bus number (1-indexed) and the row index is the hour of the year (0-indexed)
profiles_mapped = load_profiles.map_rel_load_profiles(filename_load_mapping_fullpath,repr_days)

# Retrieve normalized load time series for new load to be added to the area
new_load_profiles = load_profiles.get_profile_days(repr_days)
new_load_time_series = new_load_profiles[i_time_series_new_load]*P_max_new

# Calculate load time series in units MW (or, equivalently, MWh/h) by scaling the normalized load time series by the
# maximum load value for each of the load points in the grid data set (in units MW); the column index is the bus number
# (1-indexed) and the row index is the hour of the year (0-indexed)
load_time_series_mapped = profiles_mapped.mul(net.load['p_mw'])
# %%

## plot load time series for the load points in the grid area
plt.figure(figsize=(10,6))
load_sum = np.zeros(8760)
for bus_i in bus_i_subset:
    plt.plot(load_time_series_mapped[bus_i], label='Bus %d' % bus_i)
    load_sum += load_time_series_mapped[bus_i].to_numpy()
plt.plot(new_load_time_series, '--', label='New load (bus %d)' % i_time_series_new_load)
load_sum += new_load_time_series
plt.plot(load_sum, 'k', label='Total load in area')


window_size = 10*24
window = np.ones(window_size) / window_size

# Pad data med refleksjon for å unngå nuller på kantene
pad_width = window_size // 2
load_sum_padded = np.pad(load_sum, pad_width, mode='reflect')

# Konvolver og klipp til original lengde
load_sum_smooth = np.convolve(load_sum_padded, window, mode='valid')

plt.plot(load_sum_smooth, color='cyan', label='Total load in area (smoothed)')


#plot power flow limit
plt.plot(np.ones(8760)*P_lim, 'r--', label='Power flow limit')
plt.xlabel('Hour of the year')
plt.ylabel('Load demand (MW)')
plt.title('Load time series for load points in the grid area')
plt.legend()
plt.grid()
#plt.show()



# calculate hours where load exceeds the power flow limit
n_hours_exceeding_limit = np.sum(load_sum > P_lim)
print('Number of hours where load exceeds the power flow limit: %d' % n_hours_exceeding_limit)

# calcurate energy not served due to the power flow limit (in MWh)
energy_not_served = np.sum(load_sum[load_sum > P_lim] - P_lim)
print('Energy not served due to the power flow limit: %.2f MWh' % energy_not_served)


