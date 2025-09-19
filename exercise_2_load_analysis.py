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
import networkx as nx
from matplotlib import gridspec

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



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
repr_days = list(range(1, 366))

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

def plot_voltage_profile(net):
    """
    Plots the voltage profile for the buses in bus_i_subset.
    """
    pp.runpp(net,init='results',algorithm='bfsw')

    base_vm = net.res_bus["vm_pu"].copy()

    # 3) Build graph and find the **longest feeder** (leaf path from slack)
    graph = nx.Graph()
    graph.add_nodes_from(net.bus.index)
    for _, ln in net.line.iterrows():
        graph.add_edge(int(ln.from_bus), int(ln.to_bus))
    for _, tr in net.trafo.iterrows():
        graph.add_edge(int(tr.hv_bus), int(tr.lv_bus))

    slack_bus = int(net.ext_grid.bus.values[0])
    leaf_buses = [n for n in graph.nodes if graph.degree(n) == 1 and n != slack_bus]

    feeders = []
    for end_bus in leaf_buses:
        path = nx.shortest_path(graph, source=slack_bus, target=end_bus)
        feeders.append((len(path), path, net.bus.loc[end_bus, "name"]))
    feeders.sort(key=lambda t: t[0], reverse=True)
    path = feeders[0][1]
    end_name = feeders[0][2]

    # 4) Figure and layout (GridSpec: plot on top, controls below)
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4.5, 1.2], figure=fig)
    ax = fig.add_subplot(gs[0])

    # Curves
    x = np.arange(len(path))
    ax.plot(x, base_vm[path], "o-", label="Base case", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels([net.bus.loc[bi, "name"] for bi in path], rotation=45)
    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage [p.u.]")
    ax.set_title(f"Voltage profile along longest feeder to {end_name}")
    ax.grid()
    ax.legend()
    ax.set_ylim(0.9, 1.0)
    plt.tight_layout()
    plt.show()

def plot_area_load_time_series(load_time_series_mapped, bus_i_subset, new_load_time_series, P_lim, i_time_series_new_load):
    """
    Plotter lasttidserier for utvalgte busser i området + ny last, total last og glattet total last.
    
    Parametre
    ---------
    load_time_series_mapped : DataFrame
        Lasttidserier (MW) for alle busser i nettet.
    bus_i_subset : list
        Liste med buss-IDer i området som skal plottes (f.eks. [90, 91, 92, 96]).
    new_load_time_series : array-like
        Tidsserie for den nye lasten i området (MW).
    P_lim : float
        Effektgrense for området (MW).
    i_time_series_new_load : int
        ID til tidsserien for den nye lasten (for etikett i plottet).
    
    Returnerer
    ----------
    load_sum : ndarray
        Summen av lastene i området (inkl. ny last), per time.
    load_sum_smooth : ndarray
        Glattet tidsserie for total last.
    """
    plt.figure(figsize=(10,6))
    load_sum = np.zeros(8760)
    
    # Plott eksisterende laster
    for bus_i in bus_i_subset:
        plt.plot(load_time_series_mapped[bus_i], label=f'Bus {bus_i}')
        load_sum += load_time_series_mapped[bus_i].to_numpy()
    
    # Plott ny last
    plt.plot(new_load_time_series, '--', label=f'New load (bus {i_time_series_new_load})')
    load_sum += new_load_time_series
    
    # Total last
    plt.plot(load_sum, 'k', label='Total load in area')
    
    # Glatt total last
    window_size = 10*24
    window = np.ones(window_size) / window_size
    pad_width = window_size // 2
    load_sum_padded = np.pad(load_sum, pad_width, mode='reflect')
    load_sum_smooth = np.convolve(load_sum_padded, window, mode='valid')
    
    plt.plot(load_sum_smooth, color='cyan', label='Total load in area (smoothed)')
    
    # Effektgrense
    plt.plot(np.ones(8760)*P_lim, 'r--', label='Power flow limit')
    plt.xlabel('Hour of the year')
    plt.ylabel('Load demand (MW)')
    plt.title('Load time series for load points in the grid area')
    plt.legend()
    plt.grid()
    plt.show()
    
    return load_sum, load_sum_smooth

def make_load_table(net, bus_i_subset):
    """
    Lager tabell over lastverdiene på utvalgte busser og summen.
    
    Parameters
    ----------
    net : pandapowerNet
        Nettmodellen.
    bus_i_subset : list
        Liste med busser som skal inkluderes (f.eks. [90, 91, 92, 96]).

    Returns
    -------
    table : pandas.DataFrame
        Tabell med lastverdier (MW) per buss og summen.
    """
    # Hent ut laster for bussene i området
    area_loads = net.load.loc[net.load.bus.isin(bus_i_subset), ["bus", "p_mw"]].copy()
    
    # Legg til bussnavn (fra net.bus)
    area_loads["bus_name"] = area_loads["bus"].map(lambda b: net.bus.loc[b, "name"])
    
    # Sett indeks til buss
    table = area_loads.set_index("bus_name")[["p_mw"]]
    table.rename(columns={"p_mw": "Load (MW)"}, inplace=True)
    
    # Legg til en sumrad
    total = pd.DataFrame({"Load (MW)": [table["Load (MW)"].sum()]}, index=["Sum"])
    table = pd.concat([table, total])
    
    return table

def make_load_profile(load_sum, P_lim):
    """
    Sums up load for each time step and sort it in descending order. 
    Plots the load profile in MW as a function of the number of hours in the year.
    """
    load_sum_sorted = np.sort(load_sum)[::-1]
    hours = np.arange(1, len(load_sum_sorted)+1)
    
    plt.figure(figsize=(10,6))
    plt.plot(hours, load_sum_sorted, label='Load profile')
    plt.plot(hours, np.ones(len(hours))*P_lim, 'r--', label='Power flow limit')
    plt.xlabel('Number of hours in the year')
    plt.ylabel('Load demand (MW)')
    plt.title('Load profile for load points in the grid area')
    plt.legend()
    plt.grid()
    plt.show()
    
    return load_sum_sorted

def analyze_voltage_capacity(net, bus_i_subset, scaling_range=(1, 2), steps=11, v_limit=0.95):
    """
    Analyserer spenningsnivået i et område når lasten økes.
    
    Parametre
    ---------
    net : pandapowerNet
        Nettmodellen.
    bus_i_subset : list
        Liste med buss-IDer i området.
    scaling_range : tuple
        (min, max) for skaleringsfaktorene. Default (1,2).
    steps : int
        Antall punkter i skaleringen. Default 11.
    v_limit : float
        Nedre spenningsgrense (p.u.). Default 0.95.
    
    Returnerer
    ----------
    results_df : pandas.DataFrame
        Tabell med aggregerte laster og laveste spenninger.
    """
    scaling_factors = np.linspace(scaling_range[0], scaling_range[1], steps)
    lowest_voltages = []
    agg_loads = []

    # --- hent basislaster
    base_loads = net.load.loc[net.load.bus.isin(bus_i_subset), ["bus", "p_mw"]].copy()
    base_loads["bus_name"] = base_loads["bus"].map(lambda b: net.bus.loc[b, "name"])

    print("=== Basislaster i området ===")
    print(base_loads[["bus_name", "p_mw"]])
    print("Sum (MW):", base_loads["p_mw"].sum())

    # --- skaler laster og kjør power flow
    for sf in scaling_factors:
        # skaler opp
        for load_idx in net.load.index[net.load.bus.isin(bus_i_subset)]:
            net.load.at[load_idx, "p_mw"] = base_loads.loc[base_loads.bus == net.load.at[load_idx, "bus"], "p_mw"].values[0] * sf

        # kjør kraftflyt
        pp.runpp(net)
        vmin = net.res_bus.loc[bus_i_subset, "vm_pu"].min()
        p_sum = net.load.loc[net.load.bus.isin(bus_i_subset), "p_mw"].sum()

        lowest_voltages.append(vmin)
        agg_loads.append(p_sum)

    # --- tabell med resultater
    results_df = pd.DataFrame({
        "Scaling factor": scaling_factors,
        "Aggregated load (MW)": agg_loads,
        "Lowest voltage (p.u.)": lowest_voltages
    })

    # --- plott
    plt.figure(figsize=(8,5))
    plt.plot(results_df["Aggregated load (MW)"], results_df["Lowest voltage (p.u.)"], "o-", label="Laveste spenning")
    plt.axhline(v_limit, color="r", linestyle="--", label=f"Spenningsgrense {v_limit} p.u.")
    
    # marker skaleringsfaktor ved punktene
    for x, y, sf in zip(results_df["Aggregated load (MW)"], results_df["Lowest voltage (p.u.)"], results_df["Scaling factor"]):
        plt.annotate(f"{sf:.2f}", (x, y), textcoords="offset points", xytext=(5,5), fontsize=8, color="black")
    
    plt.xlabel("Aggregert last i området [MW]")
    plt.ylabel("Laveste spenning [p.u.]")
    plt.title("Spenning som funksjon av aggregert last")
    plt.grid()
    plt.legend()
    plt.show()

    return results_df


#plot_voltage_profile(net)  
table = make_load_table(net, bus_i_subset)
print(table)
#load_sum, load_sum_smooth = plot_area_load_time_series(load_time_series_mapped, bus_i_subset, new_load_time_series, P_lim, i_time_series_new_load)
#load_sum_sorted = make_load_profile(load_sum, P_lim)
#analyze_voltage_capacity(net, bus_i_subset)


## calculate hours where load exceeds the power flow limit
#n_hours_exceeding_limit = np.sum(load_sum > P_lim)
#print('Number of hours where load exceeds the power flow limit: %d' % n_hours_exceeding_limit)
#
## calcurate energy not served due to the power flow limit (in MWh)
#energy_not_served = np.sum(load_sum[load_sum > P_lim] - P_lim)
#print('Energy not served due to the power flow limit: %.2f MWh' % energy_not_served)


