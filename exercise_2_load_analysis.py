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
import matplotlib.cm as cm

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

#################### Task 1 ####################

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
    ax.plot(x, base_vm[path], "o-", label="Voltage Profile", color="C0")
    # mark minimum voltage and mark the value
    vmin = base_vm[path].min()
    ax.scatter(x[base_vm[path].argmin()], vmin, color="red", zorder=5)
    ax.annotate(f"{vmin:.4f} p.u.", xy=(x[base_vm[path].argmin()-1], vmin), xytext=(0,5), textcoords="offset points", fontsize=8, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels([net.bus.loc[bi, "name"] for bi in path], rotation=45)
    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage [p.u.]")
    #ax.set_title(f"Voltage profile along longest feeder to {end_name}")
    ax.grid()
    ax.legend()
    ax.set_ylim(0.94, 1.01)
    plt.tight_layout()
    plt.show()

#################### Task 2 ####################

def analyze_voltage_capacity(net, bus_i_subset, scaling_range=(1, 2), steps=11, v_limit=0.95, P_lim = P_lim):
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

    # add vertical line that intersects the voltage limit and Lowest voltage curve
    if P_lim is not None:
        # finn hvor spenningsgrensen krysses
        below_limit = results_df[results_df["Lowest voltage (p.u.)"] < v_limit]
        if not below_limit.empty:
            P_lim_cross = below_limit.iloc[0]["Aggregated load (MW)"]

    # --- plott
    plt.figure(figsize=(8,5))
    plt.plot(results_df["Aggregated load (MW)"], results_df["Lowest voltage (p.u.)"], "o-", label="Lowest voltage (Bus 96)")
    plt.axhline(v_limit, color="r", linestyle="--", label=f"Voltage limit ({v_limit} p.u.)")
    plt.axvline(P_lim, color="purple", linestyle="--", label=f"Power flow limit ({P_lim:.3f} MW)")
    plt.axvline(P_lim_cross, color="green", linestyle="--", label=f"Power flow limit ({P_lim_cross:.3f} MW)")
    # marker skaleringsfaktor ved punktene
    for x, y, sf in zip(results_df["Aggregated load (MW)"], results_df["Lowest voltage (p.u.)"], results_df["Scaling factor"]):
        plt.annotate(f"{sf:.2f}", (x, y), textcoords="offset points", xytext=(5,5), fontsize=8, color="black")
    
    plt.xlabel("Aggregated load in area (MW)")
    plt.ylabel("Lowest voltage (p.u.)")
    plt.title("Voltage vs. aggregated load in area")
    plt.grid()
    plt.legend()
    plt.show()

    return results_df

#################### Task 3 and Task 4 ####################

def plot_area_load_time_series(load_time_series_mapped, 
                               bus_i_subset, 
                               i_time_series_new_load,
                               P_lim = None,  
                               new_load_time_series = None,
                               which_plots=("buses", "new_load", "total", "smooth")):
    """
    Plot load time series for selected buses in the area + new load, total load, 
    and smoothed total load, in separate subplots. 
    Each curve gets a unique color.
    
    'total' plot is always placed at the top if selected.
    """
    # --- ensure "total" is plotted first if present
    ordered_plots = []
    if "total" in which_plots:
        ordered_plots.append("total")
    for key in ["buses", "new_load", "smooth"]:
        if key in which_plots:
            ordered_plots.append(key)

    # count how many subplots
    n_plots = 0
    if "buses" in ordered_plots:
        n_plots += len(bus_i_subset)
    if "new_load" in ordered_plots:
        n_plots += 1
    if "total" in ordered_plots:
        n_plots += 1
    if "smooth" in ordered_plots:
        n_plots += 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2*n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]  # make axes always a list

    plot_idx = 0
    load_sum = np.zeros(8760)

    # color map
    colors = cm.get_cmap("tab10", n_plots).colors  

    # --- draw in desired order
    for key in ordered_plots:
        if key == "total":
            # compute sum first
            for bus_i in bus_i_subset:
                load_sum += load_time_series_mapped[bus_i].to_numpy()

            if new_load_time_series is not None:
                load_sum += new_load_time_series

            if P_lim is not None:
                axes[plot_idx].axhline(P_lim, color='r', linestyle='--', label=f'Power flow limit {P_lim:.3f} MW')

            ax_total = axes[plot_idx]
            ax_total.plot(load_sum, color=colors[plot_idx], label='Total load in area')
            ax_total.plot(load_sum.argmax(), load_sum.max(), 'o', color='red')
            ax_total.annotate(f'{load_sum.max():.4f} MW', 
                              xy=(load_sum.argmax(), load_sum.max()), 
                              xytext=(5,0), textcoords='offset points',
                              fontsize=8, color='black', ha='left', va='center')
            ax_total.set_ylabel("MW")
            ax_total.legend(loc='upper left')
            ax_total.grid(True)
            plot_idx += 1

        elif key == "buses":
            for bus_i in bus_i_subset:
                ax = axes[plot_idx]
                series = load_time_series_mapped[bus_i]
                ax.plot(series, color=colors[plot_idx], label=f'Bus {bus_i}')
                
                max_val = series.max()
                max_hour = series.idxmax()
                ax.plot(max_hour, max_val, 'o', color='red')
                ax.annotate(f'{max_val:.4f} MW', 
                            xy=(max_hour, max_val), 
                            xytext=(5,0), textcoords='offset points',
                            fontsize=8, color='black', ha='left', va='center')
                ax.set_ylabel("MW")
                ax.legend(loc='upper left')
                ax.grid(True)
                plot_idx += 1

        elif key == "new_load":
            ax_new = axes[plot_idx]
            ax_new.plot(new_load_time_series, '--', color=colors[plot_idx], 
                        label=f'New load (bus {i_time_series_new_load})')

            max_val = new_load_time_series.max()
            max_hour = new_load_time_series.argmax()
            ax_new.plot(max_hour, max_val, 'o', color='red')
            ax_new.annotate(f'{max_val:.4f} MW', 
                            xy=(max_hour, max_val), 
                            xytext=(5,0), textcoords='offset points',
                            fontsize=8, color='black', ha='left', va='center')

            ax_new.set_ylabel("MW")
            ax_new.legend(loc='upper left')
            ax_new.grid(True)
            plot_idx += 1

        elif key == "smooth":
            window_size = 10*24
            window = np.ones(window_size) / window_size
            pad_width = window_size // 2
            load_sum_padded = np.pad(load_sum, pad_width, mode='reflect')
            load_sum_smooth = np.convolve(load_sum_padded, window, mode='valid')

            ax_smooth = axes[plot_idx]
            ax_smooth.plot(load_sum_smooth, color=colors[plot_idx], 
                           label='Total load in area (smoothed)')
            ax_smooth.set_ylabel("MW")
            ax_smooth.set_xlabel("Hour of the year")
            ax_smooth.legend(loc='upper left')
            ax_smooth.grid(True)
            plot_idx += 1

    plt.suptitle("Load time series for load points in the grid area", y=0.92)
    plt.tight_layout()
    plt.show()

    return load_sum, load_sum_smooth if "smooth" in ordered_plots else None

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

##################### Task 5 and Task 6 (deler av 6) ####################

def make_load_profile(load_sum, P_lim=None, mark_energy=False, mark_utilization_time=False, mark_peak_load=False):
    """
    Creates Load Duration Curve (LDC) and highlights:
    - Maximum load
    - Total annual energy (area under curve)
    - Utilization time (equivalent rectangle width)
    """
    # Sort load in descending order
    load_sum_sorted = np.sort(load_sum)[::-1]
    hours = np.arange(0, len(load_sum_sorted)) 

    # --- calculate metrics
    P_max = load_sum_sorted[0]               # maximum load (MW)
    E = load_sum_sorted.sum()               # total energy (MWh, since MW*1h)
    T_util = E / P_max                       # utilization time (hours)

    # --- plot
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(hours, load_sum_sorted, label='Load Duration Curve', color="tab:blue")
    
    if mark_energy:
        # shade area under curve = total energy
        ax.fill_between(hours, load_sum_sorted, color="tab:blue", alpha=0.2, label=f"Total energy = {E:.1f} MWh")

    # mark max power
    if mark_peak_load:
        ax.plot(0, P_max, 'ro')
        ax.annotate(f"Max load = {P_max:.3f} MW", xy=(0, P_max), xytext=(5, P_max*1.02),
                    arrowprops=dict(arrowstyle="->", color="red"), color="red")
    
    # mark number of hours where load exceeds P_limit
    if P_lim is not None:
        hours_exceeding_limit = np.sum(load_sum_sorted > P_lim)  # +1 to include the hour where it just exceeds
        ax.hlines(P_lim, 0, hours_exceeding_limit, colors="purple", linestyles="--")  #label=f"Hours above limit: {hours_exceeding_limit} h")
        ax.vlines(hours_exceeding_limit, 0, P_lim, colors="purple", linestyles="--")
        ax.annotate(f"{hours_exceeding_limit} h", xy=(hours_exceeding_limit, P_lim),
                    xytext=(hours_exceeding_limit+5, P_lim + 0.01), arrowprops=dict(arrowstyle="->", color="purple"),
                    color="purple")

    if mark_utilization_time:
        # mark utilization time (equivalent rectangle)
        ax.hlines(P_max, 0, T_util, colors="green", linestyles="--", label="Utilization time")
        ax.vlines(T_util, 0, P_max, colors="green", linestyles="--")
        ax.annotate(f"T_util = {T_util:.0f} h", xy=(T_util, P_max/2),
                    xytext=(T_util+200, P_max/2), arrowprops=dict(arrowstyle="->", color="green"),
                    color="green")

    # optional power flow limit line
    if P_lim is not None:
        ax.axhline(P_lim, color="r", linestyle="--", label=f"Power flow limit {P_lim:.3f} MW")

    ax.set_xlabel("Number of hours in the year")
    ax.set_ylabel("Load demand (MW)")
    #ax.set_title("Load Duration Curve with Utilization Time and Energy")
    ax.legend()
    ax.grid(True)
    plt.show()

    return load_sum_sorted, P_max, E, T_util

##################### Task 7 ####################

# Analytisk?

#################### Task 8 ####################

# bruk make_load_profile()

##################### Task 9 ####################

# Analytisk 

################# Task 10 ####################

# bruk make_load_profile()

################# Task 11 ####################

def characterize_flex_need_ldc(load_time_series_mapped, bus_i_subset, new_load_time_series, P_lim):
    """
    Karakteriserer fleksibilitetsbehovet når tidsavhengig ny last legges til området.
    Visualiserer behovet på en Load Duration Curve (LDC).
    """
    # --- total last i området (eksisterende + ny last)
    load_sum = np.zeros(8760)
    for bus_i in bus_i_subset:
        load_sum += load_time_series_mapped[bus_i].to_numpy()
    load_sum += new_load_time_series

    # --- overskridelser
    exceedance = load_sum - P_lim
    exceedance[exceedance < 0] = 0

    # sorter nedover = LDC for overskridelse
    exceed_sorted = np.sort(exceedance)[::-1]
    hours = np.arange(0, len(exceed_sorted))

    # --- nøkkelverdier
    capacity = exceed_sorted[0]                   # MW
    duration = np.count_nonzero(exceed_sorted)    # timer (inkluderer timen der det akkurat overskrides)
    energy = exceed_sorted.sum()                  # MWh

    results = {
        "Capacity (MW)": capacity,
        "Service duration (h)": duration,
        "Energy (MWh)": energy
    }

    print("=== Flexibility need (time-dependent new load, LDC) ===")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

    # --- visualisering på LDC
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(hours, exceed_sorted, label="Flexibility need", color="orange")

    # marker energi (areal)
    ax.fill_between(hours, exceed_sorted, color="orange", alpha=0.3,
                    label=f"Flexibility energy need = {energy:.1f} MWh")

    # marker maks kapasitet
    ax.plot(0, capacity, "ro")
    ax.annotate(f"Max Flexibility capacity = {capacity:.3f} MW",
                xy=(0, capacity),
                xytext=(10, capacity*0.95),
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red")

    # marker service duration
    ax.axvline(duration, color="purple", linestyle="--", label=f"Service duration = {duration} h")
    ax.annotate(f"{duration} h", 
                xy=(duration, 0),
                xytext=(duration+10, capacity*0.3),
                arrowprops=dict(arrowstyle="->", color="purple"),
                color="purple")

    ax.set_xlabel("Number of hours in the year")
    ax.set_ylabel("Flexibility need (MW)")
    ax.set_title("Flexibility need")
    ax.legend()
    ax.grid(True)
    plt.xlim(0, duration + 20)
    plt.show()

    return results

def scatter_overload_peak_vs_duration(load_time_series_mapped, bus_i_subset, new_load_time_series, P_lim):
    """
    Lager et scatterplott med overload peak (MW) som funksjon av utetid (h)
    for perioder med overskridelse av P_lim.
    """
    # --- total last
    load_sum = np.zeros(8760)
    for bus_i in bus_i_subset:
        load_sum += load_time_series_mapped[bus_i].to_numpy()
    load_sum += new_load_time_series

    # --- overskridelse
    exceedance = load_sum - P_lim
    exceedance[exceedance < 0] = 0

    # --- finn sammenhengende perioder med overskridelse
    episodes = []
    in_episode = False
    start = None

    for t in range(len(exceedance)):
        if exceedance[t] > 0 and not in_episode:
            in_episode = True
            start = t
        elif exceedance[t] == 0 and in_episode:
            in_episode = False
            end = t
            episodes.append((start, end))

    # hvis siste periode går til slutten av året
    if in_episode:
        episodes.append((start, len(exceedance)))

    # --- hent peaks og varigheter
    peaks = []
    durations = []
    for start, end in episodes:
        segment = exceedance[start:end]
        peaks.append(segment.max())
        durations.append(len(segment))

    # --- scatterplott
    plt.figure(figsize=(8,6))
    plt.scatter(durations, peaks, color="darkorange", alpha=0.7, edgecolor="k")
    plt.xlabel("Duration of overload episode (h)")
    plt.ylabel("Overload peak (MW)")
    plt.title("Overload peak vs duration of overload episodes")
    plt.grid(True)
    plt.show()

    # --- resultat som DataFrame
    results = pd.DataFrame({
        "Episode": range(1, len(peaks)+1),
        "Duration (h)": durations,
        "Overload peak (MW)": peaks
    })

    print("=== Overload episodes ===")
    print(results)

    return results

def scatter_overload_peaks_by_month(load_time_series_mapped, bus_i_subset, new_load_time_series, P_lim):
    """
    Lager et scatterplott med peak overload (MW) for hver overload-episode,
    plassert i måneden hvor episoden starter.
    """
    # --- total last
    load_sum = np.zeros(8760)
    for bus_i in bus_i_subset:
        load_sum += load_time_series_mapped[bus_i].to_numpy()
    load_sum += new_load_time_series

    # --- overskridelse
    exceedance = load_sum - P_lim
    exceedance[exceedance < 0] = 0

    # --- månedindeks (2019 = normalår 365 dager, 8760 timer)
    days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    hours_in_month = [d*24 for d in days_in_month]
    month_index = np.concatenate([np.full(h, m+1) for m, h in enumerate(hours_in_month)])

    # --- finn episoder
    episodes = []
    in_episode = False
    start = None
    for t in range(len(exceedance)):
        if exceedance[t] > 0 and not in_episode:
            in_episode = True
            start = t
        elif exceedance[t] == 0 and in_episode:
            in_episode = False
            end = t
            episodes.append((start, end))
    if in_episode:
        episodes.append((start, len(exceedance)))

    # --- hent peak og måned for hver episode
    peaks = []
    months = []
    for start, end in episodes:
        segment = exceedance[start:end]
        peak = segment.max()
        month = month_index[start]  # måned ved start
        if peak > 0:
            peaks.append(peak)
            months.append(month)

    # --- scatterplott
    plt.figure(figsize=(10,6))
    plt.scatter(months, peaks, color="darkorange", alpha=0.7, edgecolor="k")
    plt.xticks(range(1,13), ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.xlabel("Month")
    plt.ylabel("Peak overload (MW)")
    plt.title("Overload episodes: peak per month")
    plt.grid(True)
    plt.show()

    # --- resultattabell
    results = pd.DataFrame({
        "Month": months,
        "Peak overload (MW)": peaks
    })
    return results

################## Task 12 ####################

# Teorioppgave

################## Task 13 ####################

def compare_ldcs(load_time_series_mapped, bus_i_subset, new_load_time_series, P_const=0.4, P_lim=None):
    """
    Sammenlikner tre Load Duration Curves (LDCs):
      a) Eksisterende laster (uten ny last)
      b) Eksisterende + tidsavhengig ny last
      c) Eksisterende + konstant ny last (P_const MW hele året)
    
    Parametre
    ---------
    load_time_series_mapped : DataFrame
        Tidsserier for lastene i nettet (kolonner = busser, rader = timer).
    bus_i_subset : list
        Hvilke busser i området som skal inkluderes i summen.
    new_load_time_series : array-like
        Tidsserien (8760 elementer) for den tidsavhengige nye lasten.
    P_const : float
        Konstant last (MW) for det tredje scenariet. Default = 0.4 MW.
    P_lim : float eller None
        Eventuell grense som tegnes inn i plottet (MW).
    """
    # --- Eksisterende laster
    load_existing = np.zeros(8760)
    for bus_i in bus_i_subset:
        load_existing += load_time_series_mapped[bus_i].to_numpy()

    # --- Med tidsavhengig ny last
    load_with_new = load_existing + new_load_time_series

    # --- Med konstant ny last
    load_with_const = load_existing + P_const

    # --- LDCs (sorter nedover)
    ldc_existing = np.sort(load_existing)[::-1]
    ldc_with_new = np.sort(load_with_new)[::-1]
    ldc_with_const = np.sort(load_with_const)[::-1]
    hours = np.arange(1, 8761)

    # --- Plot
    plt.figure(figsize=(10,6))
    plt.plot(hours, ldc_with_const, label=f"a) With constant new load ({P_const:.1f} MW)")
    plt.plot(hours, ldc_with_new, label="b) With time-dependent new load")
    plt.plot(hours, ldc_existing, label="c) Existing loads only")
    
    

    if P_lim is not None:
        plt.axhline(P_lim, color="r", linestyle="--", label=f"Power flow limit {P_lim:.3f} MW")

        # mark hours exceeding limit for each scenario
        for ldc, label in zip([ldc_with_new, ldc_with_const],
                              ["With time-dependent new load", f"With constant new load ({P_const:.1f} MW)"]):
            hours_exceeding = np.sum(ldc > P_lim)
            plt.hlines(P_lim, 0, hours_exceeding, colors="purple", linestyles="--")
            plt.vlines(hours_exceeding, 0, P_lim, colors="purple", linestyles="--")
            plt.annotate(f"{hours_exceeding} h", xy=(hours_exceeding, P_lim),
                         xytext=(hours_exceeding+5, P_lim + 0.01), arrowprops=dict(arrowstyle="->", color="purple"),
                         color="purple")

    plt.xlabel("Number of hours in the year")
    plt.ylabel("Load demand (MW)")
    plt.title("Comparison of Load Duration Curves (LDCs)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return ldc_existing, ldc_with_new, ldc_with_const

################## Task 14 ####################

def compare_utilization_and_cf(load_time_series_mapped, bus_i_subset, new_load_time_series, P_const=0.4):
    """
    Beregner og sammenlikner utnyttelsestid og samtidighetsfaktor for tre scenarier:
      a) Eksisterende laster
      b) Eksisterende + tidsavhengig ny last
      c) Eksisterende + konstant ny last
    
    Lager en tabell og to søyleplott med verdier på toppen.
    """
    # --- Basis: eksisterende laster
    load_existing = np.zeros(8760)
    for bus_i in bus_i_subset:
        load_existing += load_time_series_mapped[bus_i].to_numpy()

    # --- Med tidsavhengig ny last
    load_with_new = load_existing + new_load_time_series

    # --- Med konstant ny last
    load_with_const = load_existing + P_const

    # --- Funksjon for metrics
    def metrics(load_area, loads_per_bus=None, add_const=None, add_series=None):
        P_max = load_area.max()
        E = load_area.sum()  # MWh
        T_util = E / P_max

        if loads_per_bus is None:
            loads_per_bus = [load_time_series_mapped[bus].to_numpy() for bus in bus_i_subset]

        Pmax_sum = sum([ld.max() for ld in loads_per_bus])
        if add_const is not None:
            Pmax_sum += add_const
        if add_series is not None:
            Pmax_sum += add_series.max()

        CF = P_max / Pmax_sum
        return T_util, CF

    # --- Beregn alle tre scenarier
    T_util_a, CF_a = metrics(load_with_const, add_const=P_const)
    T_util_b, CF_b = metrics(load_with_new, add_series=new_load_time_series)
    T_util_c, CF_c = metrics(load_existing)
    

    # --- Tabell
    results = pd.DataFrame({
        "Scenario": [f"a) With constant {P_const:.1f} MW new load", "b) With time-dependent new load", "c) Existing"],
        "Utilization time (h)": [T_util_a, T_util_b, T_util_c],
        "Coincidence factor": [CF_a, CF_b, CF_c]
    
    })

    print(results)

    # --- Søyleplott: Utnyttelsestid
    plt.figure(figsize=(7,5))
    bars = plt.bar(results["Scenario"], results["Utilization time (h)"], color=["C0","C1","C2"])
    plt.ylabel("Utilization time [h]")
    plt.title("Comparison of Utilization Time")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # legg til verdier
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01*yval, f"{yval:.0f}", 
                 ha="center", va="bottom", fontsize=9)
    plt.show()

    # --- Søyleplott: Coincidence Factor
    plt.figure(figsize=(7,5))
    bars = plt.bar(results["Scenario"], results["Coincidence factor"], color=["C0","C1","C2"])
    plt.ylabel("Coincidence factor")
    plt.title("Comparison of Coincidence Factors")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # legg til verdier
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01*yval, f"{yval:.3f}", 
                 ha="center", va="bottom", fontsize=9)
    plt.show()

    # lagre resultatene i latex tabellformat og skriv til fil
    latex_table = results.to_latex(index=False, float_format="%.3f")
    with open("utilization_and_cf_table.tex", "w") as f:
        f.write(latex_table)
        f.write("\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Comparison of Utilization Time and Coincidence Factor for Different Scenarios}\n")
        f.write("\\label{tab:utilization_cf_comparison}\n")
        f.write("\\end{table}\n")


    return results

def compare_utilization_and_cf_verbose(load_time_series_mapped, bus_i_subset, new_load_time_series, P_const=0.4):
    """
    Samme som før, men skriver ut ALLE mellomregninger:
    - P_max
    - E (energi)
    - T_util
    - P_i,max per buss
    - Eventuelle tillegg (konstant eller tidsavhengig last)
    - Sum P_i,max
    - CF
    """
    load_existing = np.zeros(8760)
    for bus_i in bus_i_subset:
        load_existing += load_time_series_mapped[bus_i].to_numpy()

    load_with_new = load_existing + new_load_time_series
    load_with_const = load_existing + P_const

    def metrics(name, load_area, loads_per_bus=None, add_const=None, add_series=None):
        P_max = load_area.max()
        E = load_area.sum()
        T_util = E / P_max

        if loads_per_bus is None:
            loads_per_bus = [load_time_series_mapped[bus].to_numpy() for bus in bus_i_subset]

        Pmax_per_bus = [ld.max() for ld in loads_per_bus]
        Pmax_sum = sum(Pmax_per_bus)

        # tillegg
        const_contrib = add_const if add_const is not None else 0
        series_contrib = add_series.max() if add_series is not None else 0

        Pmax_total = Pmax_sum + const_contrib + series_contrib
        CF = P_max / Pmax_total

        # print alt
        print(f"--- {name} ---")
        print(f"P_max (system)       = {P_max:.3f} MW")
        print(f"E (energi)           = {E:.3f} MWh")
        print(f"T_util               = {T_util:.3f} h")
        print("Bidrag til nevneren i CF:")
        for i, val in enumerate(Pmax_per_bus, 1):
            print(f"  Bus {i}: {val:.3f} MW")
        if add_const is not None:
            print(f"  Constant load: {const_contrib:.3f} MW")
        if add_series is not None:
            print(f"  Time-dependent load (max): {series_contrib:.3f} MW")
        print(f"Sum P_i,max + tillegg = {Pmax_total:.3f} MW")
        print(f"CF                    = {CF:.3f}")
        print()

        return T_util, CF

    # Beregn alle scenarier
    T_util_a, CF_a = metrics("a) With constant load", load_with_const, add_const=P_const)
    T_util_b, CF_b = metrics("b) With time-dependent load", load_with_new, add_series=new_load_time_series)
    T_util_c, CF_c = metrics("c) Existing", load_existing)

    results = pd.DataFrame({
        "Scenario": ["a) Constant 0.4 MW", "b) Time-dependent", "c) Existing"],
        "Utilization time (h)": [T_util_a, T_util_b, T_util_c],
        "Coincidence factor": [CF_a, CF_b, CF_c]
    })

    return results


################## Task 15 ####################

def scatter_overload_comparison(load_time_series_mapped, bus_i_subset, new_load_time_series, constant_load, P_lim):
    """
    Lager en figur med to subplots:
    (1) Overload peak (MW) vs duration (h)
    (2) Overload peak (MW) per month
    Sammenligner new load time series og constant load med to forskjellige farger.
    """
    def get_overload_episodes(load_sum, P_lim):
        exceedance = load_sum - P_lim
        exceedance[exceedance < 0] = 0

        # finn episoder
        episodes = []
        in_episode = False
        start = None
        for t in range(len(exceedance)):
            if exceedance[t] > 0 and not in_episode:
                in_episode = True
                start = t
            elif exceedance[t] == 0 and in_episode:
                in_episode = False
                end = t
                episodes.append((start, end))
        if in_episode:
            episodes.append((start, len(exceedance)))

        # hent peaks og varigheter
        peaks = []
        durations = []
        months = []

        days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
        hours_in_month = [d*24 for d in days_in_month]
        month_index = np.concatenate([np.full(h, m+1) for m, h in enumerate(hours_in_month)])

        for start, end in episodes:
            segment = exceedance[start:end]
            if segment.max() > 0:
                peaks.append(segment.max())
                durations.append(len(segment))
                months.append(month_index[start])

        return peaks, durations, months

    # total load for begge case
    base_load = np.zeros(8760)
    for bus_i in bus_i_subset:
        base_load += load_time_series_mapped[bus_i].to_numpy()

    load_new = base_load + new_load_time_series
    load_const = base_load + constant_load

    # hent data
    peaks_new, durations_new, months_new = get_overload_episodes(load_new, P_lim)
    peaks_const, durations_const, months_const = get_overload_episodes(load_const, P_lim)

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(14,6))

    # --- subplot 1: peak vs duration
    axes[0].scatter(durations_new, peaks_new, color="darkorange", alpha=0.9, edgecolor="k", label="New load")
    axes[0].scatter(durations_const, peaks_const, color="steelblue", alpha=0.7, edgecolor="k", label="Constant load")
    axes[0].set_xlabel("Duration of overload episode (h)")
    axes[0].set_ylabel("Overload peak (MW)")
    axes[0].set_title("Overload peak vs duration")
    axes[0].grid(True)
    axes[0].legend()

    # --- subplot 2: peaks by month
    axes[1].scatter(months_new, peaks_new, color="darkorange", alpha=0.9, edgecolor="k", label="scenario a) New load time series")
    axes[1].scatter(months_const, peaks_const, color="steelblue", alpha=0.7, edgecolor="k", label="scenario b) Constant load 0.4 MW")
    axes[1].set_xticks(range(1,13))
    axes[1].set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"])
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Peak overload (MW)")
    axes[1].set_title("Overload episodes per month")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # resultater i DataFrames
    results_new = pd.DataFrame({
        "Episode": range(1, len(peaks_new)+1),
        "Duration (h)": durations_new,
        "Peak overload (MW)": peaks_new,
        "Month": months_new,
        "Case": "New load"
    })

    results_const = pd.DataFrame({
        "Episode": range(1, len(peaks_const)+1),
        "Duration (h)": durations_const,
        "Peak overload (MW)": peaks_const,
        "Month": months_const,
        "Case": "Constant load"
    })

    results = pd.concat([results_new, results_const], ignore_index=True)

    return results


# Teorioppgave



################### Main code Execution ####################

loadsum, _ = plot_area_load_time_series(load_time_series_mapped, 
                                                  bus_i_subset,  
                                                  i_time_series_new_load,
                                                  P_lim = P_lim, 
                                                  new_load_time_series = new_load_time_series,
                                                  which_plots=("total"))

#plot_voltage_profile(net) 
#  
#table = make_load_table(net, bus_i_subset)
#print(table)

#load_sum_sorted = make_load_profile(loadsum, P_lim=None, mark_energy=False, mark_utilization_time=False, mark_peak_load=False)
#analyze_voltage_capacity(net, bus_i_subset, scaling_range=(1, 2), steps=11, v_limit=0.95, P_lim = P_lim)

#compare_ldcs(load_time_series_mapped, bus_i_subset, new_load_time_series, P_const=0.4, P_lim=P_lim)

#compare_utilization_and_cf(load_time_series_mapped, bus_i_subset, new_load_time_series, P_const=0.4)

#characterize_flex_need_ldc(load_time_series_mapped, bus_i_subset, new_load_time_series, P_lim)
#make_load_profile(loadsum, P_lim=P_lim, mark_energy=False, mark_utilization_time=False, mark_peak_load=True)

#scatter_overload_peak_vs_duration(load_time_series_mapped, bus_i_subset, new_load_time_series, P_lim)

#scatter_overload_peaks_by_month(load_time_series_mapped, bus_i_subset, new_load_time_series, P_lim)

#compare_utilization_and_cf_verbose(load_time_series_mapped, bus_i_subset, new_load_time_series, P_const=0.4)

scatter_overload_comparison(load_time_series_mapped, bus_i_subset, new_load_time_series, constant_load=0.4, P_lim=P_lim)