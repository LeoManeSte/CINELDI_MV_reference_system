# -*- coding: utf-8 -*-
"""
Created on 2023-07-13

@author: ivespe

Intro script for warm-up exercise ("exercise 0") in specialization course module 
"Flexibility in power grid operation and planning" at NTNU (TET4565/TET4575) 
"""

# %% Dependencies

from curses import panel
import pandapower as pp
import pandapower.plotting as pp_plotting
import pandas as pd
import os
import load_scenarios as ls
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import networkx as nx
from matplotlib.widgets import Button, TextBox
import numpy as np
import pandapower as pp
import matplotlib.gridspec as gridspec





# %% Define input data

# Location of (processed) data set for CINELDI MV reference system
# (to be replaced by your own local data folder)
path_data_set         = 'CINELDI_MV_reference_system_v_2023-03-06/'

filename_residential_fullpath = os.path.join(path_data_set,'time_series_IDs_primarily_residential.csv')
filename_irregular_fullpath = os.path.join(path_data_set,'time_series_IDs_irregular.csv')      
filename_load_data_fullpath = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
filename_load_mapping_fullpath = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')

# %% Read pandapower network

net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)


# %% Test running power flow with a peak load model
# (i.e., all loads are assumed to be at their annual peak load simultaneously)

pp.runpp(net,init='results',algorithm='bfsw')

print('Total load demand in the system assuming a peak load model: ' + str(net.res_load['p_mw'].sum()) + ' MW')

# %% Plot results of power flow calculations

#pp_plotting.pf_res_plotly(net)

# %%

# --- Helpers ---------------------------------------------------------------

def add_load_pf(net, bus, p_mw, cos_phi, name="extra load"):
    """Add load at bus with P and power factor cosφ. q_mvar sign follows p and cosφ."""
    q_mvar = p_mw * np.tan(np.arccos(cos_phi))
    return pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)

def add_battery_as_load(net, bus, p_mw, q_mvar=0.0, name="battery"):
    """
    Represent battery as a load element.
      - p_mw < 0  => injects active power to the grid
      - q_mvar < 0 => injects reactive power
    """
    return pp.create_load(net, bus=bus, p_mw=p_mw, q_mvar=q_mvar, name=name)

def _cleanup_user_elements(net):
    """Remove previously added 'user load' and 'battery' loads."""
    if len(net.load):
        net.load = net.load[~net.load["name"].isin(["user load", "battery"])]

# --- Main UI ---------------------------------------------------------------

def plot_longest_feeder_with_controls(
    net,
    initial_load_bus=95,
    initial_bat_bus=94,
    initial_p_mw=1.0,
    initial_cos_phi=0.95,
    initial_bat_p_mw=0.0,
    initial_bat_q_mvar=0.0,
    min_voltage_pu_default=0.95,
    title="Voltage profile of the longest feeder"
):
    # 1) Solve baseline
    pp.runpp(net)
    base_vm = net.res_bus["vm_pu"].copy()

    # 2) Add initial user load and solve
    _cleanup_user_elements(net)
    add_load_pf(net, initial_load_bus, initial_p_mw, initial_cos_phi, name="user load")
    if abs(initial_bat_p_mw) > 1e-12 or abs(initial_bat_q_mvar) > 1e-12:
        add_battery_as_load(net, initial_bat_bus, initial_bat_p_mw, initial_bat_q_mvar, name="battery")
    pp.runpp(net)
    load_vm = net.res_bus["vm_pu"].copy()

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
    ln_load, = ax.plot(x, load_vm.loc[path].values, marker="o", linewidth=1.8,
                       label=f"Feeder to {end_name} ({len(path)} buses)")
    ln_base, = ax.plot(x, base_vm.loc[path].values, marker="x", linestyle="--", alpha=0.9,
                       linewidth=1.0, label="Base case")

    # Bus ID annotations: fixed offset above each point, updated dynamically
    annots = []
    for i, b in enumerate(path):
        a = ax.annotate(str(int(b)), xy=(i, load_vm.loc[b]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7, visible=True)
        annots.append(a)

    ax.set_xlabel("Bus along feeder path")
    ax.set_ylabel("Voltage [p.u.]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Info box (figure text, stays put)
    def make_info_text(current_vm):
        lowest_bus = int(current_vm.idxmin())
        lowest_v = float(current_vm.min())
        base_at_same = float(base_vm.loc[lowest_bus])
        lines = [
            f"Global min (load): {lowest_v:.4f} p.u. at bus {lowest_bus}",
            f"Base at same bus:  {base_at_same:.4f} p.u.",
        ]
        return "\n".join(lines)

    info_txt = fig.text(
        0.78, 0.92, make_info_text(load_vm),  # anchored near top-right of the figure
        ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.4")
    )

    # 5) Control panel (two rows: LOAD on top, BATTERY below)
    panel = fig.add_subplot(gs[1])
    panel.axis("off")

    # Row positions inside panel (0..1 coordinates)
    row_load_y = 0.58
    row_bat_y  = 0.10
    row_reset_y = -0.28
    field_w, field_h = 0.18, 0.32
    gap = 0.1

    # --- Row 1: LOAD
    tb_load_bus = TextBox(panel.inset_axes([0.02, row_load_y, field_w, field_h]),
                          "Load bus", initial=str(initial_load_bus))
    tb_p        = TextBox(panel.inset_axes([0.02 + (field_w+gap), row_load_y, field_w, field_h]),
                          "P [MW]", initial=str(initial_p_mw))
    tb_pf       = TextBox(panel.inset_axes([0.02 + 2*(field_w+gap), row_load_y, field_w, field_h]),
                          "cosφ", initial=str(initial_cos_phi))

    # --- Row 2: BATTERY
    tb_bat_bus  = TextBox(panel.inset_axes([0.02, row_bat_y, field_w, field_h]),
                          "Bat bus", initial=str(initial_bat_bus))
    tb_bat_p    = TextBox(panel.inset_axes([0.02 + (field_w+gap), row_bat_y, field_w, field_h]),
                          "Bat P [MW]", initial=str(initial_bat_p_mw))
    tb_bat_q    = TextBox(panel.inset_axes([0.02 + 2*(field_w+gap), row_bat_y, field_w, field_h]),
                          "Bat Q [Mvar]", initial=str(initial_bat_q_mvar))
    tb_min_v    = TextBox(panel.inset_axes([0.02 + 3*(field_w+gap), row_bat_y, field_w, field_h]),
                          "Min V [p.u.]", initial=str(min_voltage_pu_default))

    
    
    # --- Row 3: RESET button and auto battery sizing button
    btn_auto    = Button(panel.inset_axes([0.02, row_reset_y, field_w, field_h]),
                         "Auto battery")

    btn_reset   = Button(panel.inset_axes([0.02 + (field_w+gap), row_reset_y, field_w, field_h]),
                         "Reset")

    # --- internal helpers ---------------------------------------------------

    def _update_plot_from_vm(current_vm):
        """Update line, annotations and info box for new voltages."""
        yvals = current_vm.loc[path].values
        ln_load.set_ydata(yvals)
        # Keep constant offset, move reference xy to data point
        for i, a in enumerate(annots):
            a.xy = (i, yvals[i])
        info_txt.set_text(make_info_text(current_vm))
        fig.canvas.draw_idle()

    def _run_power_flow_from_inputs(p_only_battery=False):
        """Rebuild user elements from UI, run power flow, and return vm series."""
        # Read inputs
        try:
            load_bus = int(float(tb_load_bus.text))
            p_mw = float(tb_p.text)
            cosphi = float(tb_pf.text)

            bat_bus = int(float(tb_bat_bus.text))
            bat_p = float(tb_bat_p.text)
            bat_q = 0.0 if p_only_battery else float(tb_bat_q.text)
        except Exception as e:
            print("Input parse error:", e)
            return net.res_bus["vm_pu"]

        # Rebuild elements
        _cleanup_user_elements(net)
        add_load_pf(net, load_bus, p_mw, cosphi, name="user load")
        if abs(bat_p) > 1e-9 or abs(bat_q) > 1e-9:
            add_battery_as_load(net, bat_bus, bat_p, bat_q, name="battery")

        # Solve
        pp.runpp(net)
        return net.res_bus["vm_pu"]

    # --- callbacks ----------------------------------------------------------

    def on_any_submit(_=None):
        vm = _run_power_flow_from_inputs()
        _update_plot_from_vm(vm)

    def on_reset(_):
        # reset UI fields
        tb_load_bus.set_val(str(initial_load_bus))
        tb_p.set_val(str(initial_p_mw))
        tb_pf.set_val(str(initial_cos_phi))
        tb_bat_bus.set_val(str(initial_bat_bus))
        tb_bat_p.set_val(str(initial_bat_p_mw))
        tb_bat_q.set_val(str(initial_bat_q_mvar))
        tb_min_v.set_val(str(min_voltage_pu_default))

        # reset network state
        _cleanup_user_elements(net)
        add_load_pf(net, initial_load_bus, initial_p_mw, initial_cos_phi, name="user load")
        if abs(initial_bat_p_mw) > 1e-12 or abs(initial_bat_q_mvar) > 1e-12:
            add_battery_as_load(net, initial_bat_bus, initial_bat_p_mw, initial_bat_q_mvar, name="battery")
        pp.runpp(net)
        _update_plot_from_vm(net.res_bus["vm_pu"])

    def on_auto_battery(_):
        """
        Compute the necessary battery P at tb_bat_bus to meet tb_min_v,
        using Q from tb_bat_q. Adjusts P up or down as needed.
        """
        try:
            v_min_target = float(tb_min_v.text)
        except:
            v_min_target = 0.95

        small_step = -0.02  # MW per iteration when increasing injection
        max_inj = -50.0     # MW lower bound (most negative allowed)
        min_inj = 0.0       # MW upper bound (no injection)

        # Helper: evaluate min voltage for a given P, keeping Q from textbox
        def eval_with_batP(p_mw):
            tb_bat_p.set_val(f"{p_mw:.2f}")
            vm = _run_power_flow_from_inputs(p_only_battery=False)
            return float(vm.min()), vm

        # Current starting point
        p_guess = float(tb_bat_p.text)
        v_now, _ = eval_with_batP(p_guess)

        # Case 1: Already above target -> try to reduce injection until just at target
        if v_now >= v_min_target:
            current_p = p_guess
            current_v, vm_ser = v_now, net.res_bus["vm_pu"]
            while current_v >= v_min_target and current_p < min_inj - small_step:
                current_p -= small_step  # small_step is negative, so this reduces injection
                current_v, vm_ser = eval_with_batP(current_p)
            # Step back one if we went too far
            if current_v < v_min_target:
                current_p -= small_step
                _, vm_ser = eval_with_batP(current_p)
            tb_bat_p.set_val(f"{current_p:.2f}")
            _update_plot_from_vm(vm_ser)
            return

        # Case 2: Below target -> inject more until target reached
        else:
            current_p = p_guess
            current_v, vm_ser = v_now, net.res_bus["vm_pu"]
            while current_v < v_min_target and current_p > max_inj:
                current_p += small_step  # small_step is negative, so P goes more negative
                current_v, vm_ser = eval_with_batP(current_p)
            tb_bat_p.set_val(f"{current_p:.2f}")
            _update_plot_from_vm(vm_ser)

    # Wire events
    for tb in [tb_load_bus, tb_p, tb_pf, tb_bat_bus, tb_bat_p, tb_bat_q, tb_min_v]:
        tb.on_submit(on_any_submit)
    btn_reset.on_clicked(on_reset)
    btn_auto.on_clicked(on_auto_battery)

    plt.show()

# --- Usage (assuming you already have a pandapower 'net'):
# plot_longest_feeder_with_controls(net)

# Call the function to plot feeder profiles
feeder_info = plot_longest_feeder_with_controls(net)

## plot feeder info
#for info in feeder_info:
#    print(f"Feeder to {info['end_name']} (length {info['length']}): min voltage {info['min_voltage']:.4f} p.u. at bus {info['min_bus']}")