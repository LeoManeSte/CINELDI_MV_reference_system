# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import load_profiles as lp
import pandapower_read_csv as ppcsv

PATH_DATASET = "CINELDI_MV_reference_system_v_2023-03-06/"
FILE_LOAD = os.path.join(PATH_DATASET, "load_data_CINELDI_MV_reference_system.csv")
FILE_MAP = os.path.join(PATH_DATASET, "mapping_loads_to_CINELDI_MV_reference_grid.csv")
FILE_BATT = "./battery_data.csv"
FILE_EX3 = "./profile_input.csv"

BUS_I_SUBSET = [90, 91, 92, 96]
REPR_DAYS = [31 + 28]    
GROWTH = 0.03
YEAR_START_Y = 5          # skaler til år 6: (1+g)^5
SCALE_TO_YEAR6 = (1.0 + GROWTH) ** YEAR_START_Y
SCALING_FACTOR = 10.0    

CAPACITY_MWH = 2.0
P_CH_MAX_MW = 1.0
P_DIS_MAX_MW = 2.0
P_LIM = 4.0             
DT_H = 1.0                
SELL_PRICE_FACTOR = 1.0
INITIAL_SOC_FRAC = 0.0
FINAL_SOC_EQUAL_START = True

def read_inputs():
    parameters = pd.read_csv(FILE_BATT, index_col=0).loc[1]

    ex3 = pd.read_csv(FILE_EX3)
    hours_price = ex3["Hours"].values.astype(int)    
    price = ex3["Price"].values.astype(float)

    net = ppcsv.read_net_from_csv(PATH_DATASET, baseMVA=10)
    profiles = lp.load_profiles(FILE_LOAD)
    rel_profiles = profiles.map_rel_load_profiles(FILE_MAP, REPR_DAYS) 

    load_ts_MW = rel_profiles.mul(net.load["p_mw"]) 
    agg_area_MW = (load_ts_MW[BUS_I_SUBSET] * SCALING_FACTOR).sum(axis=1).values
    agg_area_year6_MW = SCALE_TO_YEAR6 * agg_area_MW

    if len(agg_area_year6_MW) != len(hours_price):
        raise ValueError("Ulik lengde mellom last og pris – sjekk inndata!")

    hours = list(hours_price)
    return parameters, agg_area_year6_MW, price, hours


def build_data_dicts(load_MW: np.ndarray, price: np.ndarray, hours: List[int]):
    pv_MW = np.zeros_like(load_MW)
    dict_prices = dict(zip(hours, price))
    dict_load = dict(zip(hours, load_MW))
    dict_pv = dict(zip(hours, pv_MW))
    dict_sell = {t: SELL_PRICE_FACTOR * dict_prices[t] for t in hours}
    return dict_load, dict_prices, dict_pv, dict_sell


def choose_solver():
    """Velger første tilgjengelige MILP-solver."""
    for cand in ("gurobi", "highs", "glpk"):
        try:
            s = SolverFactory(cand)
            if s.available(False):
                return cand, s
        except Exception:
            pass
    raise RuntimeError("Fant ingen tilgjengelig MILP-solver (gurobi/highs/glpk).")


def build_model(
    dict_load: Dict[int, float],
    dict_prices: Dict[int, float],
    dict_pv: Dict[int, float],
    dict_sell: Dict[int, float],
    eta_ch: float,
    eta_dis: float,
):
    Mbig = 10.0 * max(1.0, float(np.max(list(dict_load.values()))))
    SoC0 = INITIAL_SOC_FRAC * CAPACITY_MWH

    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=list(dict_load.keys()), ordered=True)

    model.price = pyo.Param(model.T, initialize=dict_prices)
    model.load_MW = pyo.Param(model.T, initialize=dict_load)
    model.pv_MW = pyo.Param(model.T, initialize=dict_pv)
    model.sell_price = pyo.Param(model.T, initialize=dict_sell)

    model.P_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, P_CH_MAX_MW))
    model.P_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, P_DIS_MAX_MW))
    model.SOC = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, CAPACITY_MWH))
    model.P_imp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, P_LIM))
    model.P_exp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, Mbig))

    model.y_grid = pyo.Var(model.T, within=pyo.Binary)
    model.z_batt = pyo.Var(model.T, within=pyo.Binary)

    def imp_limit_rule(m, t): return m.P_imp[t] <= P_LIM * m.y_grid[t]
    def exp_limit_rule(m, t): return m.P_exp[t] <= Mbig * (1 - m.y_grid[t])
    model.imp_limit = pyo.Constraint(model.T, rule=imp_limit_rule)
    model.exp_limit = pyo.Constraint(model.T, rule=exp_limit_rule)

    def ch_mode_rule(m, t):  return m.P_ch[t]  <= P_CH_MAX_MW * m.z_batt[t]
    def dis_mode_rule(m, t): return m.P_dis[t] <= P_DIS_MAX_MW * (1 - m.z_batt[t])
    model.ch_mode = pyo.Constraint(model.T, rule=ch_mode_rule)
    model.dis_mode = pyo.Constraint(model.T, rule=dis_mode_rule)

    def balance_rule(m, t):
        return m.load_MW[t] == m.pv_MW[t] + m.P_dis[t] + m.P_imp[t] - m.P_ch[t] - m.P_exp[t]
    model.balance = pyo.Constraint(model.T, rule=balance_rule)

    times = model.T.ordered_data()
    t0 = times[0]
    model.soc_init = pyo.Constraint(expr=model.SOC[t0] == SoC0)

    def soc_rule(m, t):
        t_prev = times[times.index(t) - 1]
        return m.SOC[t] == (
            m.SOC[t_prev] + eta_ch * m.P_ch[t] * DT_H - (1.0 / eta_dis) * m.P_dis[t] * DT_H
        )
    model.soc_dyn = pyo.Constraint(model.T, rule=soc_rule)

    def dis_energy_rule(m, t):
        if t == t0:
            return m.P_dis[t] * DT_H <= SoC0 * eta_dis
        t_prev = times[times.index(t) - 1]
        return m.P_dis[t] * DT_H <= m.SOC[t_prev] * eta_dis

    def ch_energy_rule(m, t):
        if t == t0:
            return m.P_ch[t] * DT_H <= (CAPACITY_MWH - SoC0) / eta_ch
        t_prev = times[times.index(t) - 1]
        return m.P_ch[t] * DT_H <= (CAPACITY_MWH - m.SOC[t_prev]) / eta_ch

    model.dis_energy_limit = pyo.Constraint(model.T, rule=dis_energy_rule)
    model.ch_energy_limit = pyo.Constraint(model.T, rule=ch_energy_rule)

    if FINAL_SOC_EQUAL_START:
        model.soc_final = pyo.Constraint(expr=model.SOC[times[-1]] == SoC0)

    model.obj = pyo.Objective(
        expr=sum(model.sell_price[t] * model.P_exp[t] - model.price[t] * model.P_imp[t] for t in model.T),
        sense=pyo.maximize,
    )
    return model


def solve_model(model: pyo.ConcreteModel):
    solver_name, solver = choose_solver()
    print(f"Using solver: {solver_name}")
    res = solver.solve(model, tee=False)
    return solver_name, res


def extract_results(model: pyo.ConcreteModel, hours: List[int]):
    T = hours
    res_df = pd.DataFrame({
        "Hour": T,
        "Load_MW":   [pyo.value(model.load_MW[t]) for t in T],
        "PV_MW":     [pyo.value(model.pv_MW[t]) for t in T],
        "P_ch_MW":   [pyo.value(model.P_ch[t]) for t in T],
        "P_dis_MW":  [pyo.value(model.P_dis[t]) for t in T],
        "SOC_MWh":   [pyo.value(model.SOC[t]) for t in T],
        "Import_MW": [pyo.value(model.P_imp[t]) for t in T],
        "Export_MW": [pyo.value(model.P_exp[t]) for t in T],
        "Price":     [pyo.value(model.price[t]) for t in T],
    })
    res_df["Net_load_no_batt_MW"] = res_df["Load_MW"] - res_df["PV_MW"]
    res_df["Net_load_with_batt_MW"] = res_df["Import_MW"] - res_df["Export_MW"]

    # Konsistenssjekker
    calc_internal = res_df["Load_MW"] - res_df["PV_MW"] - res_df["P_dis_MW"] + res_df["P_ch_MW"]
    assert np.allclose(
        res_df["Net_load_with_batt_MW"].values, calc_internal.values, atol=1e-6
    ), "Mismatch mellom Net_load_with_batt og (Load - PV - Dis + Ch)."

    assert np.allclose(
        res_df["Load_MW"].values,
        (res_df["PV_MW"] + res_df["P_dis_MW"] + res_df["Import_MW"] - res_df["P_ch_MW"] - res_df["Export_MW"]).values,
        atol=1e-6,
    ), "Effektbalanse brutt – sjekk modell!"
    return res_df


def evaluate_and_print(res_df: pd.DataFrame):
    total_revenue = (res_df["Export_MW"] * res_df["Price"] - res_df["Import_MW"] * res_df["Price"]).sum()

    peak_no_batt = float(res_df["Net_load_no_batt_MW"].max())
    peak_with_batt_import = float(res_df["Import_MW"].max())  # ≤ P_LIM
    hours_over_no_batt = int((res_df["Net_load_no_batt_MW"] > P_LIM).sum())
    hours_over_with_batt = int((res_df["Import_MW"] > P_LIM + 1e-9).sum())

    print("\n--- VURDERING (Oppg. 14) ---")
    print(f"Topp uten batteri: {peak_no_batt:.3f} MW")
    print(f"Topp med batteri (import): {peak_with_batt_import:.3f} MW (grense {P_LIM} MW)")
    print(f"Timer over {P_LIM} MW uten batteri: {hours_over_no_batt} h")
    print(f"Timer over {P_LIM} MW med batteri:  {hours_over_with_batt} h (bør være 0)")
    print(f"Total inntekt (salg - kjøp): {total_revenue:.2f} (pris-enheter)")

    print("\nKommentar:")
    print("- Batteriet (1 MW / 2 MWh) kan dempe toppene med inntil 1 MW i ~2 timer.")
    print("- Overskridelser >1 MW eller lengre enn ~2 timer krever flere tiltak eller større batteri.")
    print("- Med eksplisitt importgrense (P_imp ≤ 4 MW) holdes import under grensen,")
    print("  og SoC/Net_to_grid viser hvordan batteriet flytter last i tid for å klare dette.")


def _stepify_1h(hours: np.ndarray, y: np.ndarray):
    h = np.asarray(hours, dtype=float)
    v = np.asarray(y, dtype=float)
    if len(h) > 1 and not np.allclose(np.diff(h), 1.0):
        raise ValueError("Hours må være 1-timesteget.")
    if np.isclose(h[0], 1.0):
        x = np.arange(0.0, h[-1] + 1.0, 1.0)  # 0..24
    else:
        x = np.r_[h, h[-1] + 1.0]
    y_steps = np.r_[v, v[-1]]
    return x, y_steps


def plot_results(res_df: pd.DataFrame):
    c_no = "#1f77b4"   
    c_with = "#ff7f0e" 
    c_soc = "#2ca02c"  
    c_plim = "#444444" 

    x_no, y_no = _stepify_1h(res_df["Hour"].values, res_df["Net_load_no_batt_MW"].values)
    x_with, y_with = _stepify_1h(res_df["Hour"].values, res_df["Net_load_with_batt_MW"].values)
    x_soc, y_soc = _stepify_1h(res_df["Hour"].values, res_df["SOC_MWh"].values)

    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.step(x_no, y_no, where="post", lw=2.6, color=c_no, label="Net load without battery")
    ax.step(x_with, y_with, where="post", lw=2.6, color=c_with, label="Net load with battery")
    ax.axhline(P_LIM, ls="--", lw=2, color=c_plim, label=f"Limit = {P_LIM:g} MW")


    ax.set_xlabel("Hour")
    ax.set_ylabel("Power [MW]")
    ax.set_xlim(x_no[0], x_no[-1])
    ax.grid(True, ls="--", alpha=0.5)

    ax2 = ax.twinx()
    ax2.step(x_soc, y_soc, where="post", lw=2.4, ls="--", color=c_soc, label="SoC")
    ax2.set_ylabel("State of Charge [MWh]")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=True)

    plt.title("Net load (without/with battery) og SoC")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parameters, load_MW, price, hours = read_inputs()
    eta_ch = float(parameters["Charging_efficiency"])
    eta_dis = float(parameters["Discharging_efficiency"])

    dict_load, dict_prices, dict_pv, dict_sell = build_data_dicts(load_MW, price, hours)
    #print(dict_load)

    model = build_model(dict_load, dict_prices, dict_pv, dict_sell, eta_ch, eta_dis)
    _, _ = solve_model(model)

    res_df = extract_results(model, hours)

    evaluate_and_print(res_df)

    plot_results(res_df)

if __name__ == "__main__":
    main()

