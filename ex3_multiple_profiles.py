# -*- coding: utf-8 -*-
"""
Multi-scenario battery scheduling (identical methodology as original script)
Ensures same objective values for same Pcap.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# %% --- Read battery specs ---
parametersinput = pd.read_csv('./battery_data.csv', index_col=0)
parameters = parametersinput.loc[1]

capacity = parameters['Energy_capacity']
charging_power_limit = parameters["Power_capacity"]
discharging_power_limit = parameters["Power_capacity"]
charging_efficiency = parameters["Charging_efficiency"]
discharging_efficiency = parameters["Discharging_efficiency"]

# %% --- Read load and PV data ---
testData = pd.read_csv('./profile_input.csv')
Hours = testData['Hours'].values
Base_load = testData['Base_load'].values
PV_prod = testData['PV_prod'].values
Price = testData['Price'].values

dict_Prices = dict(zip(Hours, Price))
dict_Base_load = dict(zip(Hours, Base_load))
dict_PV_prod = dict(zip(Hours, PV_prod))

# %% --- Helper function ---
def stepify_centered(x, y):
    x_new = np.concatenate(([x[0]], x)) - 1
    y_new = np.concatenate(([y[0]], y))
    x_new = np.concatenate((x_new, [24]))
    y_new = np.concatenate((y_new, [y[-1]]))
    return x_new, y_new

# %% --- Simulation setup ---
sell_price_factor = 1.0
initial_soc_frac = 0.0
SoC0 = initial_soc_frac * capacity
final_soc_equal_start = True

Pcap_values = [8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.3]  # Different power capacities to test
scenario_results = {}

# %% --- Run Pyomo model for each scenario ---
for Pcap in Pcap_values:
    model = pyo.ConcreteModel()
    T = list(Hours)
    model.T = pyo.Set(initialize=T, ordered=True)

    model.price = pyo.Param(model.T, initialize=dict_Prices)
    model.base = pyo.Param(model.T, initialize=dict_Base_load)
    model.pv = pyo.Param(model.T, initialize=dict_PV_prod)

    Pcap_big = Pcap  # identical to original
    model.P_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, charging_power_limit if np.isinf(Pcap) else min(Pcap, charging_power_limit)))
    model.P_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, discharging_power_limit if np.isinf(Pcap) else min(Pcap, discharging_power_limit)))
    model.SOC = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, capacity))
    model.P_imp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, Pcap_big))
    model.P_exp = pyo.Var(model.T, domain=pyo.NonNegativeReals)

    # Binary for import/export
    model.y = pyo.Var(model.T, within=pyo.Binary)
    M = Pcap_big
    def imp_limit_rule(m, t): return m.P_imp[t] <= M * m.y[t]
    def exp_limit_rule(m, t): return m.P_exp[t] <= M * (1 - m.y[t])
    model.imp_limit = pyo.Constraint(model.T, rule=imp_limit_rule)
    model.exp_limit = pyo.Constraint(model.T, rule=exp_limit_rule)

    # Power balance
    model.balance = pyo.Constraint(
        model.T, rule=lambda m, t: m.base[t] == m.pv[t] + m.P_dis[t] + m.P_imp[t] - m.P_ch[t] - m.P_exp[t]
    )

    # SOC dynamics (circular like original)
    times = model.T.ordered_data()
    t0 = times[0]
    model.soc_init = pyo.Constraint(expr=model.SOC[t0] == SoC0)
    if final_soc_equal_start:
        model.soc_final = pyo.Constraint(expr=model.SOC[times[-1]] == SoC0)

    def soc_rule(m, t):
        t_prev = times[times.index(t)-1]
        return m.SOC[t] == m.SOC[t_prev] + charging_efficiency*m.P_ch[t] - (1.0/discharging_efficiency)*m.P_dis[t]
    model.soc_dyn = pyo.Constraint(model.T, rule=soc_rule)

    # Objective
    sell_price = {t: sell_price_factor*dict_Prices[t] for t in T}
    model.sell_price = pyo.Param(model.T, initialize=sell_price)
    model.cost = pyo.Objective(
        expr=sum(model.price[t]*model.P_imp[t] - model.sell_price[t]*model.P_exp[t] for t in model.T),
        sense=pyo.minimize)

    # Solve
    solver = SolverFactory('gurobi')
    solver.options['MIPGap'] = 1e-9
    solver.options['OptimalityTol'] = 1e-9
    solver.solve(model, tee=False)

    res_df = pd.DataFrame({
        'Hour': T,
        'Net_load_PV_Battery': [pyo.value(model.base[t] - model.pv[t] - model.P_dis[t] + model.P_ch[t]) for t in T],
        'Objective': [pyo.value(model.cost) for t in T],
    })

    scenario_results[Pcap] = res_df

# %% --- Plot all scenarios ---
plt.style.use("seaborn-v0_8-whitegrid")
plt.figure(figsize=(14, 7))
colors = plt.cm.viridis(np.linspace(0, 1, len(Pcap_values)))

for color, Pcap in zip(colors, Pcap_values):
    df = scenario_results[Pcap]
    x, y = stepify_centered(df["Hour"].values, df["Net_load_PV_Battery"].values)
    label = "∞ (no limit)" if np.isinf(Pcap) else f"{Pcap} kW"
    plt.step(x, y, where="post", color=color, linewidth=2.5, label=f"Pcap = {label}")

plt.title("Net Load Profiles for Different Battery Power Capacities", fontsize=15, fontweight="bold")
plt.xlabel("Hour")
plt.ylabel("Net Load [kW]")
plt.xlim(0, 24)
plt.xticks(np.arange(0, 25, 3))
plt.legend(title="Battery Power Capacity", fontsize=11, title_fontsize=12, loc="best", frameon=True)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# %% --- Print identical objective values ---
objectives = []
for Pcap, df in scenario_results.items():
    label = "∞" if np.isinf(Pcap) else str(Pcap)
    obj_value = df["Objective"].iloc[0]
    objectives.append((Pcap, obj_value))
    print(f"Objective value (Pcap={label}): {obj_value:.6f}")

# %% --- Plot objective value vs Pcap ---
plt.figure(figsize=(8, 5))
finite_Pcaps = [p for p, _ in objectives if not np.isinf(p)]
finite_objs = [o for p, o in objectives if not np.isinf(p)]

plt.plot(finite_Pcaps, finite_objs, 'o-', linewidth=2.5, markersize=8, label="Objective value")
plt.xlabel("Battery Power Capacity [kW]", fontsize=12)
plt.ylabel("Objective Value (Cost)", fontsize=12)
plt.title("Objective Function vs Battery Power Capacity", fontsize=14, fontweight="bold")
plt.grid(True, linestyle="--", alpha=0.7)

# Optionally include infinite case as annotation
inf_case = [(p, o) for p, o in objectives if np.isinf(p)]
if inf_case:
    plt.axhline(y=inf_case[0][1], color='gray', linestyle='--', label="∞ (no limit)")

plt.legend()
plt.tight_layout()
plt.show()
