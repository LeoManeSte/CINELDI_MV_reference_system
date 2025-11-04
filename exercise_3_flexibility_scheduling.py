# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 15:30:27 2023

@author: merkebud, ivespe

Intro script for Exercise 3 ("Scheduling flexibility resources") 
in specialization course module "Flexibility in power grid operation and planning" 
at NTNU (TET4565/TET4575) 

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyomo.opt import SolverFactory
from pyomo.core import Var
import pyomo.environ as en
import time

#%% Read battery specifications
parametersinput = pd.read_csv('./battery_data.csv', index_col=0)
parameters = parametersinput.loc[1]


#Parse battery specification
capacity=parameters['Energy_capacity']
charging_power_limit=parameters["Power_capacity"]
discharging_power_limit=parameters["Power_capacity"]
charging_efficiency=parameters["Charging_efficiency"]
discharging_efficiency=parameters["Discharging_efficiency"]
#%% Read load demand and PV production profile data
testData = pd.read_csv('./profile_input.csv')


# Convert the various timeseries/profiles to numpy arrays
Hours = testData['Hours'].values
Base_load = testData['Base_load'].values
PV_prod = testData['PV_prod'].values
Price = testData['Price'].values

# Lag dictionary med 0–24
dict_Prices = dict(zip(Hours, Price))
dict_Base_load = dict(zip(Hours, Base_load))
dict_PV_prod = dict(zip(Hours, PV_prod))


# %% Optimal battery scheduling with Pyomo (cost minimization)
import pyomo.environ as pyo

# --- Tunables ---
sell_price_factor = 1.0      # 0.0 = ingen godtgjørelse, 1.0 = samme som kjøpspris
initial_soc_frac = 0.0     # start-SOC som andel av kapasitet
SoC0 = initial_soc_frac*capacity
final_soc_equal_start = True # håndhev at slutt-SOC = start-SOC

# Lag Pyomo-sett og parametre
T = list(Hours)  # antas å være på 1h steg og sortert

model = pyoConcrete = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T, ordered=True)

# Parametre
model.price = pyo.Param(model.T, initialize=dict_Prices)
model.base  = pyo.Param(model.T, initialize=dict_Base_load)
model.pv    = pyo.Param(model.T, initialize=dict_PV_prod)


Pcap = 1000 # large value to ensure no constraints are violated
# Vars
model.P_ch   = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, charging_power_limit))
model.P_dis  = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, discharging_power_limit))
model.SOC    = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, capacity))
model.P_imp  = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, Pcap))  # grid import
model.P_exp  = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # grid export


## ensure power to grid * power from grid = 0
#model.no_simultaneous_grid = pyo.ConstraintList()
#for t in model.T:
#    model.no_simultaneous_grid.add(model.P_imp[t] * model.P_exp[t] == 0)

# --- Binary variable for import/export mode ---
model.y = pyo.Var(model.T, within=pyo.Binary)  # 1 = import, 0 = export

M = Pcap  # big-M value (upper bound for grid flow)

# --- Linearized no simultaneous import/export constraint ---
def imp_limit_rule(m, t):
    return m.P_imp[t] <= M * m.y[t]

def exp_limit_rule(m, t):
    return m.P_exp[t] <= M * (1 - m.y[t])

model.imp_limit = pyo.Constraint(model.T, rule=imp_limit_rule)
model.exp_limit = pyo.Constraint(model.T, rule=exp_limit_rule)


# Effektbalanse hver time: last = PV + utlad + import - lad - eksport
model.balance = pyo.Constraint(
    model.T, rule=lambda m, t: m.base[t] == m.pv[t] + m.P_dis[t] + m.P_imp[t] - m.P_ch[t] - m.P_exp[t]
)

# SOC-dynamikk (1h steg): SOC_t = SOC_{t-1} + η_ch*P_ch - (1/η_dis)*P_dis
times = model.T.ordered_data()
t0 = times[0]
model.soc_init = pyo.Constraint(expr=model.SOC[t0] == SoC0)
# Slutt-SOC
if final_soc_equal_start:
    model.soc_final = pyo.Constraint(expr=model.SOC[times[-1]] == SoC0)

def soc_rule(m, t):
    t_prev = times[times.index(t)-1]
    return m.SOC[t] == m.SOC[t_prev] + charging_efficiency*m.P_ch[t] - (1.0/discharging_efficiency)*m.P_dis[t]
model.soc_dyn = pyo.Constraint(model.T, rule=soc_rule)


# Objective Function
sell_price = {t: sell_price_factor*dict_Prices[t] for t in T}
model.sell_price = pyo.Param(model.T, initialize=sell_price)
model.cost = pyo.Objective(
    expr=sum(model.price[t]*model.P_imp[t] - model.sell_price[t]*model.P_exp[t] for t in model.T), sense=pyo.minimize)

# Solve
solver = SolverFactory('gurobi')  
res = solver.solve(model, tee=False)

# Hent resultater til DataFrame
res_df = pd.DataFrame({
    'Hour': T,
    'Load': [pyo.value(model.base[t]) for t in T],
    'PV': [pyo.value(model.pv[t]) for t in T],
    'P_ch': [pyo.value(model.P_ch[t]) for t in T],
    'P_dis': [pyo.value(model.P_dis[t]) for t in T],
    'SOC': [pyo.value(model.SOC[t]) for t in T],
    'Import': [pyo.value(model.P_imp[t]) for t in T],
    'Export': [pyo.value(model.P_exp[t]) for t in T],
    'Price': [pyo.value(model.price[t]) for t in T],
    'Net_load_PV': [pyo.value(model.base[t] - model.pv[t]) for t in T],
    'Net_load_PV_Battery': [pyo.value(model.base[t] - model.pv[t] - model.P_dis[t] + model.P_ch[t]) for t in T],
    'Objective': [pyo.value(model.cost) for t in T],
})


total_cost = res_df['Import'].dot(res_df['Price']) - res_df['Export'].dot(res_df['Price']*sell_price_factor)
print(f"Total cost: {total_cost:.2f} (currency units)")


def stepify_centered(x, y):
    """
    Konverter timeserier (x=1..24) til step-plot-vennlig format,
    der hvert punkt gjelder i intervallet [t-0.5, t+0.5].
    """
    x_new = np.concatenate(([x[0]], x)) - 1
    y_new = np.concatenate(([y[0]], y))

    x_new = np.concatenate((x_new, [24])) 
    y_new = np.concatenate((y_new, [y[-1]]))
    return x_new, y_new

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# --- 1. Load & PV ---
x, y = stepify_centered(res_df["Hour"].values, res_df["Load"].values)
axs[0].step(x, y, label="Load = PV + Discharging + Import - Charging - Export", where="post", color="tab:blue", linewidth=2)

x, y = stepify_centered(res_df["Hour"].values, res_df["PV"].values)
axs[0].step(x, y, label="PV", where="post", color="tab:orange", linewidth=2)

axs[0].set_title("Load and PV production", fontsize=14, fontweight="bold")
axs[0].set_ylabel("Power [kW]")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.7)

# --- 2. Import & Export ---
x, y = stepify_centered(res_df["Hour"].values, res_df["Import"].values)
axs[1].step(x, y, label="Import", where="post", color="tab:green", linewidth=2)

x, y = stepify_centered(res_df["Hour"].values, res_df["Export"].values)
axs[1].step(x, y, label="Export", where="post", color="tab:red", linewidth=2)
# sett yticks til 1,2, 3, ... max import/export
max_import_export = max(res_df["Import"].max(), res_df["Export"].max())
axs[1].set_yticks(np.arange(0, max_import_export + 1, step=1))
axs[1].set_title("Grid import and export", fontsize=14, fontweight="bold")
axs[1].set_ylabel("Power [kW]")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.7)

# --- 3. Charge/Discharge & Price ---
ax1 = axs[2]
ax2 = ax1.twinx()

x, y = stepify_centered(res_df["Hour"].values, res_df["P_ch"].values)
ax1.step(x, y, label="Charging", where="post", color="tab:blue", linewidth=2)

x, y = stepify_centered(res_df["Hour"].values, res_df["P_dis"].values)
ax1.step(x, y, label="Discharging", where="post", color="tab:orange", linewidth=2)

x, y = stepify_centered(res_df["Hour"].values, res_df["Price"].values)
ax2.step(x, y, label="Price", where="post", color="black", linestyle="--", linewidth=2)

ax1.set_title("Battery operation and electricity price", fontsize=14, fontweight="bold")
ax1.set_ylabel("Power [kW]", color="black")
ax2.set_ylabel("Price [NOK/kWh]", color="black")
ax1.set_xlabel("Hour")

# Legender
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

ax1.grid(True, linestyle="--", alpha=0.7)

# --- Layout ---
axs[-1].set_xlim(0, 24)  # dekker [0.0, 24.0]
for ax in axs:
    ax.tick_params(axis="both", labelsize=11)

plt.tight_layout()
#plt.show()


# plot net load_profile with and without battery
plt.figure(figsize=(14, 6))
x, y = stepify_centered(res_df["Hour"].values, res_df["Net_load_PV"].values)
plt.step(x, y, label="Net load (Load - PV)", where="post", color="tab:blue", linewidth=2)   
x, y = stepify_centered(res_df["Hour"].values, res_df["Net_load_PV_Battery"].values)
plt.step(x, y, label="Net load with battery (Load - PV - Discharging + Charging)", where="post", color="tab:orange", linewidth=2)
plt.title("Net load profile with and without battery", fontsize=14, fontweight="bold")
plt.ylabel("Power [kW]")
plt.xlabel("Hour")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, 24)  # dekker [0.0, 24.0]
plt.ylim(min(0, min(res_df["Net_load_PV"].min(), res_df["Net_load_PV_Battery"].min()) - 1), 
         max(res_df["Net_load_PV"].max(), res_df["Net_load_PV_Battery"].max()) + 1) 
plt.xticks(np.arange(0, 25, step=1))
plt.yticks(np.arange(0, max(res_df["Net_load_PV"].max(), res_df["Net_load_PV_Battery"].max()) + 1, step=1))
plt.tight_layout()
#plt.show()
#%%

# make a new figure with state of charge on one axis and electricity price on the other axis
fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()
x, y = stepify_centered(res_df["Hour"].values, res_df["SOC"].values)
ax1.step(x, y, label="State of Charge (SoC)", where="post", color="tab:blue", linewidth=2)
x, y = stepify_centered(res_df["Hour"].values, res_df["Price"].values)
ax2.step(x, y, label="Price", where="post", color="black", linestyle="--", linewidth=2)
ax1.set_title("Battery State of Charge and Electricity Price", fontsize=14, fontweight="bold")
ax1.set_ylabel("State of Charge [kWh]", color="black")
ax2.set_ylabel("Price [NOK/kWh]", color="black")
ax1.set_xlabel("Hour")
# legender
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.7)
# layout
ax1.set_xlim(0, 24)  # dekker [0.0, 24.0]
ax1.tick_params(axis="both", labelsize=11)
plt.tight_layout()
#plt.show()
# %%


# kontroller at energien til batteriet er litt større enn energien som er levert fra batteriet
total_charged_energy = res_df['P_ch'].sum()
total_discharged_energy = res_df['P_dis'].sum()
print(f"Total charged energy: {total_charged_energy:.2f} kWh")
print(f"Total discharged energy: {total_discharged_energy:.2f} kWh")
# calculate energy efficiency
if total_charged_energy > 0:
    energy_efficiency = total_discharged_energy / total_charged_energy
    print(f"Energy efficiency: {energy_efficiency*100:.4f} %")
else:
    print("No energy charged, cannot calculate efficiency.")

# assert actual total efficiency of bettery
assert np.isclose(energy_efficiency, charging_efficiency * discharging_efficiency, atol=1e-4), "Energy efficiency does not match expected value."

# lag et plott med bare charge og discharge og pris på annen akse

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()
x, y = stepify_centered(res_df["Hour"].values, res_df["P_ch"].values)
ax1.step(x, y, label="Charging", where="post", color="tab:blue", linewidth=2)
x, y = stepify_centered(res_df["Hour"].values, res_df["P_dis"].values)
ax1.step(x, y, label="Discharging", where="post", color="tab:orange", linewidth=2)
x, y = stepify_centered(res_df["Hour"].values, res_df["Price"].values)
ax2.step(x, y, label="Price", where="post", color="black", linestyle="--", linewidth=2)
ax1.set_title("Battery Charging/Discharging and Electricity Price", fontsize=14, fontweight="bold")
ax1.set_ylabel("Power [kW]", color="black")
ax2.set_ylabel("Price [NOK/kWh]", color="black")
ax1.set_xlabel("Hour")
# legender
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.7)
# layout
ax1.set_xlim(0, 24)  # dekker [0.0, 24.0]
ax1.tick_params(axis="both", labelsize=11)
plt.tight_layout()
#plt.show()


# save net load profile with and without battery to csv file
res_df.to_csv('Load_profile_task7.csv', index=False)
# %%

# function to read from two csv files and plot the net load profiles in one plot and without battery
def plot_net_load_profiles(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    plt.figure(figsize=(14, 6))

    # without battery
    x, y = stepify_centered(df1["Hour"].values, df1["Net_load_PV"].values)
    plt.step(x, y, label="Net load without battery", where="post", color="tab:blue", linewidth=2)

    x, y = stepify_centered(df1["Hour"].values, df1["Net_load_PV_Battery"].values)
    plt.step(x, y, label="Net load with battery ($P_{cap} = \\infty$)", where="post", color="tab:orange", linewidth=2)   
    x, y = stepify_centered(df2["Hour"].values, df2["Net_load_PV_Battery"].values)
    plt.step(x, y, label="Net load with battery ($P_{cap}$ = 6 kW)", where="post", color="tab:green", linewidth=2)

    plt.title("Net load profile with battery from two exercises", fontsize=14, fontweight="bold")
    plt.ylabel("Power [kW]")
    plt.xlabel("Hour")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlim(0, 24)  # dekker [0.0, 24.0]
    plt.ylim(min(0, min(df1["Net_load_PV_Battery"].min(), df2["Net_load_PV_Battery"].min()) - 1), 
             max(df1["Net_load_PV_Battery"].max(), df2["Net_load_PV_Battery"].max()) + 1) 
    plt.xticks(np.arange(0, 25, step=1))
    plt.yticks(np.arange(0, max(df1["Net_load_PV_Battery"].max(), df2["Net_load_PV_Battery"].max()) + 1, step=1))
    plt.tight_layout()
    plt.show()


# call the function with the two csv files
plot_net_load_profiles('Load_profile_task4.csv', 'Load_profile_task7.csv')

# print ut objective value from both files
df1 = pd.read_csv('Load_profile_task4.csv')
df2 = pd.read_csv('Load_profile_task7.csv')
print(f"Objective value from task 4 (no Pcap limit): {df1['Objective'].iloc[0]:.2f} (currency units)")
print(f"Objective value from task 7 (with Pcap limit): {df2['Objective'].iloc[0]:.2f} (currency units)")
