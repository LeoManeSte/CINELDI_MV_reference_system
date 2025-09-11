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

# Make dictionaries (for simpler use in Pyomo)
dict_Prices = dict(zip(Hours, Price))
dict_Base_load = dict(zip(Hours, Base_load))
dict_PV_prod = dict(zip(Hours, PV_prod))
# %%


# %% Optimal battery scheduling with Pyomo (cost minimization)
import pyomo.environ as pyo

# --- Tunables ---
allow_export = False         # True = lov å selge til nettet, False = kun import
sell_price_factor = 0.0      # 0.0 = ingen godtgjørelse, 1.0 = samme som kjøpspris
initial_soc_frac = 0.5       # start-SOC som andel av kapasitet
final_soc_equal_start = True # håndhev at slutt-SOC = start-SOC
bigM = max(charging_power_limit, discharging_power_limit)

# Lag Pyomo-sett og parametre
T = list(Hours)  # antas å være på 1h steg og sortert
model = pyoConcrete = pyo.ConcreteModel()
model.T = pyo.Set(initialize=T, ordered=True)

# Parametre
model.price = pyo.Param(model.T, initialize=dict_Prices)
model.base  = pyo.Param(model.T, initialize=dict_Base_load)
model.pv    = pyo.Param(model.T, initialize=dict_PV_prod)

# Vars
model.P_ch   = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, charging_power_limit))
model.P_dis  = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, discharging_power_limit))
model.SOC    = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, capacity))
model.P_imp  = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # grid import
model.P_exp  = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # grid export

# Binære for «ikke samtidig lading/utlading»
model.y_ch = pyo.Var(model.T, domain=pyo.Binary)
def no_simul_rule(m, t):
    return (m.P_ch[t] <= bigM*m.y_ch[t], m.P_dis[t] <= bigM*(1 - m.y_ch[t]))
model.no_simultaneous = pyo.ConstraintList()
for t in model.T:
    c1, c2 = no_simul_rule(model, t)
    model.no_simultaneous.add(c1)
    model.no_simultaneous.add(c2)

# Forby eksport hvis ikke tillatt
if not allow_export:
    model.no_export = pyo.Constraint(model.T, rule=lambda m,t: m.P_exp[t] == 0)

# Effektbalanse hver time: last = PV + utlad + import - lad - eksport
model.balance = pyo.Constraint(
    model.T, rule=lambda m, t: m.base[t] == m.pv[t] + m.P_dis[t] + m.P_imp[t] - m.P_ch[t] - m.P_exp[t]
)

# SOC-dynamikk (1h steg): SOC_t = SOC_{t-1} + η_ch*P_ch - (1/η_dis)*P_dis
times = model.T.ordered_data()
t0 = times[0]
soc0 = initial_soc_frac * capacity
model.soc_init = pyo.Constraint(expr=model.SOC[t0] == soc0)

def soc_rule(m, t):
    if t == t0:
        return pyo.Constraint.Skip
    t_prev = times[times.index(t)-1]
    return m.SOC[t] == m.SOC[t_prev] + charging_efficiency*m.P_ch[t] - (1.0/discharging_efficiency)*m.P_dis[t]
model.soc_dyn = pyo.Constraint(model.T, rule=soc_rule)

# Slutt-SOC
if final_soc_equal_start:
    model.soc_final = pyo.Constraint(expr=model.SOC[times[-1]] == soc0)

# Kostnadsfunksjon
sell_price = {t: sell_price_factor*dict_Prices[t] for t in T}
model.sell_price = pyo.Param(model.T, initialize=sell_price)
model.cost = pyo.Objective(
    expr=sum(model.price[t]*model.P_imp[t] - model.sell_price[t]*model.P_exp[t] for t in model.T),
    sense=pyo.minimize
)

# Løs
solver = SolverFactory('gurobi')  # bytt til 'highs', 'cbc', 'gurobi' om tilgjengelig
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
})
total_cost = res_df['Import'].dot(res_df['Price']) - res_df['Export'].dot(res_df['Price']*sell_price_factor)
print(f"Total cost: {total_cost:.2f} (currency units)")

import matplotlib.pyplot as plt

# Bruk res_df fra modellen (antar at det finnes allerede)
# For sikkerhet, lager vi et lite eksempel dersom res_df ikke er definert:
try:
    res_df
except NameError:
    import numpy as np
    import pandas as pd
    H = np.arange(24)
    res_df = pd.DataFrame({
        "Hour": H,
        "Load": 2 + np.sin(H/3),
        "PV": np.maximum(0, 3*np.sin((H-6)/6)),
        "P_ch": np.zeros(24),
        "P_dis": np.zeros(24),
        "SOC": np.linspace(1, 2, 24),
        "Import": np.random.rand(24),
        "Export": np.zeros(24),
        "Price": 0.5 + 0.2*np.sin(H/4)
    })

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
axs = axs.flatten()

# 1. Last og PV
axs[0].plot(res_df["Hour"], res_df["Load"], label="Load")
axs[0].plot(res_df["Hour"], res_df["PV"], label="PV production")
axs[0].set_title("Load & PV")
axs[0].set_ylabel("Power [kW]")
axs[0].legend(); axs[0].grid(True)

# 2. Batteriets lading/utlading
axs[1].step(res_df["Hour"], res_df["P_ch"], label="Charging", where="mid")
axs[1].step(res_df["Hour"], res_df["P_dis"], label="Discharging", where="mid")
axs[1].set_title("Battery charge/discharge")
axs[1].set_ylabel("Power [kW]")
axs[1].legend(); axs[1].grid(True)

# 3. SOC
axs[2].plot(res_df["Hour"], res_df["SOC"], marker="o")
axs[2].set_title("State of Charge (SOC)")
axs[2].set_ylabel("Energy [kWh]")
axs[2].grid(True)

# 4. Net import/export
axs[3].step(res_df["Hour"], res_df["Import"], label="Import", where="mid")
axs[3].step(res_df["Hour"], res_df["Export"], label="Export", where="mid")
axs[3].set_title("Grid import/export")
axs[3].set_ylabel("Power [kW]")
axs[3].legend(); axs[3].grid(True)

# 5. Pris og drift sammen
axs[4].plot(res_df["Hour"], res_df["Price"], "k-", label="Price")
axs[4].step(res_df["Hour"], res_df["P_ch"], "b-", label="Charge", where="mid")
axs[4].step(res_df["Hour"], res_df["P_dis"], "r-", label="Discharge", where="mid")
axs[4].set_title("Price vs battery operation")
axs[4].set_ylabel("Power / Price")
axs[4].legend(); axs[4].grid(True)

# 6. Balansekontroll
balance = res_df["PV"] + res_df["P_dis"] + res_df["Import"] - res_df["P_ch"] - res_df["Export"]
axs[5].plot(res_df["Hour"], res_df["Load"], "k-", label="Load (given)")
axs[5].plot(res_df["Hour"], balance, "g--", label="Reconstructed balance")
axs[5].set_title("Load vs balance check")
axs[5].set_ylabel("Power [kW]")
axs[5].legend(); axs[5].grid(True)

# Felles akse og layout
for ax in axs:
    ax.set_xlabel("Hour")
plt.tight_layout()
plt.show()
