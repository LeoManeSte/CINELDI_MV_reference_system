# -*- coding: utf-8 -*-
"""
Flexibility in power grid operation and planning (TET4565/TET4575)
Oppgaver 2–10: Kompakt, komplett løsningskode
Forutsetter at datasettet og hjelpefiler ligger lokalt (se path_data_set).
"""

# %% Imports
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
import load_profiles as lp
import pandapower_read_csv as ppcsv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np

# %% Input / data
path_data_set = 'CINELDI_MV_reference_system_v_2023-03-06/'
f_load = os.path.join(path_data_set,'load_data_CINELDI_MV_reference_system.csv')
f_map  = os.path.join(path_data_set,'mapping_loads_to_CINELDI_MV_reference_grid.csv')
f_lines= os.path.join(path_data_set,'standard_overhead_line_types.csv')
f_rel  = os.path.join(path_data_set,'reldata_for_component_types.csv')
f_lp   = os.path.join(path_data_set,'CINELDI_MV_reference_system_load_point.csv')

bus_i_subset  = [90, 91, 92, 96]
P_lim         = 4.0                 # MW
scaling_factor= 10                  # skalerer lastene i området
length_km     = 20.0                # hovedmater
disc_rate     = 0.04
growth        = 0.03
years_hor     = 10
plan_hor = 20
invest_life = 40

# %% Les data
data_lines   = pd.read_csv(f_lines, delimiter=';').set_index('type')
data_comprel = pd.read_csv(f_rel, delimiter=';').set_index('main_type')
data_lp      = pd.read_csv(f_lp, delimiter=';').set_index('bus_i')
net = ppcsv.read_net_from_csv(path_data_set, baseMVA=10)

# %% Lastprofiler – bruk 28. feb (peakdag) som representativ (Oppg. 2 & 7)
repr_days = [31+28]  # 28. feb
profiles = lp.load_profiles(f_load)
rel_profiles = profiles.map_rel_load_profiles(f_map, repr_days)     # (24 rader)
load_ts = rel_profiles.mul(net.load['p_mw'])                        # MW
agg_area = (load_ts[bus_i_subset] * scaling_factor).sum(axis=1)     # 24t serie (MW)
P_max0 = float(agg_area.max())
print(f"\n[Oppg.2] Startverdi topp-last i området (dagens verdi): {P_max0:.3f} MW")

def step_peak_growth(P0, g=growth, Y=years_hor):
    y = np.arange(0, Y+1)
    return y, P0*(1+g)**y

def show_need_measure(P0, P_lim):
    y, P = step_peak_growth(P0)
    year_limit = np.argmax(P > P_lim)
    plt.figure(figsize=(8,4))
    plt.step(np.append(y,y[-1]+1), np.append(P,P[-1]), where='post', lw=2, label='Peak load growth')
    plt.axhline(P_lim, ls='--', color='r', label=f'Power transfer limit: {P_lim:.1f} MW')
    if P[year_limit] > P_lim:
        plt.scatter(year_limit, P[year_limit], color='r', zorder=5)
        plt.text(year_limit+0.1, P[year_limit]+0.05, f'Cross at time t = {year_limit}', color='r')
    plt.xlim(0, y[-1]); plt.xticks(range(0,y[-1]))
    plt.xlabel('t (time)'); plt.ylabel('Annual peak load (MW)'); plt.title('Oppg.2 – Peak load growth vs limit')
    plt.grid(True, ls=':'); plt.legend(); plt.tight_layout(); plt.show()
    print(f"[Oppg.2] Grense {P_lim:.1f} MW, overskrides ved år {year_limit} ⇒ tiltak må være på plass ved start av år 2 (y=1).")

def inv_cost_A(lines_df, new_type, L_km):
    c = float(lines_df.loc[new_type, 'cost_NOK_per_km'])
    cost = L_km*c
    print(f"\n[Oppg.3] Ny linje: {new_type} | Kostnad: {c:,.0f} NOK/km | Lengde: {L_km} km")
    print(f"[Oppg.3] Investeringskostnad Alt. A: {cost:,.0f} NOK")
    return cost

def pv(cost, r=disc_rate, t=1):
    pv_ = cost/((1+r)**t)
    print(f"\n[Oppg.4] Nåverdi (r={r*100:.1f}%, år {t}): {pv_:,.0f} NOK")
    return pv_


sns.set_style("whitegrid")

def pv_corrected(cost, r=0.04, t=1, life=40, horiz=20):
    RV = cost * (1 - (horiz - t) / life)
    print(f"\n[Oppg.5] Residualverdi etter {horiz} år: {RV:,.0f} NOK")
    
    PV_inv = cost / ((1 + r) ** t)
    PV_res = RV / ((1 + r) ** horiz)
    PV_corr = PV_inv - PV_res
    
    years = np.arange(t, t + life + 1)
    values = cost * (1 - (years - t) / life)
    values[values < 0] = 0

    t_end = t + life

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(years, values, color='#1f77b4', linewidth=2.2)

    ax.axvline(t, color='black', linestyle=':', linewidth=1)
    ax.scatter([t], [cost], color='black', s=60, zorder=5)

    ax.axvline(horiz, color='red', linestyle='--', linewidth=1)
    ax.scatter([horiz], [RV], color='green', s=60, zorder=5)

    ax.axvline(t_end, color='gray', linestyle='-.', linewidth=1)
    ax.scatter([t_end], [0], color='gray', s=50, zorder=5)

    ax.axhline(RV, color='green', linestyle='--', linewidth=1)

    ax.text(t + 0.5, cost * 1.02, f"Investment\n{cost:,.0f} NOK",
            ha='left', va='bottom', fontsize=10, color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.text(horiz - 3, RV * 1.05, f"Residual value\n{RV:,.0f} NOK",
            ha='left', va='bottom', fontsize=10, color='green',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.text(t_end - 4, cost * 0.05, f"End of life\n(t = {t_end})",
            ha='left', va='bottom', fontsize=10, color='gray',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    ax.set_xlabel('t (time)', fontsize=12)
    ax.set_ylabel('Value (NOK)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.show()

    return PV_corr, PV_inv, PV_res

def compare_with_battery(P0, P_lim, g=growth, Y=years_hor, batt_shift_MW=1.0):
    y = np.arange(0,Y+1)
    P_no = P0*(1+g)**y
    P_bt = np.maximum(P_no - batt_shift_MW, 0)
    y_no = np.argmax(P_no>P_lim); y_bt = np.argmax(P_bt>P_lim)
    plt.figure(figsize=(8,4))
    plt.step(np.append(y,y[-1]+1), np.append(P_no,P_no[-1]), where='post', lw=2, label='Without battery (Alt.A)')
    
    plt.step(np.append(y,y[-1]+1), np.append(P_bt,P_bt[-1]), where='post', lw=2, ls='--', label=f'With {batt_shift_MW:.0f} MW battery (Alt.B)', color='C0')
    plt.axhline(P_lim, ls='--', color='r', label=f'Power transfer limit: {P_lim:.1f} MW')
  
    if P_bt[y_bt] > P_lim:
        plt.scatter(y_bt, P_bt[y_bt], color='r', zorder=5)
        plt.text(y_bt, P_bt[y_bt]+0.05, f'Cross at time t = {y_bt}', color='r')


    plt.annotate('',
                 xy=(y[1], P_no[1]-batt_shift_MW),
                 xytext=(y[1], P_no[1]),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))
    plt.text(y[1]+0.1, P_no[1]-batt_shift_MW/2, f'{batt_shift_MW:.0f} MW battery', color='black', va='center')


    plt.xticks(range(0,y[-1]))
    plt.xlim(0, y[-1])
    plt.xlabel('t (time)'); plt.ylabel('Annual peak load (MW)'); #plt.title('Oppg.6 Topp-last: uten vs med batteri')
    plt.grid(True, ls=':'); plt.legend(); plt.tight_layout(); plt.show()
    return y_no, y_bt


def battery_opex_by_year(
    daily_24h: pd.Series,
    P_lim: float = 4.0,
    g: float = 0.0,
    batt_MW: float = 1.0,
    nok_per_MWh: float = 2000.0,
    days: int = 20,
    Y: int = 10,
    reinforce_at: int = 9,
    disc_rate: float = 0.05,
    excel_file: str = "battery_shifted.xlsx"
):

    base = daily_24h.values
    hourly_index = daily_24h.index

    all_data = pd.DataFrame({'Hour': hourly_index})

    annual_MWh_list = []
    annual_NOK_list = []

    for y in range(Y):
        col_name = f'Year_{y+1}'
        if y >= reinforce_at:
            shifted = np.zeros_like(base)
            daily_MWh = 0.0
        else:
            P = base * (1 + g) ** y
            shifted = np.minimum(np.maximum(P - P_lim, 0), batt_MW)
            daily_MWh = shifted.sum()

        annual_MWh = daily_MWh * days
        annual_NOK = annual_MWh * nok_per_MWh

        all_data[col_name] = shifted
        annual_MWh_list.append(annual_MWh)
        annual_NOK_list.append(annual_NOK)

    daily_sum_row = ['Daily_sum'] + [all_data[col].sum() for col in all_data.columns[1:]]
    annual_sum_row = ['Annual_sum'] + annual_MWh_list
    total_sum_val = sum(annual_MWh_list)
    total_sum_row = ['Total_all_years'] + [total_sum_val] + [''] * (len(annual_MWh_list) - 1)

    summary_df = pd.DataFrame([daily_sum_row, annual_sum_row, total_sum_row], columns=all_data.columns)
    final_df = pd.concat([all_data, summary_df], ignore_index=True)

    df_cost = pd.DataFrame({
        'Year': np.arange(1, Y + 1),
        'Annual_shifted_MWh': annual_MWh_list,
        'Annual_cost_NOK': annual_NOK_list
    })
    df_cost['PV_Annual_cost'] = df_cost['Annual_cost_NOK'] / ((1 + disc_rate) ** (df_cost['Year'] - 1))

    
    total_cost_row = pd.DataFrame({
        'Year': ['TOTAL'],
        'Annual_shifted_MWh': [sum(annual_MWh_list)],
        'Annual_cost_NOK': [sum(annual_NOK_list)],
        'PV_Annual_cost': [df_cost['PV_Annual_cost'].sum()]
    })
    df_cost = pd.concat([df_cost, total_cost_row], ignore_index=True)

   
    print("\n[Oppg.7] Årlige driftskostnader (Alt.B, 20 like dager/år):")
    print(
        df_cost.to_string(
            index=False,
            formatters={
                'Annual_cost_NOK': '{:,.0f}'.format,
                'PV_Annual_cost': '{:,.0f}'.format
            }
        )
    )

  
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        final_df.to_excel(writer, sheet_name='Shifted_matrix', index=False)
        df_cost.to_excel(writer, sheet_name='Annual_summary', index=False)

    print(f"\nFerdig! All data skrevet til: {excel_file}")
    return df_cost


def eens_altA(comp_df, main_type_like='Overhead line', L_km=20.0, unit='per_100_km_year',
              avg_load_year1_mw=1.841, g=growth, Y=years_hor):
    idx = [i for i in comp_df.index if main_type_like in i]
    if not idx: raise ValueError("Fant ikke linjetype for 1–22 kV.")
    row = comp_df.loc[idx[0]]
    lam = float(row['lambda_perm']); r_h = float(row['r_perm'])
    lam_tot = lam*(L_km/100.0) if unit=='per_100_km_year' else lam*L_km
    years = np.arange(1, Y+1)
    Pavg = avg_load_year1_mw*(1+g)**(years-1)
    EENS = Pavg * (lam_tot*r_h)     # MWh/år
    df = pd.DataFrame({'Year':years,'Avg_load_MW':Pavg,'lambda_tot':lam_tot,'r_perm_h':r_h,'EENS_MWh_per_year':EENS})
    print("\n[Oppg.8] Årlig EENS (Alt.A):")
    print(df[['Year','EENS_MWh_per_year']].round(3).to_string(index=False))
    return df

def cens_A(loadpoint_df, buses, df_eens):
    # bruk c_NOK_per_kWh_4h (nærmest 3–4h antagelse)
    cvals = loadpoint_df.loc[buses,'c_NOK_per_kWh_4h'].astype(str).str.replace(',','.', regex=False).astype(float)
    print(cvals)
    c4h = float(cvals.mean())
    CENS = df_eens['EENS_MWh_per_year']*1000.0*c4h
    out = df_eens.copy()
    out['c_4h_NOK_per_kWh']=c4h; out['CENS_NOK_per_year']=CENS
    out['PV_CENS_NOK']= CENS/((1+disc_rate)**(out['Year']-1))
    print(f"\n[Oppg.9] CENS Alt.A med gj.sn. 4h-kost {c4h:.2f} NOK/kWh:")
    print(out[['Year','EENS_MWh_per_year','CENS_NOK_per_year','PV_CENS_NOK']].to_string(index=False,
          formatters={'CENS_NOK_per_year':'{:,.0f}'.format,'PV_CENS_NOK':'{:,.0f}'.format}))
    return out

def cens_B_with_battery(df_eens, loadpoint_df, buses, batt_P_MW=1.0, batt_E_MWh=2.0):
    cvals = loadpoint_df.loc[buses,'c_NOK_per_kWh_4h'].astype(str).str.replace(',','.', regex=False).astype(float)
    c4h = float(cvals.mean())
    P_sup = np.minimum(df_eens['Avg_load_MW'].values, batt_P_MW)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_energy = np.where(P_sup>0, batt_E_MWh/P_sup, 0.0)
    t_batt = np.minimum(df_eens['r_perm_h'].values, t_energy)
    E_per_fault = P_sup * t_batt
    print(E_per_fault)                    # MWh/feil
    E_per_year  = df_eens['lambda_tot'].values * E_per_fault
    #E_per_year  = E_per_fault
    EENS_no     = df_eens['EENS_MWh_per_year'].values
    EENS_with   = np.maximum(EENS_no - E_per_year, 0.0)
    CENS_B      = EENS_with*1000.0*c4h
    PV_CENS_B   = CENS_B/((1+disc_rate)**(df_eens['Year'].values-1))
    out = pd.DataFrame({
        'Year': df_eens['Year'].values,
        'EENS_no_batt_MWh_per_year': EENS_no,
        'EENS_with_batt_MWh_per_year': EENS_with,
        'CENS_with_batt_NOK_per_year': CENS_B,
        'PV_CENS_with_batt_NOK': PV_CENS_B
    })
    print("\n[Oppg.10] CENS Alt.B (batteri 1 MW / 2 MWh):")
    print(out.to_string(index=False, formatters={
        'CENS_with_batt_NOK_per_year':'{:,.0f}'.format,
        'PV_CENS_with_batt_NOK':'{:,.0f}'.format
    }))
    return out

# ---------- Oppg. 12 og 13: Socio-økonomisk total PV for Alt. A og Alt. B ----------

def pv_year_table_altA(
    df_cens_A: pd.DataFrame,
    PVcorr_A: float,          # corrected PV of investment (incl. residual), from earlier calc
    years: int = 10,
    invest_year: int = 2,     # investment at beginning of year 2 → we place PV at row "2"
):
    """
    10-year table in PV for Alternative A.
    Columns are PV cash flows as-of year 0:
      Year | Investment (PV, corrected incl. residual) | OPEX (PV) | Interruption (PV)
    """
    idx = pd.Index(range(1, years+1), name="Year")

    invest_pv = pd.Series(0.0, index=idx)
    if 1 <= invest_year <= years:
        invest_pv.loc[invest_year] = float(PVcorr_A)  # already corrected for residual value

    opex_pv = pd.Series(0.0, index=idx)              # no flexibility OPEX in Alt. A

    cens_pv = (
        df_cens_A.set_index("Year")["PV_CENS_NOK"]    # PV of interruption costs per year
        .reindex(idx, fill_value=0.0)
        .astype(float)
    )

    table = pd.DataFrame({
        "Year": idx,
        "Investment (PV) [NOK]": invest_pv.values,
        "Operational (PV) [NOK]": opex_pv.values,
        "Interruption (PV) [NOK]": cens_pv.values,
    })

    # Totals (PV)
    total_pv = table[["Investment (PV) [NOK]", "Operational (PV) [NOK]", "Interruption (PV) [NOK]"]].sum()
    total_row = pd.DataFrame([{
        "Year": "TOTAL (PV)",
        "Investment (PV) [NOK]": total_pv["Investment (PV) [NOK]"],
        "Operational (PV) [NOK]": total_pv["Operational (PV) [NOK]"],
        "Interruption (PV) [NOK]": total_pv["Interruption (PV) [NOK]"],
    }])
    table_out = pd.concat([table, total_row], ignore_index=True)

    print("\n=== 10-year PV table — Alternative A (investment PV includes residual) ===")
    print(table_out.to_string(index=False, formatters={
        "Investment (PV) [NOK]": "{:,.0f}".format,
        "Operational (PV) [NOK]": "{:,.0f}".format,
        "Interruption (PV) [NOK]": "{:,.0f}".format,
    }))

    total_cost_altA = total_pv.sum()

    return table_out, total_cost_altA


def pv_year_table_altB(
    df_cens_B: pd.DataFrame,
    opex_B: pd.DataFrame,
    PVcorr_B: float,          
    years: int = 10,
    invest_year: int = 10,  
):

    idx = pd.Index(range(1, years+1), name="Year")

    invest_pv = pd.Series(0.0, index=idx)
    if 1 <= invest_year <= years:
        invest_pv.loc[invest_year] = float(PVcorr_B) 

    
    opex_pv = (
        opex_B.set_index("Year")["PV_Annual_cost"]
        .reindex(idx, fill_value=0.0)
        .astype(float)
    )

    cens_pv = (
        df_cens_B.set_index("Year")["PV_CENS_with_batt_NOK"]
        .reindex(idx, fill_value=0.0)
        .astype(float)
    )

    table = pd.DataFrame({
        "Year": idx,
        "Investment (PV) [NOK]": invest_pv.values,
        "Operational (PV) [NOK]": opex_pv.values,
        "Interruption (PV) [NOK]": cens_pv.values,
    })

   
    total_pv = table[["Investment (PV) [NOK]", "Operational (PV) [NOK]", "Interruption (PV) [NOK]"]].sum()
    total_row = pd.DataFrame([{
        "Year": "TOTAL (PV)",
        "Investment (PV) [NOK]": total_pv["Investment (PV) [NOK]"],
        "Operational (PV) [NOK]": total_pv["Operational (PV) [NOK]"],
        "Interruption (PV) [NOK]": total_pv["Interruption (PV) [NOK]"],
    }])
    table_out = pd.concat([table, total_row], ignore_index=True)

    print("\n=== 10-year PV table — Alternative B (investment PV includes residual) ===")
    print(table_out.to_string(index=False, formatters={
        "Investment (PV) [NOK]": "{:,.0f}".format,
        "Operational (PV) [NOK]": "{:,.0f}".format,
        "Interruption (PV) [NOK]": "{:,.0f}".format,
    }))

    total_cost_altB = total_pv.sum()

    return table_out, total_cost_altB

def plot_cumulative_pv_costs_altA_altB(table_A_pv: pd.DataFrame, table_B_pv: pd.DataFrame,
                                       years: int = 10):

    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    def per_year_pv(df, years):
        tmp = df.copy()
        tmp["Year_num"] = pd.to_numeric(tmp["Year"], errors="coerce")
        tmp = tmp.dropna(subset=["Year_num"])
        tmp = tmp[tmp["Year_num"].between(1, years)].sort_values("Year_num")
        tmp["t"] = tmp["Year_num"] - 1 
        cols = ["Investment (PV) [NOK]", "Operational (PV) [NOK]", "Interruption (PV) [NOK]"]
        tmp["PV_cost_year"] = tmp[cols].sum(axis=1)
        x = tmp["t"].astype(int).values
        y = tmp["PV_cost_year"].astype(float).values
        return x, y

    def step_xy(x, y):
        x_step = np.append(x, x[-1] + 1)
        y_step = np.append(y, y[-1] if len(y) else 0.0)
        return x_step, y_step

    xA, yA = per_year_pv(table_A_pv, years)
    xB, yB = per_year_pv(table_B_pv, years)
    yA_cum = np.cumsum(yA)
    yB_cum = np.cumsum(yB)
    xA_s, yA_s = step_xy(xA, yA_cum)
    xB_s, yB_s = step_xy(xB, yB_cum)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.step(xA_s, yA_s, where="post", lw=2.5, label="Alternative A", color="#1f77b4")
    ax.step(xB_s, yB_s, where="post", lw=2.5, ls="--", label="Alternative B", color="#ff7f0e")

    ax.set_xlim(0, years)
    ax.set_xlabel("t [time]")
    ax.set_ylabel("Cumulative PV cost [NOK]")
    ax.legend(frameon=True, loc="upper left")
    sns.despine()

    ymin = 0
    ymax = max(yA_cum[-1], yB_cum[-1]) * 1.1
    ax.set_ylim(ymin, ymax)

    ax.set_xticks(range(0, years))
    ax.set_xticklabels([str(i) for i in range(0, years)])

    shift_x = years - 0.6 
    box_props = dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8)

    ax.text(shift_x, yA_cum[-1],
            f"{yA_cum[-1]:,.0f} NOK",
            va="center", ha="left",
            fontsize=12, color="#1f77b4",
            bbox=box_props)

    ax.text(shift_x, yB_cum[-1],
            f"{yB_cum[-1]:,.0f} NOK",
            va="center", ha="left",
            fontsize=12, color="#ff7f0e",
            bbox=box_props)

    diff = abs(yA_cum[-1] - yB_cum[-1])
    y_min = min(yA_cum[-1], yB_cum[-1])
    y_max = max(yA_cum[-1], yB_cum[-1])

    ax.annotate(
        "",
        xy=(shift_x, y_max),
        xytext=(shift_x, y_min),
        arrowprops=dict(arrowstyle="<->", color="gray", lw=1.8)
    )

    ax.text(
        shift_x + 0.25,
        (y_min + y_max) / 2,
        f"Δ {diff:,.0f} NOK",
        va="center",
        ha="left",
        color="black",
        fontsize=13,
        fontweight="bold",
        bbox=box_props
    )

    plt.tight_layout()
    plt.show()


show_need_measure(P_max0, P_lim)


type_FeAl70 = '111-AL1/19-ST1A (FeAl nr. 70 6/1)'
cost_A = inv_cost_A(data_lines, type_FeAl70, length_km)

PV_A = pv(cost_A, disc_rate, t=1)


PVcorr_A, PVinv_A, PVres_A = pv_corrected(cost_A, disc_rate, t=1, life=invest_life, horiz=plan_hor)
print(f"\n[Oppg.5] Restverdi etter {plan_hor} år: {PVres_A:,.0f} NOK | PV(inv): {PVinv_A:,.0f}")
print(f"[Oppg.5] Korrigert nåverdi: {PVcorr_A:,.0f} NOK")

y_no, y_bt = compare_with_battery(P_max0, P_lim, g=growth, Y=years_hor, batt_shift_MW=1.0)
print(f"\n[Oppg.6] Uten batteri: grense passeres ved år {y_no}. Med batteri: ved år {y_bt} ⇒ forsterkning utsettes til start år 10 (y=9).")

PVcorr_B, PVinv_B, PVres_B = pv_corrected(cost_A, disc_rate, t=9, life=40, horiz=20)
print(f"[Oppg.6] Restverdi etter {plan_hor} år: {PVres_B:,.0f} NOK | PV(inv): {PVinv_B:,.0f}")
print(f"[Oppg.6] Korrigert nåverdi: {PVcorr_B:,.0f} NOK")
print(f"[Oppg.6] Reduksjon i korrigert nåverdi ved utsettelse: {(PVcorr_A - PVcorr_B):,.0f} NOK")

opex_B = battery_opex_by_year(agg_area, P_lim=P_lim, g=growth, batt_MW=1.0,
                              nok_per_MWh=2000.0, days=20, Y=years_hor, reinforce_at=9)

df_eens_A = eens_altA(data_comprel, main_type_like='Overhead line', L_km=length_km,
                      unit='per_100_km_year', avg_load_year1_mw=1.841, g=growth, Y=years_hor)

df_cens_A = cens_A(data_lp, bus_i_subset, df_eens_A)

df_cens_B = cens_B_with_battery(df_eens_A, data_lp, bus_i_subset, batt_P_MW=1.0, batt_E_MWh=2.0)

print("\n=== Oppg. 12 og 13: Socio-økonomisk total PV for Alt. A og Alt. B ===")

table_A_pv, total_cost_altA = pv_year_table_altA(
    df_cens_A=df_cens_A,
    PVcorr_A=PVcorr_A,       
    years=10,
    invest_year=2
)

table_B_pv, total_cost_altB = pv_year_table_altB(
    df_cens_B=df_cens_B,
    opex_B=opex_B,
    PVcorr_B=PVcorr_B,        
    years=10,
    invest_year=10
)

print(f"\n[Oppg.12] Total PV kost Alt.A: {total_cost_altA:,.0f} NOK")
print(f"[Oppg.12] Total PV kost Alt.B: {total_cost_altB:,.0f} NOK")


plot_cumulative_pv_costs_altA_altB(table_A_pv, table_B_pv, years=10)
