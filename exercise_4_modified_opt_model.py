# -*- coding: utf-8 -*-
"""
Oppgave 14 – Modifisert Ex3-modell for detaljert batteridrift i område

Forutsetninger:
- PV = 0
- Batteri: 1 MW / 2 MWh
- Importgrense P_lim = 4 MW
- Maksimerer inntekter (prisarbitrasje) med priser fra Exercise 3 (profile_input.csv)
- Last: aggregert (bus_i 90, 91, 92, 96) for 28. feb, skalert til år 6 (y=5)
"""

from __future__ import annotations

# ----------------------------- Imports -----------------------------
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import load_profiles as lp
import pandapower_read_csv as ppcsv

# ----------------------------- Konstanter / I/O -----------------------------
# Filer og stier
PATH_DATASET = "CINELDI_MV_reference_system_v_2023-03-06/"
FILE_LOAD = os.path.join(PATH_DATASET, "load_data_CINELDI_MV_reference_system.csv")
FILE_MAP = os.path.join(PATH_DATASET, "mapping_loads_to_CINELDI_MV_reference_grid.csv")
FILE_BATT = "./battery_data.csv"
FILE_EX3 = "./profile_input.csv"

# Områdedefinisjon og skaleringsfaktorer
BUS_I_SUBSET = [90, 91, 92, 96]
REPR_DAYS = [31 + 28]     # 28. feb (ikke-skuddår)
GROWTH = 0.03
YEAR_START_Y = 6          # skaler til år 6 ⇒ (1+g)^5
SCALE_TO_YEAR6 = (1.0 + GROWTH) ** YEAR_START_Y
SCALING_FACTOR = 10.0     # ekstra skaleringsfaktor for området (som i originalkode)

# Batteri- og nettparametre
CAPACITY_MWH = 4.0
P_CH_MAX_MW = 2.0
P_DIS_MAX_MW = 2.0
P_LIM = 4.0               # MW importgrense
DT_H = 1.0                 # tidssteg i timer
SELL_PRICE_FACTOR = 1.0
INITIAL_SOC_FRAC = 0.3
FINAL_SOC_EQUAL_START = False

# ----------------------------- Hjelpefunksjoner -----------------------------
def read_inputs() -> Tuple[
    pd.Series, np.ndarray, np.ndarray, List[int]
]:
    """Leser batteriparametre, priser og aggregerer område-last for 28. feb."""
    # Batteriparametre (virkningsgrader)
    parameters = pd.read_csv(FILE_BATT, index_col=0).loc[1]

    # Priser (Exercise 3)
    ex3 = pd.read_csv(FILE_EX3)
    hours_price = ex3["Hours"].values.astype(int)     # forventet 1..24
    price = ex3["Price"].values.astype(float)

    # CINELDI: rel. profiler for 28. feb, mappet og skalert
    net = ppcsv.read_net_from_csv(PATH_DATASET, baseMVA=10)
    profiles = lp.load_profiles(FILE_LOAD)
    rel_profiles = profiles.map_rel_load_profiles(FILE_MAP, REPR_DAYS)  # [24 x n_loads]

    load_ts_MW = rel_profiles.mul(net.load["p_mw"])  # [24 x n_loads] MW
    agg_area_MW = (load_ts_MW[BUS_I_SUBSET] * SCALING_FACTOR).sum(axis=1).values
    agg_area_year6_MW = SCALE_TO_YEAR6 * agg_area_MW

    # Konsistenskontroll
    if len(agg_area_year6_MW) != len(hours_price):
        raise ValueError("Ulik lengde mellom last og pris – sjekk inndata!")

    hours = list(hours_price)
    return parameters, agg_area_year6_MW, price, hours


def build_data_dicts(
    load_MW: np.ndarray, price: np.ndarray, hours: List[int]
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], List[int]]:
    """Bygger dicts for Pyomo-parametre."""
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
) -> pyo.ConcreteModel:
    """Setter opp Pyomo-modellen."""
    # Big-M for eksport
    Mbig = 10.0 * max(1.0, float(np.max(list(dict_load.values()))))
    SoC0 = INITIAL_SOC_FRAC * CAPACITY_MWH

    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=list(dict_load.keys()), ordered=True)

    # Parametre
    model.price = pyo.Param(model.T, initialize=dict_prices)
    model.load_MW = pyo.Param(model.T, initialize=dict_load)
    model.pv_MW = pyo.Param(model.T, initialize=dict_pv)
    model.sell_price = pyo.Param(model.T, initialize=dict_sell)

    # Variabler
    model.P_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, P_CH_MAX_MW))
    model.P_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, P_DIS_MAX_MW))
    model.SOC = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, CAPACITY_MWH))
    model.P_imp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, P_LIM))
    model.P_exp = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0.0, Mbig))

    # Binære: nett-retning (1=import, 0=eksport), batteri-retning (1=lad, 0=disch)
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

    # Effektbalanse (MW): Load = PV + Dis + Import - Charge - Export
    def balance_rule(m, t):
        return m.load_MW[t] == m.pv_MW[t] + m.P_dis[t] + m.P_imp[t] - m.P_ch[t] - m.P_exp[t]
    model.balance = pyo.Constraint(model.T, rule=balance_rule)

    # SoC-dynamikk
    times = model.T.ordered_data()
    t0 = times[0]
    model.soc_init = pyo.Constraint(expr=model.SOC[t0] == SoC0)

    def soc_rule(m, t):
        #if t == t0:
        #    return pyo.Constraint.Skip
        t_prev = times[times.index(t) - 1]
        return m.SOC[t] == (
            m.SOC[t_prev] + eta_ch * m.P_ch[t] * DT_H - (1.0 / eta_dis) * m.P_dis[t] * DT_H
        )
    model.soc_dyn = pyo.Constraint(model.T, rule=soc_rule)

    # Energi-begrensninger per time
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

    # Slutt-SOC = start-SOC (ren arbitrasje)
    if FINAL_SOC_EQUAL_START:
        model.soc_final = pyo.Constraint(expr=model.SOC[times[-1]] == SoC0)

    # Objektiv: maksimer inntekt (salg – kjøp)
    model.obj = pyo.Objective(
        expr=sum(model.sell_price[t] * model.P_exp[t] - model.price[t] * model.P_imp[t] for t in model.T),
        sense=pyo.maximize,
    )
    return model


def solve_model(model: pyo.ConcreteModel) -> Tuple[str, pyo.SolverResults]:
    """Løser modellen med tilgjengelig solver."""
    solver_name, solver = choose_solver()
    print(f"Using solver: {solver_name}")
    res = solver.solve(model, tee=False)
    return solver_name, res


def extract_results(model: pyo.ConcreteModel, hours: List[int]) -> pd.DataFrame:
    """Ekstraherer resultater til DataFrame og gjør konsistenssjekker."""
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


def evaluate_and_print(res_df: pd.DataFrame) -> None:
    """Beregner nøkkeltall og skriver kort vurdering."""
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


def _stepify_1h(hours: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forbereder timeserier til step-plot (where='post') med korrekte endepunkter (0–24)."""
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


def plot_results(res_df: pd.DataFrame) -> None:
    """Lager et kompakt step-plot for net-load (uten/med batteri) og SoC."""
    # Farger (WCAG-vennlige)
    c_no = "#1f77b4"   # blå
    c_with = "#ff7f0e" # oransje
    c_soc = "#2ca02c"  # grønn
    c_plim = "#444444" # mørk grå

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

# ----------------------------- Helpers for step='post' segments -----------------------------
def _largest_violation_segment_intervals(y_interval: np.ndarray, plim: float) -> tuple[int, int, float]:
    """
    On hourly intervals y_interval[i] over [i, i+1], find the single largest contiguous
    segment with y > plim. Return (start, end_exclusive, area_MWh).
    """
    above = y_interval > plim
    if not np.any(above):
        return -1, -1, 0.0

    starts, ends, in_seg = [], [], False
    for i, a in enumerate(above):
        if a and not in_seg:
            in_seg = True
            starts.append(i)
        if not a and in_seg:
            in_seg = False
            ends.append(i)
    if in_seg:
        ends.append(len(above))

    best_idx, best_area = -1, -1.0
    for k, (s, e) in enumerate(zip(starts, ends)):
        area = float(np.sum((y_interval[s:e] - plim) * DT_H))
        if area > best_area:
            best_idx, best_area = k, area

    return starts[best_idx], ends[best_idx], best_area


def _initial_headroom_segment_intervals(y_interval: np.ndarray, plim: float) -> tuple[int, int, float]:
    """
    Initial contiguous segment from the start where y <= plim (i.e., before the first exceedance).
    Returns (0, end_exclusive, headroom_area_MWh). If the first hour already exceeds plim, area=0.
    """
    # find first index where y >= plim (cap reached or exceeded)
    k = 0
    n = len(y_interval)
    while k < n and y_interval[k] < plim - 1e-12:
        k += 1
    # segment is [0, k); if k==0, no headroom
    if k == 0:
        return 0, 0, 0.0
    area = float(np.sum((plim - y_interval[0:k]) * DT_H))
    return 0, k, area


def _segmented_step_series(y_interval: np.ndarray, s: int, e: int) -> np.ndarray:
    """
    Convert 24-interval series to step-post (length 25) and keep only [s,e).
    Crucial: set value at index e to close the last interval for fill_between with step='post'.
    """
    y_post = np.r_[y_interval, y_interval[-1]]
    y_seg = np.full_like(y_post, np.nan, dtype=float)
    if s >= 0 and e > s:
        y_seg[s:e] = y_post[s:e]
        y_seg[e] = y_post[e - 1]  # ensure right-edge vertex exists
    return y_seg


# ----------------------------- Plot: Largest overload + initial headroom (Year 6 vs Year 7) -----------------------------
def plot_overload_and_headroom_year6_vs_year7(res_df_year6: pd.DataFrame, hours: list[int]) -> None:
    """
    Two subplots (Year 6, Year 7) with the SAME step style as your main plot (where='post', 0–24 edges).
    - Blue step: net load without BESS
    - Dashed line: P_lim
    - Hatched dark fill: largest overload area (above P_lim)
    - Hatched light fill: initial charging headroom (below P_lim before first exceedance)
    Both areas are annotated with bold, boxed labels.
    """
    # Interval series (values apply on [i,i+1])
    y6_interval = res_df_year6["Net_load_no_batt_MW"].to_numpy(dtype=float)
    if len(y6_interval) != 24:
        raise ValueError("Expected 24 hourly values for a single day.")
    y7_interval = y6_interval * (1.0 + GROWTH)

    # Overload segments
    s6_hi, e6_hi, area6_hi = _largest_violation_segment_intervals(y6_interval, P_LIM)
    s7_hi, e7_hi, area7_hi = _largest_violation_segment_intervals(y7_interval, P_LIM)

    # Initial headroom segments
    s6_lo, e6_lo, area6_lo = _initial_headroom_segment_intervals(y6_interval, P_LIM)
    s7_lo, e7_lo, area7_lo = _initial_headroom_segment_intervals(y7_interval, P_LIM)

    # Prepare step-post (0..24)
    x_post = np.arange(0.0, 25.0, 1.0)
    y6_post = np.r_[y6_interval, y6_interval[-1]]
    y7_post = np.r_[y7_interval, y7_interval[-1]]
    y6_hi_post = _segmented_step_series(y6_interval, s6_hi, e6_hi)
    y7_hi_post = _segmented_step_series(y7_interval, s7_hi, e7_hi)
    y6_lo_post = _segmented_step_series(y6_interval, s6_lo, e6_lo)
    y7_lo_post = _segmented_step_series(y7_interval, s7_lo, e7_lo)

    # Colors to match your style
    c_no = "#1f77b4"   # blue
    c_plim = "#444444" # dark gray

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), sharey=True)

    for ax, y_post, y_hi_post, y_lo_post, area_hi, area_lo, title, s_hi, e_hi, s_lo, e_lo in [
        (axes[0], y6_post, y6_hi_post, y6_lo_post, area6_hi, area6_lo, "Year 6", s6_hi, e6_hi, s6_lo, e6_lo),
        (axes[1], y7_post, y7_hi_post, y7_lo_post, area7_hi, area7_lo, "Year 7", s7_hi, e7_hi, s7_lo, e7_lo),
    ]:
        # Main step line and cap
        ax.step(x_post, y_post, where="post", lw=2.6, color=c_no, label="Net load (no BESS)")
        ax.axhline(P_LIM, ls="--", lw=2.2, color=c_plim, label=f"P_lim = {P_LIM:g} MW")

        # Initial headroom fill (below cap) with a light hatch
        if area_lo > 0 and e_lo > s_lo:
            coll_lo = ax.fill_between(
                x_post,
                y_lo_post,
                P_LIM,
                where=~np.isnan(y_lo_post) & (y_lo_post < P_LIM),
                step="post",
                alpha=0.20,
                label="Initial charging headroom"
            )
            try:
                coll_lo.set_hatch("..")
            except Exception:
                pass
            # Label headroom prominently near its center
            x_center_lo = (s_lo + e_lo) / 2.0
            ax.text(
                x_center_lo, P_LIM,
                f"{area_lo:.2f} MWh",
                ha="center", va="top",
                fontsize=12, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25", alpha=0.95)
            )

        # Largest overload fill (above cap) with a distinct hatch
        if area_hi > 0 and e_hi > s_hi:
            coll_hi = ax.fill_between(
                x_post,
                P_LIM,
                y_hi_post,
                where=~np.isnan(y_hi_post) & (y_hi_post > P_LIM),
                step="post",
                alpha=0.25,
                label="Largest overload area"
            )
            try:
                coll_hi.set_hatch("////")
            except Exception:
                pass
            # Label overload prominently
            x_center_hi = (s_hi + e_hi) / 2.0
            ax.text(
                x_center_hi, P_LIM,
                f"{area_hi:.2f} MWh",
                ha="center", va="bottom",
                fontsize=12, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25", alpha=0.95)
            )

        # Axes formatting
        ax.set_xlim(0.0, 24.0)
        ax.set_xlabel("Hour")
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_title(title)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Power [MW]")
    plt.suptitle("Initial charging headroom and largest overload vs P_lim — Year 6 and Year 7", y=1.02)
    plt.tight_layout()
    plt.show()


# ----------------------------- Segment utilities (step='post', 0..24) -----------------------------
def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [(start,end_exclusive), ...] for contiguous True segments in a boolean array."""
    segs, in_seg, s = [], False, 0
    for i, a in enumerate(mask):
        if a and not in_seg:
            in_seg, s = True, i
        if not a and in_seg:
            in_seg = False
            segs.append((s, i))
    if in_seg:
        segs.append((s, len(mask)))
    return segs


def _step_post_series(y_interval: np.ndarray) -> np.ndarray:
    """Convert 24-interval array (applies on [i,i+1]) to step-post array length 25."""
    return np.r_[y_interval, y_interval[-1]]


def _segment_step_series(y_interval: np.ndarray, s: int, e: int) -> np.ndarray:
    """
    Keep only [s,e) of a 24-interval series in step-post (len 25), set the right-edge value at e
    so fill_between with step='post' closes the polygon.
    """
    y_post = _step_post_series(y_interval)
    out = np.full_like(y_post, np.nan, dtype=float)
    if e > s:
        out[s:e] = y_post[s:e]
        out[e] = y_post[e - 1]  # crucial for the final hour fill
    return out


def _areas_for_segments(y_interval: np.ndarray, plim: float, over: bool) -> list[tuple[int, int, float]]:
    """
    Return list of (s, e, area_MWh) for all segments above (over=True) or below (over=False) plim.
    """
    mask = (y_interval > plim) if over else (y_interval < plim)
    segs = _contiguous_segments(mask)
    areas = []
    for s, e in segs:
        diff = (y_interval[s:e] - plim) if over else (plim - y_interval[s:e])
        area = float(np.sum(diff * DT_H))
        if area > 1e-9:
            areas.append((s, e, area))
    return areas


# ----------------------------- Plot all over/under-limit areas (Year 6 vs Year 7) -----------------------------
def plot_all_areas_over_under_year6_vs_year7(res_df_year6: pd.DataFrame, hours: list[int]) -> None:
    """
    Two subplots (Year 6, Year 7), same style as your main figure:
    - Blue step: net load without BESS
    - Dashed line: P_lim
    - Every segment with P > P_lim hatched/filled and labeled with its area (MWh)
    - Every segment with P < P_lim hatched/filled (different hatch) and labeled with its area (MWh)
    """
    # Interval series for a single day
    y6 = res_df_year6["Net_load_no_batt_MW"].to_numpy(dtype=float)
    if len(y6) != 24:
        raise ValueError("Expected 24 hourly values for a single day.")
    y7 = y6 * (1.0 + GROWTH)

    x_post = np.arange(0.0, 25.0, 1.0)
    y6_post, y7_post = _step_post_series(y6), _step_post_series(y7)

    # Segment lists
    segs6_over = _areas_for_segments(y6, P_LIM, over=True)
    segs6_under = _areas_for_segments(y6, P_LIM, over=False)
    segs7_over = _areas_for_segments(y7, P_LIM, over=True)
    segs7_under = _areas_for_segments(y7, P_LIM, over=False)

    # Aesthetics to match your plots
    c_no = "#1f77b4"   # blue
    c_plim = "#444444" # dark gray

    # Hatching/styles for multiple segments
    hatch_over = ["////", "xxxx", "\\\\\\\\", "||||"]
    hatch_under = ["..", "++", "--", "oo"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)

    for ax, y_post, y_int, segs_over, segs_under, title in [
        (axes[0], y6_post, y6, segs6_over, segs6_under, "Year 6"),
        (axes[1], y7_post, y7, segs7_over, segs7_under, "Year 7"),
    ]:
        # Base lines
        ax.step(x_post, y_post, where="post", lw=2.6, color=c_no, label="Net load (no BESS)")
        ax.axhline(P_LIM, ls="--", lw=2.2, color=c_plim, label=f"P_lim = {P_LIM:g} MW")

        # Plot all UNDER-limit segments
        for idx, (s, e, area) in enumerate(segs_under):
            y_seg = _segment_step_series(y_int, s, e)
            coll = ax.fill_between(
                x_post, y_seg, P_LIM,
                where=~np.isnan(y_seg) & (y_seg < P_LIM),
                step="post", alpha=0.18,
                label="Under-limit area" if idx == 0 else None
            )
            try:
                coll.set_hatch(hatch_under[idx % len(hatch_under)])
            except Exception:
                pass
            # Label
            x_center = (s + e) / 2.0
            ax.text(
                x_center, P_LIM,
                f"{area:.2f} MWh",
                ha="center", va="top",
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25", alpha=0.95)
            )

        # Plot all OVER-limit segments
        for idx, (s, e, area) in enumerate(segs_over):
            y_seg = _segment_step_series(y_int, s, e)
            coll = ax.fill_between(
                x_post, P_LIM, y_seg,
                where=~np.isnan(y_seg) & (y_seg > P_LIM),
                step="post", alpha=0.25,
                label="Over-limit area" if idx == 0 else None
            )
            try:
                coll.set_hatch(hatch_over[idx % len(hatch_over)])
            except Exception:
                pass
            # Label
            x_center = (s + e) / 2.0
            ax.text(
                x_center, P_LIM,
                f"{area:.2f} MWh",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.25", alpha=0.95)
            )

        # Axes formatting
        ax.set_xlim(0.0, 24.0)
        ax.set_xlabel("Hour")
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_title(title)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("Power [MW]")
    plt.suptitle("All areas above and below P_lim — Year 6 vs Year 7", y=1.02)
    plt.tight_layout()
    plt.show()



# ----------------------------- Hovedflyt -----------------------------
def main() -> None:
    # 1) Inndata
    parameters, load_MW, price, hours = read_inputs()
    eta_ch = float(parameters["Charging_efficiency"])
    eta_dis = float(parameters["Discharging_efficiency"])

    # 2) Datadicts for Pyomo
    dict_load, dict_prices, dict_pv, dict_sell = build_data_dicts(load_MW, price, hours)
    #print(dict_load)

    # 3) Bygg og løs modell
    model = build_model(dict_load, dict_prices, dict_pv, dict_sell, eta_ch, eta_dis)
    _, _ = solve_model(model)

    # 4) Resultater
    res_df = extract_results(model, hours)

    # 5) Vurdering
    evaluate_and_print(res_df)

    # 6) Plot
    plot_results(res_df)

    plot_all_areas_over_under_year6_vs_year7(res_df, hours)

    plot_overload_and_headroom_year6_vs_year7(res_df, hours)



if __name__ == "__main__":
    main()

