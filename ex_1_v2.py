from math import exp
import matplotlib.pyplot as plt
import numpy as np

from math import exp
import matplotlib.pyplot as plt
import numpy as np


def make_load_profile_ewh(time_steps,P,T,S,T_a,C,R,T_min,T_max, P_m,t_act,S_act):
    """
    Generate load time series for electric water heater (EWH).
    """
    P_list = [P]
    T_list = [T]
    S_list = [S]

    for t in range(1,time_steps):        
        T_prev = T
        S_prev = S

        # Solve differential equation for the change of temperature for the next time step
        T = T_a - exp(-(1/60)/(C*R))*(T_a + P_m * R * S_prev  - T_prev) + P_m * R * S_prev 

        if (T <= T_min) & (S_prev == 0):
            # Turn EWH on if the temperature becomes too low
            S = 1
        elif (T > T_max) & (S_prev == 1):
            # Turn EWH off if the temperature becomes too high
            S = 0
        else:
            S = S_prev

        if (t == t_act) & (S_act is not None):
            # Activate flexibility
            S = S_act

        # The EWH operates at full power capacity if turned on
        P = S_prev * P_m

        P_list.append(P)
        T_list.append(T)
        S_list.append(S)

    return P_list, T_list, S_list

def simulate_ewh(
    P_m: float = 2.0,
    T_a: float = 24.0,
    T_min: float = 70.0,
    T_max: float = 75.0,
    C: float = 0.335,
    R: float = 600.0,
    T_init_single: float = 73,
    S_init: int = 0,
    P_init: float = 0.0,
    t_act: int = None,
    S_act: int = None,
    time_steps: int = 24*60,
    N_EWH: int = 1,
    seed: int = 42,
    plot: bool = True,
):
    """
    Run an Electric Water Heater (EWH) model for one or multiple units.
    """

    # --- Initialize starting temperatures ---
    if N_EWH == 1:
        T_init = [T_init_single]
    elif N_EWH > 1:
        rng = np.random.default_rng(seed=seed)
        T_init = rng.uniform(70, 75, N_EWH)
    else:
        raise ValueError("N_EWH must be >= 1")

    # --- Aggregated time series ---
    P_list_all = np.zeros(time_steps)
    P_list_base_all = np.zeros(time_steps)

    all_ewh = []  # store individual unit data for inspection

    # --- Run model for each heater ---
    S_list = None
    P_list = None

    for i_EWH in range(N_EWH):
        T0 = T_init[i_EWH]

        if i_EWH > 0 and S_list is not None and P_list is not None:
            S0 = S_list[-1]
            P0 = P_list[-1]
        else:
            S0 = S_init
            P0 = P_init

        # Baseline
        P_list_base, T_list_base, _ = make_load_profile_ewh(
            time_steps, P0, T0, S0, T_a, C, R, T_min, T_max, P_m, None, None
        )

        # With flexibility
        P_list, T_list, S_list = make_load_profile_ewh(
            time_steps, P0, T0, S0, T_a, C, R, T_min, T_max, P_m, t_act, S_act
        )

        # Aggregate results
        P_list_all += np.array(P_list)
        P_list_base_all += np.array(P_list_base)

        all_ewh.append({
            "P_base": np.array(P_list_base),
            "T_base": np.array(T_list_base),
            "P": np.array(P_list),
            "T": np.array(T_list),
            "S": np.array(S_list),
            "T0": T0,
            "S0": S0,
            "P0": P0,
        })

    # --- Plotting ---
    if plot:
        if N_EWH == 1:
            T_list_base = all_ewh[0]["T_base"]
            T_list = all_ewh[0]["T"]
            P_list_base = all_ewh[0]["P_base"]
            P_list = all_ewh[0]["P"]

            fig, ax1 = plt.subplots()
            if (t_act is not None) and (S_act is not None):
                h_T_base, = ax1.plot(T_list_base, 'r--')
            h_T, = ax1.plot(T_list, 'r')
            ax1.set_ylim(ymin=T_min * 0.95, ymax=T_max * 1.05)
            ax1.set_xlabel('Minutes')
            ax1.set_ylabel('Temperature (°C)', color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            if (t_act is not None) and (S_act is not None):
                ax1.legend([h_T_base, h_T], ['without flex.', 'with flex.'], loc='upper left')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Power (kW)', color='tab:blue')
            ax2.plot(P_list_base, color='tab:blue', linestyle='dashed')
            ax2.plot(P_list, color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            fig.tight_layout()
            plt.show()

        else:  # N_EWH > 1
            fig, ax1 = plt.subplots()
            ax1.set_ylim(ymin=0, ymax=P_list_base_all.max() * 1.05 if P_list_base_all.max() > 0 else 1)
            ax1.set_ylabel('Aggregated Power (kW)', color='tab:blue')
            if (t_act is not None) and (S_act is not None):
                h_P_base, = ax1.plot(P_list_base_all, color='tab:blue', linestyle='dashed')
            h_P, = ax1.plot(P_list_all, color='tab:blue')
            if (t_act is not None) and (S_act is not None):
                ax1.legend([h_P_base, h_P], ['without flex.', 'with flex.'], loc='upper left')
            plt.xlabel('Minutes')
            plt.show()

    result = {
        "P_agg": P_list_all,
        "P_agg_base": P_list_base_all,
        "per_ewh": all_ewh,
        "params": {
            "P_m": P_m, "T_a": T_a, "T_min": T_min, "T_max": T_max,
            "C": C, "R": R, "time_steps": time_steps, "N_EWH": N_EWH,
            "t_act": t_act, "S_act": S_act, "seed": seed
        }
    }
    return result

def analyze_shift(result):
    """
    Finn hvor lenge lasten er forskjøvet og beregn energien (kWh).
    Håndterer både tidligere og senere lastforskyvning.
    """

    P_base = result["per_ewh"][0]["P_base"]
    P_flex = result["per_ewh"][0]["P"]
    T_base = result["per_ewh"][0]["T_base"]
    T_flex = result["per_ewh"][0]["T"]
    t_act = result["params"]["t_act"]

    if t_act is None:
        print("Ingen aktivering satt.")
        return

    # --- Finn start/slutt for base og flex ---
    try:
        t_base_on = np.where(P_base[t_act:] > 0)[0][0] + t_act
        print(f"t_base_on: {t_base_on}")
    except IndexError:
        t_base_on = None

    try:
        t_base_off = np.where(P_base[t_act:] == 0)[0][0] + t_act
    except IndexError:
        t_base_off = len(P_base)

    try:
        t_flex_off = np.where(P_flex[t_act:] == 0)[0][0] + t_act
    except IndexError:
        t_flex_off = len(P_flex)

    # --- Sjekk forskyvningstype ---
    if t_base_on is not None and t_base_on > t_act:
        # Tidlig: fleks starter før baseline
        shift_type = "tidligere"
        tol = 1e-8
        try:
            t_off = np.where(np.isclose(T_flex[t_act:], T_flex.max(), atol=tol))[0][0] + t_act
        except IndexError:
            t_off = None


        delay = t_off - t_act
        energy_shifted = np.trapz(P_flex[t_act:t_off], dx=1/60)
        area_x = range(t_act, t_base_on)
        area_y1 = P_flex[t_act:t_base_on]
        area_y2 = P_base[t_act:t_base_on]
    else:
        # Sen: baseline varer lenger enn fleks
        shift_type = "senere"
        delay = t_base_off - t_flex_off
        energy_shifted = np.trapz(P_base[t_act:t_base_off], dx=1/60)
        area_x = range(t_act, t_base_off)
        area_y1 = P_base[t_act:t_base_off]
        area_y2 = P_flex[t_act:t_base_off]

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Temperatur
    h_T_base, = ax1.plot(T_base, "r--")
    h_T, = ax1.plot(T_flex, "r")
    ax1.set_ylim(ymin=min(T_base.min(), T_flex.min()) * 0.95,
                 ymax=max(T_base.max(), T_flex.max()) * 1.05)
    ax1.set_xlabel("Minutes")
    ax1.set_ylabel("Temperature (°C)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.axvline(t_act, color="black", linestyle=":", label="Activation")

    # Effekt
    ax2 = ax1.twinx()
    h_P_base, = ax2.plot(P_base, color="tab:blue", linestyle="dashed")
    h_P, = ax2.plot(P_flex, color="tab:blue")
    ax2.set_ylabel("Power (kW)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Skygge lagt areal
    h_area = ax2.fill_between(
        area_x,
        area_y1,
        area_y2,
        color="orange", alpha=0.5,
        label="Shifted energy"
    )

    # Tekst
    ax1.text(
        0.02, 0.95,
        f"Shift type: {shift_type}\n"
        f"Service Duration: {delay} min\n"
        f"Energy shifted: {energy_shifted:.3f} kWh",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    # Legend
    ax1.legend([h_T_base, h_T, h_P_base, h_P, h_area],
               ["Temp baseline", "Temp flex", "Power baseline", "Power flex", "Shifted energy"],
               loc="upper right")

    plt.title("Last- og temperatursvar med fleksibilitet (tidlig eller sen lastforskyvning)")
    plt.tight_layout()
    plt.show()

def plot_flexibility_activation(result):
    """
    Plot fleksibilitetsaktivering (kW) som differansen P_flex - P_base.
    Viser både aktivering (under null) og rebound-effekt (over null).
    """

    P_base = result["per_ewh"][0]["P_base"]
    P_flex = result["per_ewh"][0]["P"]
    t_act = result["params"]["t_act"]

    # Differanse mellom fleksibel og baseline last
    flex_signal = P_base - P_flex 

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(flex_signal, color="tab:green", linewidth=2,
             label="Flexibility Characteristic")
    plt.axhline(0, color="black", linestyle=":")  # baseline

    if t_act is not None:
        plt.axvline(t_act, color="red", linestyle="--", label="Activation start")

    plt.xlabel("Minutes")
    plt.ylabel("Δ Power (kW)")
    plt.title("Flexibility activation incl. rebound effect")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def quantify_capacity_flexibility(t_act_values, **kwargs):
    """
    Run simulations for different activation times and quantify
    the power capacity flexibility of EWHs.

    Parameters
    ----------
    t_act_values : list of ints
        Activation times in minutes.
    kwargs : dict
        Extra parameters for simulate_ewh.

    Returns
    -------
    dict
        Dictionary with results for each t_act.
    """

    results = {}

    plt.figure(figsize=(10, 6))

    for t_act in t_act_values:
        result = simulate_ewh(
            t_act=t_act,
            S_act=0,         # turn OFF
            N_EWH=100,       # 100 heaters
            plot=False,
            **kwargs
        )

        P_base = result["P_agg_base"]
        P_flex = result["P_agg"]

        # Flexibility potential = reduction
        deltaP = P_base - P_flex

        # Quantify capacity at activation time
        flex_capacity = deltaP[t_act]

        # Average over 30 min after activation
        window = 30
        flex_avg = np.mean(deltaP[t_act:t_act+window])

        results[t_act] = {
            "flex_capacity_at_tact": flex_capacity,
            "flex_avg_30min": flex_avg,
            "P_base": P_base,
            "P_flex": P_flex
        }

        # Plot curves
        #plt.plot(P_base, "k--", alpha=0.5, label="Baseline" if t_act==t_act_values[0] else "")
        #plt.plot(P_flex, label=f"Flex, t_act={t_act} min")
        # plot deltaP
        plt.plot(deltaP, label=f"Delta, t_act={t_act} min")

    plt.xlabel("Minutes")
    plt.ylabel("Power Capacity (kW)")
    plt.title("Load time series with baseline and flexible operation")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results

#t_act_values = [200, 400, 600, 800, 1000, 1200]  # 13:20, 16:00, 18:20
#results = quantify_capacity_flexibility(
#    t_act_values,
#    P_m=2.0,
#    T_a=24.0,
#    T_min=70.0,
#    T_max=75.0,
#    C=0.335,
#    R=600.0,
#    T_init_single=73.0,
#    S_init=0,
#    P_init=0.0,
#    time_steps=24*60,
#    seed=123
#)

# --- Eksempelkjøring ---
#result = simulate_ewh(
#    P_m=2.0,
#    T_a=24.0,
#    T_min=70.0,
#    T_max=75.0,
#    C=0.335,
#    R=600.0,
#    T_init_single=73.0,
#    S_init=0,
#    P_init=0.0,
#    t_act=780,      # Aktivering 13:00
#    S_act=0,        # Slå av
#    time_steps=24*60,
#    N_EWH=1,
#    seed=42,
#    plot=True
#)

result = simulate_ewh(
    P_m=2.0,
    T_a=24.0,
    T_min=70.0,
    T_max=75.0,
    C=0.335,
    R=600.0,
    T_init_single=73.0,
    S_init=0,
    P_init=0.0,
    t_act=780,      # Aktivering 13:00
    S_act=0,        # Slå av
    time_steps=24*60,
    N_EWH=1,
    seed=42,
    plot=True
)

#analyze_shift(result)

#plot_flexibility_activation(result)
#### # Exercise questions:

# 1) Plott uten å gjøre endringer.

#simulate_ewh(
#    P_m=2.0,
#    T_a=24.0,
#    T_min=70.0,
#    T_max=75.0,
#    C=0.335,
#    R=600.0,
#    T_init_single=73.0,
#    S_init=0,
#    P_init=0.0,
#    t_act=None,      # Change to None to disable flexibility
#    S_act=None,      # Change to None to disable flexibility
#    time_steps=24*60,
#    N_EWH=1,         # Change to >1 for multiple heaters
#    seed=42,
#    plot=True
#)

#  Når temperaturen når T_min skrus EWH på, og når den
# når T_max skrus den av. Dette gir en syklus som repeterer seg. 

# Flytte last til senere tidspunkt: Fra plottet i del 1 ser vi at EWH på i tidsrommet. 
# Aktiver signal på ca. 13:00 (780 min) for å skru av EWH som er på, 
# og dermed flytte last til senere tidspunkt.

#simulate_ewh(
#    P_m=2.0,
#    T_a=24.0,
#    T_min=70.0,
#    T_max=75.0,
#    C=0.335,
#    R=600.0,
#    T_init_single=73.0,
#    S_init=0,
#    P_init=0.0,
#    t_act=780,      # Activate flexibility at 13:00
#    S_act=0,        # Turn all EWH off at activation
#    time_steps=24*60,
#    N_EWH=1,        # Change to >1 for multiple heaters
#    seed=42,
#    plot=True
#)


#result = simulate_ewh(
#    P_m=2.0,
#    T_a=24.0,
#    T_min=70.0,
#    T_max=75.0,
#    C=0.335,
#    R=600.0,
#    T_init_single=73.0,
#    S_init=0,
#    P_init=0.0,
#    t_act=None,      # Activate flexibility at 13:00
#    S_act=None,        # Slå av
#    time_steps=24*60,
#    N_EWH=100,
#    seed=123,
#    plot=True
#)