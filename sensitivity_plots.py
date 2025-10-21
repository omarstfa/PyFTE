# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 13:06:09 2025

@author: cj6253
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Dict
from collections.abc import Iterable

# Make sure we can import your simulator
sys.path.append(".")            # adjust if needed
sys.path.append("/mnt/data")    # adjust if you run in that environment
from components.critical_rank import simulate_reliability  # :contentReference[oaicite:0]{index=0}

# -----------------------------
# Fault tree → minimal cut sets
# -----------------------------
# From main05.py structure (no components import needed): :contentReference[oaicite:1]{index=1}
# Top = OR( IE1 , IE2 )
# IE2 = OR(BE17, BE18) → {BE17}, {BE18}
# IE1 includes:
#   - BE1, BE2 (OR) → {BE1}, {BE2}
#   - IE3 = OR( IE4 , IE5 )
#       IE4 = OR(BE5..BE16) → {BE5}..{BE16}
#       IE5 = AND(BE3, BE4) → {BE3, BE4}
minimal_cut_sets_BE = (
    [["BE1"], ["BE2"]] +
    [[f"BE{i}"] for i in range(5, 17)] +
    [["BE3","BE4"]] +
    [["BE17"], ["BE18"]]
)

# -----------------------------
# Base inputs (copied from main05)  :contentReference[oaicite:2]{index=2}
# -----------------------------
T_MISSION = 100_000.0   # hours
DT        = 50.0        # time step (coarser = faster; 1.0 is fine if runtime allows)
N_SIM     = 2000        # Monte Carlo reps (increase for tighter CI)

failure_rates_base = {
    'BE1':1e-7, 'BE2':2e-7,
    'BE3':8e-7, 'BE4':8e-7,
    'BE5':8.46e-6, 'BE6':4.9e-6,
    'BE7':3e-7, 'BE8':1.3e-6,
    'BE9':8.8e-6, 'BE10':1.115e-5,
    'BE11':1.487e-5,'BE12':1.01e-6,
    'BE13':2.1e-7,'BE14':5.2e-7,
    'BE15':7.29e-6,'BE16':5.7e-6,
    'BE17':1e-7,'BE18':2e-7
}

repair_times = {
    'BE1': 4,   'BE2': 4,
    'BE3': 10,  'BE4': 10,
    'BE5': 24,  'BE6': 24,
    'BE7': 24,  'BE8': 8,
    'BE9': 8,   'BE10': 48,
    'BE11': 48, 'BE12': 24,
    'BE13': 24, 'BE14': 12,
    'BE15': 48, 'BE16': 48,
    'BE17': 4,  'BE18': 4,
}

# -----------------------------
# Sensitivity driver
# -----------------------------
def run_sensitivity(be_name, multipliers=(1, 2, 3, 4), *,
                    failure_rates_base=failure_rates_base,
                    T=T_MISSION, dt=DT, N=N_SIM, seed0=12345):
    """
    For the given BE, multiply its failure rate by each factor in `multipliers`,
    run the simulation, and return a DataFrame with mean unavailability and 95% CI.
    """
    rows = []
    for i, m in enumerate(multipliers):
        fr = dict(failure_rates_base)
        fr[be_name] = failure_rates_base[be_name] * m
        res = simulate_reliability(
            minimal_cut_sets=minimal_cut_sets_BE,
            failure_rates=fr,
            repair_times=repair_times,
            T=T,
            dt=dt,
            N_SIM=N,
            be_to_component=None,
            rng_seed=seed0 + i,
        )
        mean_u = res["system_unavailability_mean"]
        lo, hi = map(float, res["system_unavailability_ci"])
        rows.append({
            "BE": be_name,
            "Multiplier": m,
            "FailureRate (/h)": fr[be_name],
            "System Unavailability (mean)": mean_u,
            "CI low (2.5%)": lo,
            "CI high (97.5%)": hi,
        })
    df = pd.DataFrame(rows)
    return df

def run_sensitivity_for_events(
    events: Iterable[str] = ("BE4","BE11"),
    multipliers=None,
    *,
    max_multiplier: int = 4,
    failure_rates_base=failure_rates_base,
    T=T_MISSION, dt=DT, N=N_SIM, seed0=12345
) -> Dict[str, pd.DataFrame]:
    """
    Run the sensitivity analysis for each event in `events`.

    Controls:
      - multipliers: explicit sequence (e.g., (1,2,3,4)).
      - max_multiplier: if `multipliers` is None, use range 1..max_multiplier.

    Returns dict: { 'BE4': df4, 'BE11': df11, ... } with
    columns including 'Multiplier' and CI bounds.
    """
    if multipliers is None:
        if not isinstance(max_multiplier, int) or max_multiplier < 1:
            raise ValueError("max_multiplier must be an integer >= 1.")
        multipliers = tuple(range(1, max_multiplier + 1))
    else:
        multipliers = tuple(multipliers)

    out = {}
    for k, be in enumerate(events):
        out[be] = run_sensitivity(
            be, multipliers,
            failure_rates_base=failure_rates_base,
            T=T, dt=dt, N=N, seed0=seed0 + 1000*k
        )
        # optional: write per-BE CSV
        out[be].to_csv(f"{be}_sensitivity.csv", index=False)
    return out



def plot_sensitivity_multi(
    df_map,
    *,
    multipliers=None,
    max_multiplier: int = 4,
    offset_width: float = 0.15,
    capsize: float = 5,
    markersize: float = 7,
    capthick: float =1.5, 
    elinewidth: float =1.5,
    markeredgewidth: float = 1.5,
    out_png: str = "sensitivity_multi.png",
    ylim=None,
    dpi: int = 600,
    cmap: str = "Set1_r",
    cmap_range=(0.7, 1),   # <— only use 25%..100% of Reds; raise 0.25 to avoid white more
):
    """
    Combined plot for multiple BEs with horizontal 'dodge' and a green→red gradient
    across the basic events in their given order (iteration order of df_map).
    """

    # Establish the centers on the x-axis
    if multipliers is None:
        if not isinstance(max_multiplier, int) or max_multiplier < 1:
            raise ValueError("max_multiplier must be an integer >= 1.")
        centers = np.arange(1, max_multiplier + 1, dtype=float)
    else:
        centers = np.array(tuple(multipliers), dtype=float)

    n_series = len(df_map)
    offsets = np.zeros(n_series) if n_series == 1 else np.linspace(-offset_width, offset_width, n_series)

    # Build gradient colors sampled from a truncated portion of the colormap
    cm = plt.get_cmap(cmap)
    n_series = len(df_map)
    lo, hi = cmap_range
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("cmap_range must be a (lo, hi) within [0,1] and lo < hi.")

    if n_series <= 1:
        colors = [cm(hi)]  # pick a strong red
    else:
        # sample linearly within [lo, hi]
        colors = [cm(lo + (hi - lo) * i / (n_series - 1)) for i in range(n_series)]

    fig = plt.figure(figsize=(9, 5), dpi=dpi)
    ax = fig.add_subplot(111)

    # Plot each BE with a horizontal dodge and its gradient color
    for j, (be_name, df) in enumerate(df_map.items()):
        x  = df["Multiplier"].to_numpy(float)
        y  = df["System Unavailability (mean)"].to_numpy(float)
        lo = df["CI low (2.5%)"].to_numpy(float)
        hi = df["CI high (97.5%)"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])

        x_off = x + offsets[j]
        color = colors[j]

        ax.errorbar(
            x_off, y, yerr=yerr,
            fmt='_', linestyle='none',
            capsize=capsize, markersize=markersize,
            color=color, ecolor=color,capthick=capthick, 
            elinewidth=elinewidth, markeredgewidth=markeredgewidth,
            label=be_name
        )

    # Ticks at centers (e.g., 1..M) labeled x1..xM
    ax.set_xticks(centers)
    ax.set_xticklabels([f"x{int(v)}" for v in centers], fontsize=12)
    
    
    # Room for offsets at both ends
    pad = 0.4 + offset_width
    ax.set_xlim(centers.min() - pad, centers.max() + pad)

    # Flexible ylim control: number => top (bottom=0), tuple => (ymin, ymax)
    if ylim is None:
        ax.set_ylim(bottom=0)
    elif isinstance(ylim, (int, float)):
        ax.set_ylim(bottom=0, top=float(ylim))
    elif isinstance(ylim, Iterable) and len(tuple(ylim)) == 2:
        lo_y, hi_y = tuple(ylim)
        ax.set_ylim(lo_y, hi_y)
    else:
        raise ValueError("ylim must be None, a number (ymax), or a 2-tuple (ymin, ymax).")

    ax.tick_params(axis='y', labelsize=12)
    ax.minorticks_off()
    ax.set_xlabel("Increase Factor of Event Occurrence", fontsize=14)
    ax.set_ylabel("System Unavailability", fontsize=14)
    ax.legend(title="Basic Event", frameon=False, fontsize=12, title_fontsize=12)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.show()

# %% Run

if __name__ == "__main__":

    # Choose which events to include
    events = ['BE4', 'BE15', 'BE11']
    
    # Option A: control by max multiplier (e.g., 1..6)
    dfs = run_sensitivity_for_events(events, max_multiplier=6)
    plot_sensitivity_multi(
        dfs,
        max_multiplier=5,       # x = 1..6
        offset_width=0.15,      # spread series more at each multiplier
        out_png=None,
        ylim=0.009
    )
    
    # # Option B: explicit multipliers (e.g., 1,2,3,4 only)
    # dfs = run_sensitivity_for_events(events, multipliers=(1,2,3,4))
    # plot_sensitivity_multi(
    #     dfs,
    #     multipliers=(1,2,3,4),
    #     offset_width=0.15,
    #     out_png="sensitivity_multi_m4_offset015.png",
    # )
