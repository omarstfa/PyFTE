import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Importance factors (analytical, deterministic given inputs)
# ------------------------------------------------------------
def calculate_importance_factors(minimal_cut_sets, label_map, failure_probs):
    """
    Parameters
    ----------
    minimal_cut_sets : list[list[str]]
        Minimal cut sets expressed as lists of basic-event labels (e.g., "BE1", "BE2").
    label_map : dict[str, int]
        Maps each basic-event label to an index in `failure_probs`.
    failure_probs : np.ndarray
        Failure probabilities (P[fail during mission]) for each basic event.

    Returns
    -------
    pd.DataFrame with columns:
      - Birnbaum
      - Criticality
      - Fussell-Vesely
    """
    def prod(vals):
        out = 1.0
        for v in vals:
            out *= float(v)
        return out

    # Top-event unavailability (Q) via MCS sum of products approximation
    def top_unavailability(probs):
        return sum(prod(probs[label_map[e]] for e in cut) for cut in minimal_cut_sets)

    Q = top_unavailability(failure_probs)

    birnbaum = {}
    criticality = {}
    fv = {}

    be_labels = list(label_map.keys())
    for be in be_labels:
        i = label_map[be]
        p_i = failure_probs[i]

        # Birnbaum importance: ∂Q/∂p_i under the MCS SOP model
        dQ_dp = 0.0
        for cut in minimal_cut_sets:
            if be in cut:
                others = [e for e in cut if e != be]
                dQ_dp += prod(failure_probs[label_map[e]] for e in others)
        birnbaum[be] = dQ_dp

        # Criticality importance: P(BE_i and TOP) / Q
        joint = 0.0
        for cut in minimal_cut_sets:
            if be in cut:
                others = [e for e in cut if e != be]
                joint += p_i * prod(failure_probs[label_map[e]] for e in others)
        criticality[be] = (joint / Q) if Q > 0 else 0.0

        # Fussell–Vesely: fraction of Q attributable to cut sets containing the BE
        q_with = 0.0
        for cut in minimal_cut_sets:
            if be in cut:
                q_with += prod(failure_probs[label_map[e]] for e in cut)
        fv[be] = (q_with / Q) if Q > 0 else 0.0

    df = pd.DataFrame({
        "Birnbaum": pd.Series(birnbaum),
        "Criticality": pd.Series(criticality),
        "Fussell-Vesely": pd.Series(fv),
    }).sort_index(key=lambda s: s.str.extract(r'(\d+)').astype(int)[0])

    return df


# ------------------------------------------------------------
# Monte Carlo reliability with CIs over simulations
# ------------------------------------------------------------
import numpy as np
import pandas as pd

# (Keep any other functions you already have here, e.g., calculate_importance_factors)

def simulate_reliability(minimal_cut_sets, failure_rates, repair_times, T=1000, dt=1, N_SIM=5000):
    """
    Returns (same as before, with two extras appended at the end):
      1) sys_unavail
      2) sys_ci                  <-- NOTE: this remains the CI of *availability* for backward compatibility
      3) comp_df
      4) availability_time_series
      5) time_grid
      6) all_component_states
      7) availability_sys
      8) MTTF_sys
      9) MTTR_sys
     10) comp_unavail_summary    <-- NEW: per-component 95% CI for unavailability
     11) sys_unavail_ci          <-- NEW: 95% CI for system unavailability
    """
    time_grid = np.arange(0, T + dt, dt)
    basic_events = {
        f'BE{i}': {'fail_rate': failure_rates[f'BE{i}'], 'repair_mean': repair_times[f'BE{i}']}
        for i in range(1, 19)
    }

    def simulate_component(f_rate, repair_mean):
        t = 0
        events = []
        while t < T:
            t += np.random.exponential(1 / f_rate)
            if t >= T:
                break
            t_down = t
            t += np.random.exponential(repair_mean)
            events.append((t_down, min(t, T)))
        return events

    sys_avail = np.zeros(N_SIM)
    component_stats = {be: {'up_times': [], 'down_times': []} for be in basic_events}
    top_event_states = np.zeros((N_SIM, len(time_grid)), dtype=bool)
    all_component_states = {be: np.zeros((N_SIM, len(time_grid)), dtype=bool) for be in basic_events}

    top_event_up_times = []
    top_event_down_times = []

    for sim in range(N_SIM):
        comp_states = {be: np.zeros_like(time_grid, dtype=bool) for be in basic_events}
        for be, params in basic_events.items():
            f_rate = params['fail_rate']
            events = simulate_component(f_rate, params['repair_mean'])
            for down, up in events:
                idx = (time_grid >= down) & (time_grid < up)
                comp_states[be][idx] = True
                all_component_states[be][sim][idx] = True
                component_stats[be]['down_times'].append(up - down)

        # Top event state (system failure state)
        top_state = np.zeros_like(time_grid, dtype=bool)
        for cut in minimal_cut_sets:
            mask = np.logical_and.reduce([comp_states[e] for e in cut])
            top_state |= mask

        top_event_states[sim] = top_state
        sys_avail[sim] = 1 - np.mean(top_state)

        # Track up and down periods for top event
        prev_state = False
        start_time = 0
        for i, state in enumerate(top_state):
            t = time_grid[i]
            if state and not prev_state:  # up → down
                top_event_up_times.append(t - start_time)
                start_time = t
                prev_state = True
            elif not state and prev_state:  # down → up
                top_event_down_times.append(t - start_time)
                start_time = t
                prev_state = False

        # Handle tail segment
        if not prev_state:
            top_event_up_times.append(T - start_time)
        else:
            top_event_down_times.append(T - start_time)

        for be in basic_events:
            # This is actually the fraction of time UP in this simulation
            component_stats[be]['up_times'].append(np.mean(~comp_states[be]))

    # -------------------------------
    # Component-level point estimates
    # -------------------------------
    comp_df = pd.DataFrame({
        be: {
            'Failure Rate': failure_rates[be],
            'Unavailability': 1 - np.mean(stats['up_times']),
            'MTBF': (np.mean(stats['up_times']) * T / len(stats['down_times'])) if stats['down_times'] else np.nan,
            'MTTR': (np.mean(stats['down_times'])) if stats['down_times'] else np.nan
        }
        for be, stats in component_stats.items()
    }).T

    # ---------------------------------------------
    # NEW: per-component 95% CI for unavailability
    # ---------------------------------------------
    comp_unavail_summary = pd.DataFrame({
        be: {
            'Unavailability_mean': float(1 - np.mean(stats['up_times'])),
            'Unavailability_low':  float(np.percentile(1 - np.array(stats['up_times']), 2.5)),
            'Unavailability_high': float(np.percentile(1 - np.array(stats['up_times']), 97.5)),
        }
        for be, stats in component_stats.items()
    }).T

    # --------------------
    # System-level metrics
    # --------------------
    # System unavailability per simulation (direct, simple, unambiguous)
    sys_unavail_per_sim = 1 - sys_avail

    sys_unavail = float(np.mean(sys_unavail_per_sim))
    sys_unavail_ci = np.percentile(sys_unavail_per_sim, [2.5, 97.5])  # NEW: CI on UNAVAILABILITY

    # Keep your original availability CI for backward compatibility
    sys_ci = np.percentile(sys_avail, [2.5, 97.5])

    MTTF_sys = np.mean(top_event_up_times) if top_event_up_times else None
    MTTR_sys = np.mean(top_event_down_times) if top_event_down_times else None
    availability_sys = (
        MTTF_sys / (MTTF_sys + MTTR_sys) if MTTF_sys is not None and MTTR_sys is not None else None
    )

    # == Availability A(t) ==
    
    # Time series for plotting (mean availability over sims at each time)
    availability_time_series = 1 - np.mean(top_event_states, axis=0)
    
    # 95% Wilson binomial CI for P(UP) at each time t
    N = float(N_SIM)
    z = 1.959963984540054  # 95% z-score
    p_hat = 1 - top_event_states.mean(axis=0)  # same as availability_time_series
    
    denom = 1.0 + (z**2) / N
    center = (p_hat + (z**2) / (2.0 * N)) / denom
    half_width = z * np.sqrt((p_hat * (1.0 - p_hat) / N) + (z**2) / (4.0 * N**2)) / denom
    
    availability_time_low = np.clip(center - half_width, 0.0, 1.0)
    availability_time_high = np.clip(center + half_width, 0.0, 1.0)
    
    
    # == Reliability R(t) ==
    
    # R(t) = P(no failure yet by time t) = 1 - P(ever failed by t)
    # Convert per-sim failure-state to "ever failed" by cumulative OR along time
    ever_failed = np.logical_or.accumulate(top_event_states, axis=1)    # shape (N_SIM, len(time_grid))
    reliability_time_series = 1.0 - ever_failed.mean(axis=0)            # shape (len(time_grid),)
    
    # 95% Wilson binomial CI for R(t) at each time (stable even when near 0 or 1)
    N = float(N_SIM)
    z = 1.959963984540054  # 95% z-score
    p_hat = reliability_time_series
    denom = 1.0 + (z**2) / N
    center = (p_hat + (z**2) / (2.0 * N)) / denom
    half_width = z * np.sqrt((p_hat * (1.0 - p_hat) / N) + (z**2) / (4.0 * N**2)) / denom
    reliability_time_low  = np.clip(center - half_width, 0.0, 1.0)
    reliability_time_high = np.clip(center + half_width, 0.0, 1.0)



# =============================================================================
# 
# =============================================================================

    dt = float(time_grid[1] - time_grid[0]) if len(time_grid) > 1 else 0.0
    N_SIM, Nt = top_event_states.shape
    
    # All missions complete at T in your current code; adjust later if you allow early ends
    mission_end_times = np.full(N_SIM, time_grid[-1])
    elig = (mission_end_times[:, None] >= time_grid[None, :])  # (N_SIM, Nt)
    
    # ---- Availability(t): Σ uptime / Σ mission time (over completed missions) ----
    # Left-interval cumulative uptime (per run, up to t_k): sum UP over intervals [t_i, t_{i+1}), i = 0..k-1
    cum_up_intervals = np.cumsum((~top_event_states)[:, :-1], axis=1)   # (N_SIM, Nt-1)
    cum_up_left = np.pad(cum_up_intervals, ((0, 0), (1, 0)), mode="constant") * dt  # (N_SIM, Nt), starts at 0
    numerator_A = (cum_up_left * elig).sum(axis=0)                       # (Nt,)
    
    # Total mission time over completed missions at t: each eligible run contributes t
    denominator_A = elig.sum(axis=0) * time_grid                         # (Nt,)
    denominator_A = np.where(denominator_A == 0.0, 1.0, denominator_A)   # avoid 0/0 at t=0
    availability_time_series_mission = numerator_A / denominator_A
    if Nt > 0:
        availability_time_series_mission[0] = 1.0                        # define A(0) = 1
    
    # ---- Reliability(t): #successful missions / #considered runs ----
    # If you want "no failure strictly before t" (aligned with left-interval convention), shift once:
    ever_failed_before = np.concatenate(
        [np.zeros((N_SIM, 1), dtype=bool),
         np.logical_or.accumulate(top_event_states[:, :-1], axis=1)],
        axis=1
    )                                                                     # (N_SIM, Nt)
    success_matrix = ~ever_failed_before
    runs_counts_t = np.where(elig.sum(axis=0) == 0, 1, elig.sum(axis=0))  # (Nt,)
    reliability_time_series_mission = (success_matrix & elig).sum(axis=0) / runs_counts_t

    
        
    return (
        sys_unavail,
        sys_ci,                   # availability CI
        comp_df,
        availability_time_series,
        time_grid,
        all_component_states,
        availability_sys,
        MTTF_sys,
        MTTR_sys,
        comp_unavail_summary,     
        sys_unavail_ci,            
        availability_time_low,
        availability_time_high,
        top_event_states,
        reliability_time_series,
        reliability_time_low,
        reliability_time_high,
        availability_time_series_mission,
        reliability_time_series_mission,
    )
