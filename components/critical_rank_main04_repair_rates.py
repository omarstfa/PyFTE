
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
    failure_probs : np.ndarray or dict[str,float]
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

    # Allow dict-like failure_probs
    if isinstance(failure_probs, dict):
        # build a vector aligned with label_map order
        vec = np.zeros(len(label_map), dtype=float)
        for be, idx in label_map.items():
            vec[idx] = failure_probs[be]
        failure_probs = vec

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
# Event-driven Monte Carlo for Availability & Reliability
# ------------------------------------------------------------

from typing import Dict, List, Optional, Tuple

def _simulate_be_timeline(T: float, dt: float, lam: float, mu: float, rng) -> np.ndarray:
    """
    Returns a boolean array of shape (len(time_grid),) that is True when the BE is DOWN.
    Time-to-failure ~ Exp(lam); repair duration ~ Exp(mu).
    If mu == 0, the BE is non-repairable (stays down after first failure).
    """
    time_grid = np.arange(0, T + dt, dt)
    down = np.zeros_like(time_grid, dtype=bool)
    t = 0.0
    while t < T:
        # Up-time until next failure
        if lam <= 0.0:
            break
        t += rng.exponential(1.0 / lam)
        if t >= T:
            break
        down_start = t
        # Exponential repair time (mu per hour)
        if mu > 0.0:
            rep_dur = rng.exponential(1.0 / mu)
        else:
            # Non-repairable
            rep_dur = T - down_start
        t = down_start + rep_dur
        down_end = min(t, T)
        idx = (time_grid >= down_start) & (time_grid < down_end)
        down[idx] = True
    return down


def simulate_reliability(
    minimal_cut_sets: List[List[str]],
    failure_rates: Dict[str, float],
    repair_rates: Dict[str, float],
    T: float = 1000.0,
    dt: float = 1.0,
    N_SIM: int = 3000,
    be_to_component: Optional[Dict[str, str]] = None,
    rng_seed: int = 42,
):
    """
    Core, compact simulator that aligns with the algorithm spec.

    minimal_cut_sets : list of MCS, each a list of BE labels (e.g., ["BE3","BE4"])
    failure_rates    : dict BE->lambda (per hour)
    repair_rates     : dict BE->mu (per hour) for exponential repair durations
    T, dt, N_SIM     : horizon, resolution, replications
    be_to_component  : optional dict mapping each BE to a component name
                       (if omitted, each BE is treated as its own component)

    Returns dict with:
      - time_grid
      - availability_time_series (mean A(t))
      - availability_time_low/high (95% Wilson CI)
      - reliability_time_series (mean R(t))
      - reliability_time_low/high (95% Wilson CI)
      - system_unavailability_mean, system_unavailability_ci
      - component_unavailability (DataFrame, mean unavailability per component)
      - top_event_states (N_SIM x len(time_grid) bool)
    """
    rng = np.random.default_rng(rng_seed)
    time_grid = np.arange(0, T + dt, dt)
    Nt = len(time_grid)

    be_list = sorted(failure_rates.keys(), key=lambda x: int(x[2:]))
    # Default mapping: each BE is its own component
    if be_to_component is None:
        be_to_component = {be: be for be in be_list}

    # --- Pre-sim allocations ---
    top_event_states = np.zeros((N_SIM, Nt), dtype=bool)  # True => system DOWN
    be_states = {be: np.zeros((N_SIM, Nt), dtype=bool) for be in be_list}

    # --- Simulate BE timelines ---
    for s in range(N_SIM):
        for be in be_list:
            lam = float(failure_rates[be])
            mu  = float(repair_rates[be])
            be_states[be][s, :] = _simulate_be_timeline(T, dt, lam, mu, rng)

        # Top event: OR over cut-ANDs of BE states
        te = np.zeros(Nt, dtype=bool)
        for cut in minimal_cut_sets:
            cut_mask = np.logical_and.reduce([be_states[be][s, :] for be in cut])
            te |= cut_mask
        top_event_states[s, :] = te

    # --- Availability A(t) ---
    A_t = 1.0 - top_event_states.mean(axis=0)

    # Wilson CI for binomial mean
    N = float(N_SIM)
    z = 1.959963984540054  # 95%
    denom = 1.0 + (z**2) / N

    def wilson_ci(p_hat_vec):
        center = (p_hat_vec + (z**2) / (2.0 * N)) / denom
        half_width = z * np.sqrt((p_hat_vec * (1.0 - p_hat_vec) / N) + (z**2) / (4.0 * N**2)) / denom
        lo = np.clip(center - half_width, 0.0, 1.0)
        hi = np.clip(center + half_width, 0.0, 1.0)
        return lo, hi

    A_lo, A_hi = wilson_ci(A_t)

    # --- Reliability R(t) = P(no failure by t) ---
    ever_failed = np.logical_or.accumulate(top_event_states, axis=1)  # (N_SIM, Nt)
    R_t = 1.0 - ever_failed.mean(axis=0)
    R_lo, R_hi = wilson_ci(R_t)

    # --- System unavailability summary over [0,T] ---
    sys_unavail_per_sim = top_event_states.mean(axis=1)  # fraction of time DOWN in each run
    sys_unavail_mean = float(sys_unavail_per_sim.mean())
    sys_unavail_ci = np.percentile(sys_unavail_per_sim, [2.5, 97.5]).astype(float)

    # --- Component rollup: OR of BEs mapped to the component ---
    # build component->list[BE]
    comp_to_bes = {}
    for be, comp in be_to_component.items():
        comp_to_bes.setdefault(comp, []).append(be)

    comp_unavail = {}
    for comp, bes in comp_to_bes.items():
        # OR across BEs for each sim/time, then mean across sims/time
        comp_down = np.zeros((N_SIM, Nt), dtype=bool)
        for be in bes:
            comp_down |= be_states[be]
        comp_unavail[comp] = float(comp_down.mean())  # fraction of time down

    component_unavailability = pd.DataFrame.from_dict(
        comp_unavail, orient="index", columns=["Unavailability"]
    ).sort_index(key=lambda s: s.str.lower())

    return {
        "time_grid": time_grid,
        "availability_time_series": A_t,
        "availability_time_low": A_lo,
        "availability_time_high": A_hi,
        "reliability_time_series": R_t,
        "reliability_time_low": R_lo,
        "reliability_time_high": R_hi,
        "system_unavailability_mean": sys_unavail_mean,
        "system_unavailability_ci": sys_unavail_ci,
        "component_unavailability": component_unavailability,
        "top_event_states": top_event_states,
    }


# ------------------------------------------------------------
# Reliability Function helpers
# ------------------------------------------------------------
from typing import Callable

def _first_failure_times(top_event_states: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute time-to-first-failure per simulation (right-censored at the horizon).
    top_event_states: (N_SIM, Nt) bool where True means system DOWN at t
    Returns array of shape (N_SIM,) with times in hours; if never failed, value is np.inf.
    """
    N_SIM, Nt = top_event_states.shape
    fft = np.full(N_SIM, np.inf, dtype=float)
    first_idx = np.argmax(top_event_states, axis=1)  # 0 if all False, else first True index
    # But if all False, argmax returns 0; we need to detect that
    never = ~top_event_states.any(axis=1)
    # time grid assumed uniform with step dt, starting at 0
    fft[~never] = first_idx[~never] * dt
    return fft

def fit_exponential_reliability(top_event_states: np.ndarray, dt: float, T: float) -> float:
    """
    MLE for exponential failure rate with right-censoring ("time on test").
    Returns lambda_hat (per hour). If there are zero failures, returns 0.0.
    """
    fft = _first_failure_times(top_event_states, dt)
    failures = np.isfinite(fft)
    n_fail = failures.sum()
    total_time_on_test = np.where(failures, fft, T).sum()
    if total_time_on_test <= 0:
        return 0.0
    lam_hat = n_fail / total_time_on_test
    return float(lam_hat)

def make_reliability_function(time_grid: np.ndarray, R_t: np.ndarray) -> Callable[[float], float]:
    """
    Create a callable R(t) via linear interpolation on (time_grid, R_t).
    Values outside the grid are clamped to the nearest endpoint.
    """
    tg = np.asarray(time_grid, dtype=float)
    rv = np.asarray(R_t, dtype=float)
    def R_of_t(t: float) -> float:
        t = float(t)
        if t <= tg[0]:
            return float(rv[0])
        if t >= tg[-1]:
            return float(rv[-1])
        # numpy.interp is linear; for stepwise, one could use searchsorted.
        return float(np.interp(t, tg, rv))
    return R_of_t
