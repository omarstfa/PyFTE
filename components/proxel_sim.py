# PyFTE/components/proxel_sim.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import warnings

# ----------------------------------------------------------------------
# Distribution helpers (using scipy if available, else fallback math)
# ----------------------------------------------------------------------
try:
    from scipy.stats import expon, weibull_min, norm, lognorm, uniform
    _has_scipy = True
except ImportError:
    _has_scipy = False
    warnings.warn("scipy not installed; some distributions may not work. "
                  "Install with: pip install scipy")

def _hazard_exp(t, rate):
    """Exponential hazard = rate (constant)."""
    return rate

def _hazard_weibull(t, shape, scale):
    """Weibull hazard = (shape/scale) * (t/scale)^(shape-1)."""
    if t < 0:
        return 0.0
    return (shape / scale) * (t / scale) ** (shape - 1)

def _hazard_norm(t, loc, scale):
    """Normal hazard = pdf(t) / (1 - cdf(t))."""
    from math import erf, sqrt, exp, pi
    z = (t - loc) / scale
    pdf = exp(-0.5 * z * z) / (scale * sqrt(2 * pi))
    cdf = 0.5 * (1 + erf(z / sqrt(2)))
    surv = 1.0 - cdf
    if surv <= 0:
        return float('inf')
    return pdf / surv

def _hazard_lognorm(t, s, loc, scale):
    """Lognormal hazard."""
    from math import erf, sqrt, exp, pi, log
    if t <= loc:
        return 0.0
    if _has_scipy:
        from scipy.stats import lognorm
        pdf = lognorm.pdf(t, s, loc=loc, scale=scale)
        cdf = lognorm.cdf(t, s, loc=loc, scale=scale)
    else:
        x = (t - loc) / scale
        if x <= 0:
            return 0.0
        lnx = log(x)
        pdf = exp(-0.5 * (lnx / s) ** 2) / (s * x * sqrt(2 * pi))
        cdf = 0.5 + 0.5 * erf(lnx / (s * sqrt(2)))
    surv = 1.0 - cdf
    if surv <= 0:
        return float('inf')
    return pdf / surv

def _hazard_unif(t, a, b):
    """Uniform hazard."""
    if t < a:
        return 0.0
    if t > b:
        return float('inf')
    pdf = 1.0 / (b - a)
    cdf = (t - a) / (b - a)
    surv = 1.0 - cdf
    return pdf / surv

def hazard(t: float, dist: str, params: Tuple) -> float:
    """
    Compute hazard rate h(t) for given distribution.
    dist : one of 'exp', 'weibull', 'norm', 'lnorm', 'unif'
    params : tuple of parameters appropriate for the distribution.
    """
    if dist == 'exp':
        rate, = params
        return _hazard_exp(t, rate)
    elif dist == 'weibull':
        shape, scale = params
        return _hazard_weibull(t, shape, scale)
    elif dist == 'norm':
        loc, scale = params
        return _hazard_norm(t, loc, scale)
    elif dist == 'lnorm':
        # Typically lognormal parameters: shape (s), scale (sigma), loc (0)
        s, scale, loc = params if len(params) == 3 else (params[0], params[1], 0)
        return _hazard_lognorm(t, s, loc, scale)
    elif dist == 'unif':
        a, b = params
        return _hazard_unif(t, a, b)
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

# ----------------------------------------------------------------------
# Basic Event definition
# ----------------------------------------------------------------------
class BasicEvent:
    """
    Represents a single basic event (component) with multiple states.
    Attributes:
        states : list of state names (e.g., ['OK', 'F'] or ['OK','IS','F'])
        G      : transition matrix (list of lists). 
                 0 = impossible, 1 = possible, NA = diagonal (will be computed)
        dist   : list of distribution names for each possible transition (same order as non-zero entries)
        param  : list of parameter tuples for each distribution
    """
    def __init__(self, states: List[str], G: List[List], dist: List[str], param: List):
        self.states = states
        self.n = len(states)
        self.G = np.array(G, dtype=object)  # keep NA as object
        self.dist = dist
        self.param = param
        # Build index mapping state name to position
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.idx_to_state = {i: s for i, s in enumerate(states)}
        # Pre‑compute whether all transitions are exponential (for fast path)
        self.all_exp = all(d == 'exp' for d in dist)

    def transition_prob_matrix(self, t: float, delta: float) -> np.ndarray:
        """
        Compute the n x n transition probability matrix for this basic event,
        given current age t and time step delta.
        """
        n = self.n
        P = np.zeros((n, n))
        # Count how many transitions are defined (non-zero off-diagonals)
        # We'll fill off-diagonals where G[i,j] == 1
        dist_idx = 0
        for i in range(n):
            row_sum = 0.0
            for j in range(n):
                if i == j:
                    continue
                if self.G[i, j] == 1:
                    # Transition i -> j is possible
                    h = hazard(t, self.dist[dist_idx], self.param[dist_idx])
                    p_trans = delta * h
                    if p_trans > 1.0:
                        p_trans = 1.0   # clamp
                    P[i, j] = p_trans
                    row_sum += p_trans
                    dist_idx += 1
                # else G[i,j]==0 => P[i,j]=0
            # Diagonal: probability of staying in state i
            P[i, i] = 1.0 - row_sum
        return P

# ----------------------------------------------------------------------
# Proxel simulation for a single basic event
# ----------------------------------------------------------------------
def simulate_basic_event(be: BasicEvent, target_state: str,
                         totaltime: float, delta: float,
                         tol: float = 1e-7) -> np.ndarray:
    """
    Run proxel simulation for a single basic event.
    Returns array of length steps = int(totaltime/delta)+1,
    giving probability of being in `target_state` at each time step.
    """
    steps = int(totaltime / delta) + 1
    unav = np.zeros(steps)

    # Fast path: all transitions are exponential (memoryless)
    if be.all_exp:
        # Constant transition probability matrix (age irrelevant)
        P = be.transition_prob_matrix(0.0, delta)
        # Initial state: first state (assumed OK)
        state_probs = np.zeros(be.n)
        state_probs[0] = 1.0
        target_idx = be.state_to_idx[target_state]
        for k in range(steps):
            unav[k] = state_probs[target_idx]
            if k == steps - 1:
                break
            state_probs = state_probs @ P
        return unav

    # General path: track age for non‑exponential distributions
    # Initial proxel: first state (OK), age 0
    proxels = {(be.states[0], 0.0): 1.0}
    for k in range(steps):
        # Record probability of target state at current time
        prob_target = 0.0
        for (state, age), p in proxels.items():
            if state == target_state:
                prob_target += p
        unav[k] = prob_target

        if k == steps - 1:
            break

        # Compute next proxels
        new_proxels = {}
        for (state, age), prob in proxels.items():
            P = be.transition_prob_matrix(age, delta)
            i = be.state_to_idx[state]
            for j, next_state in enumerate(be.states):
                p_trans = P[i, j]
                if p_trans <= 0:
                    continue
                new_age = 0.0 if j != i else age + delta
                key = (next_state, new_age)
                new_proxels[key] = new_proxels.get(key, 0.0) + prob * p_trans

        # Prune negligible probabilities
        if tol > 0:
            new_proxels = {k: v for k, v in new_proxels.items() if v > tol}

        proxels = new_proxels

    return unav

# ----------------------------------------------------------------------
# System simulation using minimal cut sets
# ----------------------------------------------------------------------
def proxel_system(belist: Dict[str, BasicEvent],
                  mcs: List[List[str]],
                  totaltime: float,
                  delta: float,
                  tol: float = 1e-7) -> Dict:
    """
    belist : dict mapping BE name to BasicEvent object.
    mcs    : list of cut sets, each a list of BE names.
    totaltime, delta, tol : as before.

    Returns dict with:
        'time_grid' : array of time points
        'be_unavailability' : dict BE_name -> array of unavailability
        'system_unavailability' : array of system down probability at each time step
    """
    steps = int(totaltime / delta) + 1
    time_grid = np.linspace(0, totaltime, steps)

    # Simulate each basic event independently
    be_unav = {}
    for name, be in belist.items():
        # target state is the failed state (assumed last state in list, e.g., 'F')
        target = be.states[-1]
        unav = simulate_basic_event(be, target, totaltime, delta, tol)
        be_unav[name] = unav

    # Compute system unavailability using MCS approximation
    sys_unav = np.zeros(steps)
    for t_idx in range(steps):
        prob_sys_down = 0.0
        for cut in mcs:
            prod = 1.0
            for be_name in cut:
                prod *= be_unav[be_name][t_idx]
            prob_sys_down += prod
        sys_unav[t_idx] = min(prob_sys_down, 1.0)   # clamp

    return {
        'time_grid': time_grid,
        'be_unavailability': be_unav,
        'system_unavailability': sys_unav
    }