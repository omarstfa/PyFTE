import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Set
from datetime import datetime

plt.rcParams.update({"figure.dpi": 600, "savefig.dpi": 600})

# ------------------------------
# Helpers
# ------------------------------

def _prod(vals: Iterable[float]) -> float:
    out = 1.0
    for v in vals:
        out *= float(v)
    return out

def _wilson_band(p_hat: np.ndarray, n: int, z: float = 1.959963984540054) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized Wilson 95% CI for a binomial mean with sample size n."""
    p_hat = p_hat.astype(float)
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    half   = z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4.0 * n**2)) / denom
    lo = np.clip(center - half, 0.0, 1.0)
    hi = np.clip(center + half, 0.0, 1.0)
    return lo, hi

# ------------------------------
# Importance factors (Birnbaum, Criticality, FV)
# ------------------------------

def calculate_importance_factors(minimal_cut_sets: List[List[str]],
                                 label_map: Dict[str, int],
                                 failure_probs: Dict[str, float] | np.ndarray) -> pd.DataFrame:
    """
    Match of your 'critical_rank.calculate_importance_factors' but inlined here to keep
    example05 self-contained.

    Returns a DataFrame indexed by BE with columns: Birnbaum, Criticality, Fussell-Vesely.
    """
    # Normalize failure_probs to a vector in label_map order
    if isinstance(failure_probs, dict):
        vec = np.zeros(len(label_map), dtype=float)
        for be, idx in label_map.items():
            vec[idx] = float(failure_probs[be])
        failure_probs = vec

    def top_unavailability(probs) -> float:
        return sum(_prod(probs[label_map[e]] for e in cut) for cut in minimal_cut_sets)

    Q = top_unavailability(failure_probs)

    birnbaum, criticality, fv = {}, {}, {}
    for be, i in label_map.items():
        p_i = failure_probs[i]
        # Birnbaum = ∂Q/∂p_i under SOP(MCS) approximation
        dQ_dp = 0.0
        joint = 0.0
        q_with = 0.0
        for cut in minimal_cut_sets:
            if be in cut:
                others = [e for e in cut if e != be]
                dQ_dp += _prod(failure_probs[label_map[e]] for e in others)
                joint += p_i * _prod(failure_probs[label_map[e]] for e in others)
                q_with += _prod(failure_probs[label_map[e]] for e in cut)
        birnbaum[be]    = dQ_dp
        criticality[be] = (joint / Q) if Q > 0 else 0.0
        fv[be]          = (q_with / Q) if Q > 0 else 0.0

    df = pd.DataFrame({
        "Birnbaum": pd.Series(birnbaum),
        "Criticality": pd.Series(criticality),
        "Fussell-Vesely": pd.Series(fv),
    }).sort_index(key=lambda s: s.str.extract(r'(\d+)').astype(float)[0] if s.dtype==object else None)

    return df

# ------------------------------
# Fault tree from logs (copy+enhance of example04 approach)
# ------------------------------

@dataclass
class FaultTreeFromLogs05:
    fault_logs_path: str

    def __post_init__(self):
        self.fault_logs = pd.read_csv(self.fault_logs_path)
        self.fault_logs['Timestamp'] = pd.to_datetime(self.fault_logs['Timestamp'])
        self.basic_events: List[str] = sorted(self.fault_logs['Basic_Event'].unique(),
                                              key=lambda s: (int(''.join(filter(str.isdigit, s))) if any(ch.isdigit() for ch in s) else 1e9, s))
        self.minimal_cut_sets: List[Set[str]] = []
        self.failure_rates: Dict[str, float] = {}
        self.repair_times: Dict[str, float] = {}
        os.makedirs("output", exist_ok=True)
        os.makedirs("output/sensitivity", exist_ok=True)

    # ---------- Fault tree structure from logs ----------
    def extract_fault_tree(self) -> List[Set[str]]:
        """Extract minimal cut sets from time-ordered fault logs """
        failure_states = set()
        for ts, grp in self.fault_logs.groupby('Timestamp', sort=True):
            # replay state up to ts
            events_up_to_now = self.fault_logs[self.fault_logs['Timestamp'] <= ts]
            active = set()
            for _, ev in events_up_to_now.iterrows():
                if ev['Status'] == 'Active':
                    active.add(ev['Basic_Event'])
                else:
                    active.discard(ev['Basic_Event'])
            if bool(grp['System_Failure'].iloc[-1]):
                failure_states.add(frozenset(active))

        # Reduce to minimal cut sets
        states = [set(s) for s in failure_states]
        states.sort(key=len)
        mcs: List[Set[str]] = []
        for cand in states:
            if not any(existing.issubset(cand) for existing in mcs):
                mcs.append(cand)
        self.minimal_cut_sets = mcs
        return mcs

    # ---------- Parameter estimation from logs ----------
    def estimate_parameters(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        # horizon
        t0 = self.fault_logs['Timestamp'].min()
        t1 = self.fault_logs['Timestamp'].max()
        total_hours = (t1 - t0).total_seconds()/3600.0 if pd.notnull(t0) and pd.notnull(t1) else 0.0

        failure_rates: Dict[str, float] = {}
        repair_times: Dict[str, float] = {}
        for be in self.basic_events:
            df = self.fault_logs[self.fault_logs['Basic_Event'] == be].sort_values('Timestamp')
            n_fail = (df['Status'] == 'Active').sum()
            lam = (n_fail / total_hours) if total_hours > 0 else 0.0

            # MTTR from Active->Cleared gaps
            clears = []
            for i in range(len(df)-1):
                if df.iloc[i]['Status']=='Active' and df.iloc[i+1]['Status']=='Cleared':
                    clears.append((df.iloc[i+1]['Timestamp'] - df.iloc[i]['Timestamp']).total_seconds()/3600.0)
            mttr = float(np.mean(clears)) if clears else 24.0

            failure_rates[be] = float(lam)
            repair_times[be] = float(mttr)

        self.failure_rates = failure_rates
        self.repair_times = repair_times
        return failure_rates, repair_times

    # ---------- Exact analytical R(t) from minimal cut sets ----------
    def _p_TE_given_p(self, p_fail: Dict[str, float]) -> float:
        """
        Exact P(Top Event) assuming independent BEs with failure probabilities p_fail[be],
        using a dynamic recursion over the minimal cut sets.
        """
        vars_all = sorted({be for cut in self.minimal_cut_sets for be in cut},
                          key=lambda s: (int(''.join(filter(str.isdigit, s))) if any(ch.isdigit() for ch in s) else 1e9, s))
        cuts = tuple(frozenset(c) for c in self.minimal_cut_sets)

        from functools import lru_cache

        @lru_cache(None)
        def rec(cuts_state: Tuple[frozenset, ...], idx: int) -> float:
            # If any cut set is emptied → all its members have failed → TE
            for c in cuts_state:
                if len(c) == 0:
                    return 1.0
            if idx >= len(vars_all):
                return 0.0
            v = vars_all[idx]
            p1 = float(p_fail.get(v, 0.0))
            p0 = 1.0 - p1
            # If v fails → remove v from every cut; if v survives → drop any cut that contained v
            cuts_if1 = tuple(frozenset(c - {v}) for c in cuts_state)
            cuts_if0 = tuple(frozenset(c) for c in cuts_state if v not in c)
            return p1 * rec(cuts_if1, idx+1) + p0 * rec(cuts_if0, idx+1)

        return rec(cuts, 0)

    def analytical_R(self, t_hours: float, failure_rates: Dict[str, float] | None = None) -> float:
        """Non‑repairable analytical reliability at time t (exact over MCS)."""
        if not self.minimal_cut_sets:
            self.extract_fault_tree()
        lam = failure_rates if failure_rates is not None else self.failure_rates
        p_fail = {be: 1.0 - math.exp(-float(lam.get(be, 0.0)) * t_hours) for be in self._all_bes()}
        return 1.0 - self._p_TE_given_p(p_fail)

    def _all_bes(self) -> List[str]:
        if self.minimal_cut_sets:
            return sorted({be for cut in self.minimal_cut_sets for be in cut},
                          key=lambda s: (int(''.join(filter(str.isdigit, s))) if any(ch.isdigit() for ch in s) else 1e9, s))
        return self.basic_events

    # ---------- Equivalent analytical failure rates ----------
    def equivalent_lambda_small_t(self) -> float:
        """
        λ_eq at t→0 is the sum of λ over all singleton minimal cut sets,
        because only those contribute linearly to P(TE ≤ t) for small t.
        """
        lam = self.failure_rates
        lam_sum = 0.0
        for cut in self.minimal_cut_sets:
            if len(cut) == 1:
                (be,) = tuple(cut)
                lam_sum += float(lam.get(be, 0.0))
        return float(lam_sum)

    def equivalent_lambda_mttf(self,
                               grid_max: float = 1e6,
                               n_grid: int = 6000,
                               auto_expand: bool = True,
                               expand_factor: float = 4.0,
                               tol_R_tail: float = 1e-8) -> float:
        """
        λ_eq = 1 / ∫_0^∞ R(t) dt. Numerically integrate R(t) and (optionally) expand
        the grid until the tail is negligible.
        """
        T = grid_max
        # Optionally expand T until R(T) is tiny
        if auto_expand:
            for _ in range(6):  # at most 6 expansions
                rT = self.analytical_R(T)
                if rT < tol_R_tail:
                    break
                T *= expand_factor
        # Integrate on [0,T]
        grid = np.linspace(0.0, T, n_grid, dtype=float)
        Rvals = np.array([self.analytical_R(t) for t in grid], dtype=float)
        area = float(np.trapz(Rvals, grid))
        # If tail is still non‑negligible, approximate it by an exponential using local slope
        R_T = Rvals[-1]
        if R_T > 0:
            # local slope near T
            eps = T / n_grid
            R_Tm = self.analytical_R(max(T - eps, 0.0))
            slope = (R_T - R_Tm) / eps if eps > 0 else 0.0
            lam_tail = -slope / max(R_T, 1e-300)
            if lam_tail > 0:
                area += R_T / lam_tail
        lam_eq = 1.0 / area if area > 0 else 0.0
        return float(lam_eq)

    # ---------- Monte Carlo (repairable) with per‑component time‑series ----------
    def _simulate_component(self, lam: float, repair_time: float,
                            T_mission: float, dt: float, rng: np.random.Generator) -> np.ndarray:
        """Return boolean array (True=DOWN) sampled on uniform grid."""
        time_grid = np.arange(0, T_mission + dt, dt)
        down = np.zeros_like(time_grid, dtype=bool)
        t = 0.0
        if lam <= 0.0 and repair_time >= 0.0:
            return down
        while t < T_mission:
            # up to next failure
            if lam > 0.0:
                t += rng.exponential(1.0 / lam)
            else:
                break
            if t >= T_mission:
                break
            start = t
            t = start + repair_time
            end = min(t, T_mission)
            mask = (time_grid >= start) & (time_grid < end)
            down[mask] = True
        return down

    def monte_carlo(self, T_mission: float = 1000.0, N_sim: int = 1000, dt: float = 5.0,
                    seed: int = 42) -> dict:
        if not self.minimal_cut_sets:
            self.extract_fault_tree()
        rng = np.random.default_rng(seed)
        time_grid = np.arange(0, T_mission + dt, dt)
        Nt = len(time_grid)

        # Simulate components
        be_list = self._all_bes()
        comp_down = {be: np.zeros((N_sim, Nt), dtype=bool) for be in be_list}
        for s in range(N_sim):
            for be in be_list:
                comp_down[be][s, :] = self._simulate_component(
                    float(self.failure_rates.get(be, 0.0)),
                    float(self.repair_times.get(be, 24.0)),
                    T_mission, dt, rng
                )

        # System down = OR over MCS (AND within each cut)
        sys_down = np.zeros((N_sim, Nt), dtype=bool)
        for cut in self.minimal_cut_sets:
            cut = list(cut)
            # AND within cut, then OR over cuts
            cut_mask = comp_down[cut[0]].copy()
            for be in cut[1:]:
                cut_mask &= comp_down[be]
            sys_down |= cut_mask

        # Availability & Reliability (means and 95% Wilson)
        A_t = 1.0 - sys_down.mean(axis=0)
        A_lo, A_hi = _wilson_band(A_t, N_sim)

        ever_failed = np.logical_or.accumulate(sys_down, axis=1)
        R_t = 1.0 - ever_failed.mean(axis=0)
        R_lo, R_hi = _wilson_band(R_t, N_sim)

        # System unavailability summary across runs
        sys_unavail_per_sim = sys_down.mean(axis=1)
        sys_unavail_mean = float(sys_unavail_per_sim.mean())
        ci_lo, ci_hi = np.percentile(sys_unavail_per_sim, [2.5, 97.5]).astype(float)

        # Per‑component availability time‑series
        comp_A_ts = {}
        comp_A_lo = {}
        comp_A_hi = {}
        for be in be_list:
            A_be = 1.0 - comp_down[be].mean(axis=0)
            lo, hi = _wilson_band(A_be, N_sim)
            comp_A_ts[be] = A_be
            comp_A_lo[be] = lo
            comp_A_hi[be] = hi

        return {
            "time_grid": time_grid,
            "system": {
                "A_t": A_t, "A_lo": A_lo, "A_hi": A_hi,
                "R_t": R_t, "R_lo": R_lo, "R_hi": R_hi,
                "unavailability_mean": sys_unavail_mean,
                "unavailability_CI": (float(ci_lo), float(ci_hi)),
                "down": sys_down,
            },
            "components": {
                "A_t": comp_A_ts,
                "A_lo": comp_A_lo,
                "A_hi": comp_A_hi,
                "down": comp_down,
            }
        }

    # ---------- Importance (Birnbaum + Criticality + FV) ----------
    def importance_factors(self, mission_time: float = 1000.0) -> pd.DataFrame:
        if not self.minimal_cut_sets:
            self.extract_fault_tree()
        # Prepare inputs
        mcs_as_lists = [list(c) for c in self.minimal_cut_sets]
        be_list = self._all_bes()
        label_map = {be: i for i, be in enumerate(be_list)}
        failure_probs = {be: 1.0 - math.exp(-float(self.failure_rates.get(be,0.0)) * mission_time) for be in be_list}
        return calculate_importance_factors(mcs_as_lists, label_map, failure_probs)

    # ---------- Sensitivity: multiply λ of one BE by factors, re-simulate ----------
    def sensitivity_for_be(self, be: str, multipliers: Iterable[int] = (1,2,3,4,5),
                           T_mission: float = 1000.0, N_sim: int = 500, dt: float = 5.0,
                           seed0: int = 12345) -> pd.DataFrame:
        rows = []
        base = dict(self.failure_rates)
        for i, m in enumerate(multipliers):
            fr = dict(base)
            fr[be] = base.get(be, 0.0) * m
            # Temporarily swap rates
            res = self.monte_carlo(T_mission=T_mission, N_sim=N_sim, dt=dt, seed=seed0+i) if fr == self.failure_rates else \
                  FaultTreeFromLogs05(self.fault_logs_path)._sensitivity_reuse(self.minimal_cut_sets, fr, self.repair_times,
                                                                               T_mission, N_sim, dt, seed0+i)
            mean_u = res["system"]["unavailability_mean"]
            lo, hi = res["system"]["unavailability_CI"]
            rows.append({
                "BE": be, "Multiplier": int(m),
                "FailureRate (/h)": fr[be],
                "System Unavailability (mean)": mean_u,
                "CI low (2.5%)": float(lo),
                "CI high (97.5%)": float(hi),
            })
        df = pd.DataFrame(rows)
        out_csv = f"output/sensitivity/{be}_sensitivity.csv"
        df.to_csv(out_csv, index=False)
        return df

    def _sensitivity_reuse(self, minimal_cut_sets, failure_rates, repair_times, T_mission, N_sim, dt, seed):
        """Internal helper to run MC with provided (mcs, rates) without re-reading logs."""
        # Clone-like instance
        self.minimal_cut_sets = minimal_cut_sets
        self.failure_rates = failure_rates
        self.repair_times = repair_times
        return self.monte_carlo(T_mission=T_mission, N_sim=N_sim, dt=dt, seed=seed)

    def sensitivity_all_bes(self, multipliers: Iterable[int] = (1,2,3,4,5),
                            T_mission: float = 1000.0, N_sim: int = 500, dt: float = 5.0,
                            seed0: int = 23456) -> Dict[str, pd.DataFrame]:
        out = {}
        for k, be in enumerate(self._all_bes()):
            out[be] = self.sensitivity_for_be(
                be, multipliers, T_mission=T_mission, N_sim=N_sim, dt=dt, seed0=seed0 + 1000*k
            )
        # Create a single combined figure after gathering all BE tables
        self._plot_sensitivity_combined(out)
        return out

    # ---------- Plotting ----------
    def plot_system_and_components(self, mc_res: dict, title_suffix: str = "") -> None:
        t = mc_res["time_grid"]
        sys = mc_res["system"]
        comp = mc_res["components"]

        # System availability with 95% band
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(t, sys["A_t"], lw=1.8, label="System A(t)")
        # ax.fill_between(t, sys["A_lo"], sys["A_hi"], alpha=0.18, label="95% CI")
        ax.set_xlabel("Time (hours)", fontsize=14)
        ax.set_ylabel("Availability", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        # ax.set_title(f"System Availability")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, 1000)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)
        fig.tight_layout()
        fig.savefig("output/availability_system.png", bbox_inches="tight", dpi=600)
        plt.close(fig)

        # Components: plot a few per panel to keep legible
        be_list = self._all_bes()
        fig, ax = plt.subplots(figsize=(11,6))
        # Plot system availability on the same axes for context
        ax.plot(t, sys["A_t"], lw=2.5, label="System A(t)")
        for be in be_list:
            ax.plot(t, comp["A_t"][be], lw=1.0, alpha=0.9, label=be)
        ax.set_xlabel("Time (hours)", fontsize=14)
        ax.set_ylabel("Availability", fontsize=14)
        # ax.set_title(f"Component Availability Time‑Series")
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.0, 1000)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)
        ax.legend(ncols=3, fontsize=14)
        fig.tight_layout()
        fig.savefig("output/availability_components.png", bbox_inches="tight", dpi=600)
        plt.close(fig)

        # Export per‑component availability time‑series
        df = pd.DataFrame({"time_hours": t})
        for be in be_list:
            df[f"{be}_A(t)"] = comp["A_t"][be]
        df.to_csv("output/component_availability_timeseries.csv", index=False)

    def _plot_sensitivity_single(self, df: pd.DataFrame, be: str) -> None:
        fig, ax = plt.subplots(figsize=(7.2,4.2))
        x = df["Multiplier"].to_numpy()
        y = df["System Unavailability (mean)"].to_numpy(float)
        lo = df["CI low (2.5%)"].to_numpy(float)
        hi = df["CI high (97.5%)"].to_numpy(float)
        yerr = np.vstack([y - lo, hi - y])

        ax.errorbar(x, y, yerr=yerr, fmt="o", linestyle="none", capsize=5, lw=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"x{int(v)}" for v in x])
        ax.set_xlabel("Increase Factor for failure rate of " + be, fontsize=14)
        ax.set_ylabel("System Unavailability", fontsize=14)
        ax.set_title(f"Sensitivity of System Unavailability to {be}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = f"output/sensitivity/{be}_sensitivity.png"
        fig.savefig(out_png, bbox_inches="tight", dpi=600)
        plt.close(fig)



    def _plot_sensitivity_combined(self, dfs_by_be: Dict[str, pd.DataFrame], *, offset_width: float = 0.15) -> None:
        """One sensitivity figure with a series per BE, horizontally offset to avoid overlap."""
        import numpy as np
        # Shared x-centers inferred from the first DF
        first_df = next(iter(dfs_by_be.values()))
        centers = first_df["Multiplier"].to_numpy(float)
        n_series = len(dfs_by_be)
        offsets = np.zeros(n_series) if n_series == 1 else np.linspace(-offset_width, offset_width, n_series)

        fig, ax = plt.subplots(figsize=(9, 5))
        # Plot each BE at a small horizontal offset (markers only, error bars)
        for j, (be, df) in enumerate(dfs_by_be.items()):
            x  = df["Multiplier"].to_numpy(float) + offsets[j]
            y  = df["System Unavailability (mean)"].to_numpy(float)
            lo = df["CI low (2.5%)"].to_numpy(float)
            hi = df["CI high (97.5%)"].to_numpy(float)
            yerr = np.vstack([y - lo, hi - y])
            # Matplotlib cycles colors by default; use same for markers and e-bars
            ax.errorbar(x, y, yerr=yerr, fmt="o", linestyle="none", capsize=5, lw=1.5, label=be)

        # Ticks at the centers (no offset), labeled x1..xM
        ax.set_xticks(centers)
        ax.set_xticklabels([f"x{int(v)}" for v in centers], fontsize=14)
        
        # Room for offsets at both ends
        pad = 0.4 + offset_width
        ax.set_xlim(centers.min() - pad, centers.max() + pad)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlabel("Increase Factor for failure rate", fontsize=14)
        ax.set_ylabel("System Unavailability", fontsize=14)
        # ax.set_title("Sensitivity of System Unavailability")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, ncols=2)
        fig.tight_layout()
        fig.savefig("output/combined_sensitivity.png", bbox_inches="tight", dpi=600)
        plt.close(fig)

# ------------------------------
# Demo / CLI
# ------------------------------

def run_demo():
    logs = "fault_logs_example.csv"
    if not os.path.exists(logs):
        print(f"[demo] '{logs}' not found. Demo will skip MC & sensitivity. "
              "Place your log CSV next to this script to run the full demo.")
        return

    print("="*68)
    print("EXAMPLE05 — COMPLETE FAULT TREE ANALYSIS WITH ENHANCEMENTS")
    print("="*68)
    ft = FaultTreeFromLogs05(logs)

    print("\n[1] Extracting minimal cut sets from logs…")
    mcs = ft.extract_fault_tree()
    print(f"  • Found {len(mcs)} minimal cut sets: {['{' + ','.join(sorted(s)) + '}' for s in mcs]}")

    print("\n[2] Estimating BE failure rates & repair times from logs…")
    failure_rates, repair_times = ft.estimate_parameters()

    # Analytical reliability + equivalent λ
    print("\n[3] Analytical Reliability & Equivalent λ")
    sample_times = [0, 100, 250, 500, 1000]
    for t in sample_times:
        print(f"  R({t:5.1f} h) = {ft.analytical_R(t):.6f}")
    lam_small_t = ft.equivalent_lambda_small_t()
    lam_mttf   = ft.equivalent_lambda_mttf()
    print(f"  λ_eq (small‑time slope): {lam_small_t:.6e} /h")
    print(f"  λ_eq (1/∫R):            {lam_mttf:.6e} /h")

    # Importance (Birnbaum + Criticality + FV) at mission time
    print("\n[4] Importance factors @ T=1000 h (Birnbaum • Criticality • F‑V)")
    imp = ft.importance_factors(mission_time=1100.0)
    print(imp)

    # Monte Carlo
    print("\n[5] Monte Carlo (repairable) with per‑component availability time‑series…")
    mc = ft.monte_carlo(T_mission=1100.0, N_sim=800, dt=5.0, seed=17)
    print(f"  System unavailability (mean): {mc['system']['unavailability_mean']:.6g} "
          f"[95% CI: {mc['system']['unavailability_CI'][0]:.6g}, {mc['system']['unavailability_CI'][1]:.6g}]")
    ft.plot_system_and_components(mc, title_suffix=" — 1000h mission")

    # Sensitivity (one plot per BE, x1..x5)
    print("\n[6] Sensitivity: multiplying each BE’s λ by x1..x5 and re‑simulating…")
    _ = ft.sensitivity_all_bes(multipliers=(1,2,3,4,5), T_mission=1000.0, N_sim=400, dt=5.0, seed0=123)

    print("\nOutputs written to ./output/  (plots and CSVs).")

if __name__ == "__main__":
    run_demo()

    def _plot_sensitivity_combined(self, dfs_by_be: Dict[str, pd.DataFrame],
                                   offset_width: float = 0.15,
                                   markersize: float = 6,
                                   capsize: float = 5,
                                   elinewidth: float = 1.5,
                                   capthick: float = 1.5,
                                   markeredgewidth: float = 1.0) -> None:
        """One sensitivity figure with a small horizontal offset per BE to avoid overlap."""
        import numpy as np
        fig, ax = plt.subplots(figsize=(9, 5))
        # Centers (true multipliers)
        first_df = next(iter(dfs_by_be.values()))
        centers = first_df["Multiplier"].to_numpy(float)
        # Offsets for each BE series
        n_series = len(dfs_by_be)
        offsets = np.zeros(n_series) if n_series <= 1 else np.linspace(-offset_width, offset_width, n_series)
        # Plot each BE with an offset (markers only; no connecting lines)
        for j, (be, df) in enumerate(dfs_by_be.items()):
            x = df["Multiplier"].to_numpy(float) + offsets[j]
            y = df["System Unavailability (mean)"].to_numpy(float)
            lo = df["CI low (2.5%)"].to_numpy(float)
            hi = df["CI high (97.5%)"].to_numpy(float)
            yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(
                x, y, yerr=yerr,
                fmt="o", linestyle="none",
                capsize=capsize, markersize=markersize,
                lw=1.5, elinewidth=elinewidth, capthick=capthick, markeredgewidth=markeredgewidth,
                label=be
            )
        # Keep ticks at the centers (1..M) while points are slightly offset
        ax.set_xticks(centers)
        ax.set_xticklabels([f"x{int(v)}" for v in centers])
        # Room for offsets at both ends
        pad = 0.4 + offset_width
        ax.set_xlim(centers.min() - pad, centers.max() + pad)
        ax.set_xlabel("Increase Factor for failure rate", fontsize=14)
        ax.set_ylabel("System Unavailability", fontsize=14)
        ax.set_title("Sensitivity of System Unavailability (All BEs)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, ncols=2)
        fig.tight_layout()
        fig.savefig("output/sensitivity/combined_sensitivity.png", bbox_inches="tight", dpi=600)
        plt.close(fig)
