import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_importance_factors(mcs, label_map, failure_probs):
    def prod(iterable):
        p = 1
        for x in iterable:
            p *= x
        return p

    def top_prob(probs):
        return sum(prod(probs[label_map[e]] for e in cut) for cut in mcs)

    Q0 = top_prob(failure_probs)
    structure_importance = {}
    for ev in label_map:
        terms = []
        for cut in mcs:
            if ev in cut:
                Nj = len(cut)
                terms.append(1 - 1 / (2 ** (Nj - 1)))
        I_phi = 1 - prod(terms) if terms else 0
        structure_importance[label_map[ev]] = I_phi

    results = []
    for ev in label_map:
        code = label_map[ev]
        probs1 = failure_probs.copy()
        probs0 = failure_probs.copy()
        probs1[code] = 1.0
        probs0[code] = 0.0
        Q1 = top_prob(probs1)
        Q0i = top_prob(probs0)
        IB = Q1 - Q0i
        qi = failure_probs[code]
        ICR = IB * qi / Q0 if Q0 > 0 else None
        FV = sum(
            prod(failure_probs[label_map[e]] for e in cut)
            for cut in mcs if ev in cut
        )
        IFV = FV / Q0 if Q0 > 0 else None
        results.append({
            'Event': code,
            'Structure': structure_importance[code],
            'Birnbaum': IB,
            'Criticality': ICR,
            'Fussell-Vesely': IFV
        })
    df = pd.DataFrame(results).set_index('Event')
    return df.sort_values(by='Criticality', ascending=False)


def simulate_reliability(minimal_cut_sets, failure_rates, repair_times, T=1000, dt=1, N_SIM=5000):

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
            component_stats[be]['up_times'].append(np.mean(~comp_states[be]))

    # Component-level metrics
    comp_df = pd.DataFrame({
        be: {
            'Failure Rate': failure_rates[be],
            'Unavailability': 1 - np.mean(stats['up_times']),
            'MTBF': np.mean(stats['up_times']) * T / len(stats['down_times']) if stats['down_times'] else np.nan,
            'MTTR': np.mean(stats['down_times']) if stats['down_times'] else np.nan
        }
        for be, stats in component_stats.items()
    }).T

    # System-level metrics
    sys_unavail = 1 - np.mean(sys_avail)
    sys_ci = np.percentile(sys_avail, [2.5, 97.5])
    MTTF_sys = np.mean(top_event_up_times) if top_event_up_times else None
    MTTR_sys = np.mean(top_event_down_times) if top_event_down_times else None
    availability_sys = (
        MTTF_sys / (MTTF_sys + MTTR_sys) if MTTF_sys is not None and MTTR_sys is not None else None
    )

    # Time series for plotting
    availability_time_series = 1 - np.mean(top_event_states, axis=0)

    return (
            sys_unavail,
            sys_ci,
            comp_df,
            availability_time_series,
            time_grid,
            all_component_states,
            availability_sys,
            MTTF_sys,
            MTTR_sys
            )



#%% Ploting

# import matplotlib.pyplot as plt

# def plot_unavailability(comp_df, sys_unavail):
#     plt.figure(figsize=(12, 6))

#     # Plot component unavailabilities
#     plt.bar(comp_df.index, comp_df['Unavailability'], label='Basic Events', color='gray')

#     # Plot top event
#     # plt.bar('Top Event', sys_unavail, color='red', label='Top Event')

#     # Formatting
#     plt.ylabel('Unavailability')
#     plt.title('Unavailability of Basic Events and Top Event')
#     plt.xticks(rotation=90)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
    