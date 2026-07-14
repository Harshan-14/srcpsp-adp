"""
Trajectory generator for the sRCPSP simulation.

Two responsibilities:
  1. run_trajectory:      one heuristic-driven trajectory from initial
                          to terminal. Returns a list of (state, tail_J)
                          pairs — one per decision-epoch state visited,
                          with tail_J being the cost-to-go from that
                          state onward under the policy.
  2. build_subset_and_J0: aggregates N trajectories into the confined
                          state subset (union of visited states) and
                          Ĵ⁰ (per-state Monte Carlo mean tail-J).

Both feed directly into bellman.py.

Design decisions locked:
 1. Single shared RNG across all N trajectories. Seed at the top of the
    run controls all sampling; independent per-trajectory RNGs would add
    complexity without reproducibility benefit.
 2. tail_J computed forward via cumulative J tracking, then differenced
    at the end. Simpler than backward-accumulation and gives identical
    numbers.
 3. State is hashable (frozen dataclass, tuple fields), so we use it
    directly as dict/set key — no separate state-id scheme needed.
 4. States within a single trajectory are unique because t is
    monotonically increasing, so no intra-trajectory correlation in
    Ĵ⁰ samples.
"""
from typing import Callable, Dict, List, Set, Tuple

from .models import ProblemSpec
from .reward import accrue
from .state import State, initial_state, is_terminal
from .transitions import (
    Decision,
    advance_to_next_event,
    apply_decision,
    enumerate_decisions,
)


PolicyFn = Callable[[State, ProblemSpec], Decision]


def run_trajectory(
    spec: ProblemSpec,
    policy: PolicyFn,
    rng,
) -> List[Tuple[State, float]]:
    """One policy-driven trajectory from initial state to terminal.

    Returns
    -------
    List[Tuple[State, float]]
        (state, tail_J) pairs, one per decision-epoch state visited,
        including terminal. tail_J is the sum of (cost - reward) for
        all events occurring from `state` onward in this trajectory.
    """
    s = initial_state(spec)
    visited: List[Tuple[State, float]] = [(s, 0.0)]
    J_cum = 0.0

    while not is_terminal(s, spec):
        legal = enumerate_decisions(s, spec)
        if not legal:
            break

        d = policy(s, spec)
        s = apply_decision(s, d, spec, rng)

        if any(s.lab_owner):
            s, events = advance_to_next_event(s, spec)
            for e in events:
                cost, reward = accrue(e, s, spec)
                J_cum += cost - reward

        visited.append((s, J_cum))

    total_J = J_cum
    return [(state, total_J - J_before) for state, J_before in visited]


def build_subset_and_J0(
    spec: ProblemSpec,
    policy: PolicyFn,
    num_trajectories: int,
    rng,
) -> Tuple[Set[State], Dict[State, float]]:
    """Run `num_trajectories` trajectories; aggregate the visited states
    into a subset and per-state mean tail-J into Ĵ⁰.

    Returns
    -------
    (subset, J0)
        subset : set of every State visited at least once.
        J0     : dict mapping each State to its Monte Carlo mean tail_J.
    """
    samples: Dict[State, List[float]] = {}
    for _ in range(num_trajectories):
        traj = run_trajectory(spec, policy, rng)
        for state, tail_J in traj:
            samples.setdefault(state, []).append(tail_J)

    subset = set(samples.keys())
    J0 = {s: sum(vs) / len(vs) for s, vs in samples.items()}
    return subset, J0