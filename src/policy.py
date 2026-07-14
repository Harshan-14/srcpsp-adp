"""
Online barrier policy using Ĵ* from Bellman iteration.

Barrier method: pick the legal decision minimizing E[j_delta + Ĵ*(s')].
The expectation is computed by reusing bellman._expected_val so the
online policy is guaranteed to be consistent with the Ĵ* it was
trained to.

If every legal decision is barred (all outcomes lead to states outside
the confined subset — possible on held-out trajectories that hit
realization combinations never observed during training), fall back
to H1 so the simulator keeps making progress. This IS the paper's
"heuristic detour" pattern: use H1 to skip past the unknown region;
Ĵ* resumes on the next well-defined state (the Markov property makes
this valid).

Design decisions locked:
 1. Reuse bellman._expected_val rather than reimplement — one source of
    truth for the expectation calculation.
 2. Fallback to H1 only when best_val == +∞ (all decisions barred).
    Otherwise argmin over the finite candidates.
 3. Deterministic — no RNG. Same state + same Ĵ* → same decision.
 4. Provided as both a direct function (adp_policy) and a closure
    factory (make_adp_policy) that produces a plain PolicyFn compatible
    with simulator.run_trajectory / build_subset_and_J0.
"""
from typing import Callable, Dict, Set

from .bellman import _expected_val
from .heuristic import heuristic_1
from .models import ProblemSpec
from .state import State, is_terminal
from .transitions import Decision, enumerate_decisions


INF = float("inf")


def adp_policy(
    state: State,
    spec: ProblemSpec,
    J_star: Dict[State, float],
    subset: Set[State],
) -> Decision:
    """Barrier method — argmin over legal decisions of E[j_delta + Ĵ*(s')].

    Fallback to heuristic_1 when all decisions are barred.
    """
    if is_terminal(state, spec):
        return ()

    legal = enumerate_decisions(state, spec)
    if not legal:
        return ()

    best_val = INF
    best_d = None
    for d in legal:
        v = _expected_val(state, d, J_star, subset, spec)
        if v < best_val:
            best_val = v
            best_d = d

    if best_d is None or best_val == INF:
        # All decisions barred — H1 fallback (Markov property makes
        # resuming Ĵ* valid on the next well-defined state).
        return heuristic_1(state, spec)

    return best_d


def make_adp_policy(
    J_star: Dict[State, float],
    subset: Set[State],
) -> Callable[[State, ProblemSpec], Decision]:
    """Return a plain PolicyFn (state, spec) -> Decision compatible with
    simulator.run_trajectory and simulator.build_subset_and_J0. Encloses
    J_star and subset."""
    def _policy(state: State, spec: ProblemSpec) -> Decision:
        return adp_policy(state, spec, J_star, subset)
    return _policy