"""
Bellman iteration on the confined state subset.

Iterates the Bellman operator until relative convergence:

    Ĵ_new(s) = min over legal decisions d of  E[j_delta + Ĵ(s')]

where the expectation is over the joint realization outcomes of tasks
started by decision d. E is computed EXACTLY (not by Monte Carlo) by
enumerating outcomes and their probabilities.

Missing-next-state handling — Type 1 barrier + Type 2 renormalization
hybrid (paper's Phase 4):

  For each decision, enumerate all realization outcomes. For each
  outcome, compute the resulting next-state s'.
    - If s' ∈ subset:  contribute (joint_prob * (j_delta + Ĵ(s'))) to
                       the numerator and joint_prob to the denominator.
    - If s' ∉ subset:  skip.

  Then E[...] = numerator / denominator, renormalizing over observed
  outcomes (Type 2). If denominator = 0 (no outcomes observed for this
  decision), return +∞ — the decision is barred (Type 1).

Design decisions locked:
 1. Exact expectation, not Monte Carlo. Outcomes per decision are at
    most 3^(num_labs) = 9 for our 2-lab problem, so enumeration is cheap.
 2. Type 1 + Type 2 as a single unified normalization step. Cleaner than
    branching on "any missing" vs "all missing".
 3. Terminal states pinned to Ĵ = 0 every iteration. Prevents drift.
 4. Relative convergence: max |ΔĴ| / max |Ĵ| < tolerance. Default 0.01
    matches paper's 1%. Absolute would be scale-brittle.
 5. If all decisions from a non-terminal state are barred (should not
    happen for H1-visited states, but defensively), keep the previous
    Ĵ(s) value rather than propagating +∞.
"""
from itertools import product
from typing import Dict, Set, Tuple

from .models import ProblemSpec
from .reward import accrue
from .state import State, is_terminal
from .transitions import (
    Decision,
    advance_to_next_event,
    enumerate_decisions,
)


INF = float("inf")


# ---------------------------------------------------------------------------
# _apply_decision_det (private)
# ---------------------------------------------------------------------------

def _apply_decision_det(
    state: State,
    decision: Decision,
    realizations: Tuple[int, ...],
    spec: ProblemSpec,
) -> State:
    """Deterministic variant of apply_decision. Uses the supplied
    realization indices instead of sampling. Used only for expectation
    enumeration inside Bellman iteration; kept out of transitions.py
    to avoid loading the simulator-facing API with Bellman-specific
    helpers."""
    if len(realizations) != len(decision):
        raise ValueError("realizations length must match decision length")

    next_task_l = list(state.next_task)
    lab_owner_l = list(state.lab_owner)
    lab_realization_l = list(state.lab_realization)
    lab_time_in_use_l = list(state.lab_time_in_use)

    for (project_id, lab_id), r in zip(decision, realizations):
        i = project_id - 1
        j = lab_id - 1
        task_idx = next_task_l[i]

        lab_owner_l[j] = project_id
        lab_realization_l[j] = r
        lab_time_in_use_l[j] = 0
        next_task_l[i] = task_idx + 1

    return State(
        next_task=tuple(next_task_l),
        z=state.z,
        lab_owner=tuple(lab_owner_l),
        lab_realization=tuple(lab_realization_l),
        lab_time_in_use=tuple(lab_time_in_use_l),
        t=state.t,
    )


# ---------------------------------------------------------------------------
# _outcome_probs (private)
# ---------------------------------------------------------------------------

def _outcome_probs(state: State, decision: Decision, spec: ProblemSpec):
    """For each pair in decision, list [(realization_idx, prob), ...]
    with only strictly positive probabilities included."""
    lists = []
    for project_id, _lab_id in decision:
        i = project_id - 1
        task_idx = state.next_task[i]
        project = spec.projects[i]
        if task_idx == 0:
            probs = project.PI
        else:
            assert state.z[i] >= 0, (
                "prev_realization must be set for task_idx > 0"
            )
            probs = project.PM[task_idx - 1][:, state.z[i]]
        num_r = project.tasks[task_idx].num_realizations()
        lists.append([
            (k, float(probs[k])) for k in range(num_r) if probs[k] > 0
        ])
    return lists


# ---------------------------------------------------------------------------
# _expected_val (private)
# ---------------------------------------------------------------------------

def _expected_val(
    s: State,
    d: Decision,
    J: Dict[State, float],
    subset: Set[State],
    spec: ProblemSpec,
) -> float:
    """E[j_delta + Ĵ(s')] for state s under decision d.

    Renormalization + barrier hybrid: numerator/denominator accumulate
    only over outcomes whose next-state is in `subset`. Returns +∞ if
    no outcome lands in-subset.
    """
    if not d:
        # No tasks started; deterministic advance.
        s_next, events = advance_to_next_event(s, spec)
        j_delta = 0.0
        for e in events:
            c, r = accrue(e, s_next, spec)
            j_delta += c - r
        if s_next in subset:
            return j_delta + J[s_next]
        return INF

    outcome_lists = _outcome_probs(s, d, spec)
    numerator = 0.0
    denominator = 0.0

    for combo in product(*outcome_lists):
        realizations = tuple(r for r, _p in combo)
        joint_p = 1.0
        for _r, p in combo:
            joint_p *= p
        if joint_p == 0.0:
            continue

        s_after = _apply_decision_det(s, d, realizations, spec)
        s_next, events = advance_to_next_event(s_after, spec)
        j_delta = 0.0
        for e in events:
            c, r = accrue(e, s_next, spec)
            j_delta += c - r

        if s_next in subset:
            numerator += joint_p * (j_delta + J[s_next])
            denominator += joint_p
        # else: drop this outcome (Type 2 renormalization)

    if denominator == 0.0:
        return INF
    return numerator / denominator


# ---------------------------------------------------------------------------
# bellman_iterate (public)
# ---------------------------------------------------------------------------

def bellman_iterate(
    subset: Set[State],
    J0: Dict[State, float],
    spec: ProblemSpec,
    tolerance: float = 0.01,
    max_iters: int = 100,
) -> Tuple[Dict[State, float], int]:
    """Iterate the Bellman operator on `subset` until convergence.

    Parameters
    ----------
    subset      : the confined state domain from simulator.
    J0          : per-state Monte Carlo starting estimate.
    tolerance   : relative convergence threshold (default 0.01 = 1%).
    max_iters   : hard cap on iterations.

    Returns
    -------
    (Ĵ, iterations)
        Ĵ : dict mapping each state in subset to its final cost-to-go.
        iterations : number of iterations actually run.
    """
    J = dict(J0)
    # Pin terminal states.
    for s in subset:
        if is_terminal(s, spec):
            J[s] = 0.0

    for it in range(1, max_iters + 1):
        J_new: Dict[State, float] = {}
        max_change = 0.0

        for s in subset:
            if is_terminal(s, spec):
                J_new[s] = 0.0
                continue

            legal = enumerate_decisions(s, spec)
            if not legal:
                J_new[s] = 0.0
                continue

            best = INF
            for d in legal:
                v = _expected_val(s, d, J, subset, spec)
                if v < best:
                    best = v

            if best == INF:
                # All decisions barred — keep previous value.
                J_new[s] = J.get(s, 0.0)
            else:
                J_new[s] = best

            change = abs(J_new[s] - J.get(s, 0.0))
            if change > max_change:
                max_change = change

        # Relative convergence.
        max_J = max((abs(v) for v in J_new.values() if v != INF), default=1.0)
        if max_J == 0.0:
            max_J = 1.0

        J = J_new
        if max_change / max_J < tolerance:
            return J, it

    return J, max_iters