"""
Scheduling heuristics for the sRCPSP.

Scope B implements ONE heuristic — H1: Highest Success Probability. It is
used in two places:

    1. During heuristic simulation to generate the confined state subset
       that Bellman iteration will later operate on.
    2. As the fallback policy for the barrier method when it needs to
       skip past a state whose cost-to-go is not defined in the subset.

The other two heuristics from the paper (H2: shortest expected duration,
H3: highest reward priority) are DEFERRED per Scope B. If restored later,
they would live as sibling functions in this file.

Design decisions locked (see chat notes):

 1. Greedy per-lab, not decision enumeration. For each free lab, pick
    the single eligible project with highest P(success). Faithful to
    H1's semantics; O(free_labs * num_projects) rather than exponential.
 2. "Success" = realization result string != "fail". Robust to variants
    with different fail-index conventions.
 3. Tie-break by lowest project_id. Deterministic, reproducible, no RNG.
 4. Aggressive filling — never voluntarily leaves a lab idle if any
    eligible exists. Timing sophistication belongs to policy.py, not H1.
 5. state.z[i] drives PM column choice. PI used only when next_task[i]
    == 0 (project has completed no task yet). Consistent with
    markov.sample_realization and the locked PM shape convention.
 6. Decision returned is a sorted tuple of (project_id, lab_id) pairs
    — canonical form matches transitions.py Decision alias, so H1's
    output is directly consumable by apply_decision.
"""
from typing import Tuple

from .models import Project, ProblemSpec
from .state import State
from .transitions import Decision


# ---------------------------------------------------------------------------
# _success_probability (private)
# ---------------------------------------------------------------------------

def _success_probability(
    project: Project,
    task_idx: int,
    prev_realization: int,
) -> float:
    """P(next task's result != 'fail') for project's task at task_idx.

    Distribution source:
        task_idx == 0 -> project.PI
        task_idx  > 0 -> project.PM[task_idx - 1][:, prev_realization]

    Fail-index identification: scan the task's realization triples for
    entries with result == "fail". Robust to data where fail is not at
    index 0 or where multiple realizations share the "fail" label.

    Assumes prev_realization is a valid index into the previous task's
    realizations when task_idx > 0. Invariant is guaranteed by the
    simulator (advance_to_next_event sets z[i] on completion), so no
    defensive checks here.
    """
    if task_idx == 0:
        probs = project.PI
    else:
        probs = project.PM[task_idx - 1][:, prev_realization]

    task = project.tasks[task_idx]
    fail_prob = 0.0
    for k, (result, _, _) in enumerate(task.realizations):
        if result == "fail":
            fail_prob += float(probs[k])

    return 1.0 - fail_prob


# ---------------------------------------------------------------------------
# heuristic_1
# ---------------------------------------------------------------------------

def heuristic_1(state: State, spec: ProblemSpec) -> Decision:
    """H1 — Highest Success Probability.

    For each free lab, select the eligible project whose next task has
    the greatest probability of a non-'fail' realization. Ties broken by
    lowest project_id. If a free lab has no eligible project, it stays
    idle.

    Returns
    -------
    Decision
        Sorted tuple of (project_id, lab_id) pairs. May be empty () if
        no free lab has any eligible project — legal only when at least
        one lab is currently busy (invariant preserved by transitions.py).

    Notes
    -----
    Because each task fixes its required lab, a project is eligible for
    at most one lab at any moment. Consequently the per-lab greedy loop
    cannot accidentally assign the same project to two labs — no
    duplicate check needed.
    """
    num_projects = len(state.next_task)
    num_labs = len(state.lab_owner)

    free_labs = [j + 1 for j in range(num_labs) if state.lab_owner[j] == 0]
    running_projects = {owner for owner in state.lab_owner if owner != 0}

    pairs = []

    for lab_id in free_labs:
        candidates = []  # list of (score, project_id) for this lab

        for i in range(num_projects):
            project_id = i + 1
            if project_id in running_projects:
                continue
            if state.next_task[i] >= spec.projects[i].num_tasks():
                continue

            project = spec.projects[i]
            task_idx = state.next_task[i]
            if project.tasks[task_idx].lab != lab_id:
                continue

            score = _success_probability(project, task_idx, state.z[i])
            candidates.append((score, project_id))

        if not candidates:
            continue  # lab stays idle

        # Highest score wins; tie-break by lowest project_id.
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_project_id = candidates[0][1]
        pairs.append((best_project_id, lab_id))

    return tuple(sorted(pairs))
