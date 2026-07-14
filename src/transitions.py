"""
State-evolution logic for the sRCPSP simulation.

This is the only module that produces new States from old ones. Everything
else reads state, this file writes it. Three responsibilities, three
functions, cleanly separated:

    enumerate_decisions      — pure, no RNG. Lists legal decisions.
    apply_decision           — starts chosen tasks. Samples realizations.
                               Does NOT advance time.
    advance_to_next_event    — advances the clock to the earliest task
                               completion. Frees the completing lab(s),
                               records realization in z[i]. Returns the
                               resulting decision-epoch state PLUS an
                               explicit list of completion events (which
                               reward.py will consume later).

Design decisions locked (see chat notes):

 1. next_task[i] increments at task START. In-flight task index on lab j
    is next_task[lab_owner[j] - 1] - 1.
 2. Empty decision () is legal iff at least one lab is currently busy.
    Prevents time from freezing.
 3. A task fixes its lab. So a project is eligible for at most one lab
    at any moment — enumeration is a clean product over free labs.
 4. Decision = sorted tuple of (project_id, lab_id) pairs. Canonical form
    → hash-equality for free.
 5. Sampling at task START, applying effect at task COMPLETION. Matches
    the state.py convention: lab_realization[j] populated on start,
    z[i] populated on completion.
 6. advance_to_next_event returns (new_state, completed_events). Explicit
    hand-off rather than diff-inference.
 7. Simultaneous completions handled in a single call — all finishing
    labs freed at once.
 8. NO special failure handling. Path B PMs do not zero the failure
    column, so recovery is modeled. Any reward penalty for failure is
    reward.py's concern, not transitions.py's.
 9. Terminal state → enumerate returns []. Empty tuple return [()]
    signals "only wait is legal."

Sentinels preserved: 0 = lab free, -1 = no realization.
"""
from itertools import product
from typing import List, Tuple

from .markov import sample_realization
from .models import ProblemSpec
from .state import State


# Type aliases for clarity.
Decision = Tuple[Tuple[int, int], ...]
# (project_id, task_idx, realization_idx, completion_time)
CompletedEvent = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Small helpers (private)
# ---------------------------------------------------------------------------

def _tuple_set(t: Tuple[int, ...], idx: int, val: int) -> Tuple[int, ...]:
    """Return a new tuple with position `idx` replaced by `val`."""
    lst = list(t)
    lst[idx] = val
    return tuple(lst)


# ---------------------------------------------------------------------------
# enumerate_decisions
# ---------------------------------------------------------------------------

def enumerate_decisions(state: State, spec: ProblemSpec) -> List[Decision]:
    """List all legal decisions from `state`.

    Returns
    -------
    List[Decision]
        Empty list at terminal state. Otherwise a list of decisions,
        each a sorted tuple of (project_id, lab_id) pairs. The empty
        tuple () appears only if at least one lab is currently busy.

    Legality rules:
      • project must have unstarted work
      • project must not already be running on any lab
      • the project's next task's required lab must be `lab_id`
      • no lab is scheduled to more than one project (auto-satisfied since
        each project's next task fixes its lab)
    """
    num_projects = len(state.next_task)
    num_labs = len(state.lab_owner)

    # Terminal check: all work done AND no lab still running.
    all_done = all(
        state.next_task[i] >= spec.projects[i].num_tasks()
        for i in range(num_projects)
    )
    any_busy = any(owner != 0 for owner in state.lab_owner)

    if all_done and not any_busy:
        return []

    # Which labs are free right now?
    free_labs: List[int] = [
        j + 1 for j in range(num_labs) if state.lab_owner[j] == 0
    ]

    # Projects currently in flight cannot be scheduled again.
    running_projects = {owner for owner in state.lab_owner if owner != 0}

    # For each free lab, list project_ids eligible to start on it.
    eligibility: List[List[int]] = []
    for lab_id in free_labs:
        eligible = []
        for i in range(num_projects):
            project_id = i + 1
            if project_id in running_projects:
                continue
            if state.next_task[i] >= spec.projects[i].num_tasks():
                continue
            if spec.projects[i].tasks[state.next_task[i]].lab == lab_id:
                eligible.append(project_id)
        eligibility.append(eligible)

    # Choices per free lab: schedule one of its eligibles, or skip (0).
    choices_per_lab = [[0] + elig for elig in eligibility]

    decisions_set = set()
    for combo in product(*choices_per_lab):
        pairs = tuple(sorted(
            (choice, free_labs[k])
            for k, choice in enumerate(combo)
            if choice != 0
        ))
        decisions_set.add(pairs)

    # Empty decision illegal when no lab is busy — time would freeze.
    if not any_busy:
        decisions_set.discard(())

    return sorted(decisions_set)


# ---------------------------------------------------------------------------
# apply_decision
# ---------------------------------------------------------------------------

def apply_decision(
    state: State,
    decision: Decision,
    spec: ProblemSpec,
    rng,
) -> State:
    """Start the tasks specified in `decision`. Sample realizations.

    Does NOT advance the clock. Returns a new frozen State reflecting
    the started tasks. Called before `advance_to_next_event` within a
    single decision epoch.

    For each (project_id, lab_id) in `decision`:
      • Sample the task's realization index using markov.sample_realization
        (uses PI for task 0, otherwise PM column indexed by state.z[i]).
      • Set lab_owner[j] = project_id.
      • Set lab_realization[j] = sampled index.
      • Set lab_time_in_use[j] = 0.
      • Increment next_task[i].

    Raises ValueError on any illegal decision (busy lab, wrong lab for
    the task, no work left).
    """
    # Empty decision legality: only valid if at least one lab is busy.
    if not decision and not any(state.lab_owner):
        raise ValueError(
            "Empty decision illegal when all labs are free — time cannot advance."
        )

    next_task_l = list(state.next_task)
    lab_owner_l = list(state.lab_owner)
    lab_realization_l = list(state.lab_realization)
    lab_time_in_use_l = list(state.lab_time_in_use)

    for project_id, lab_id in decision:
        i = project_id - 1
        j = lab_id - 1

        # Bounds and eligibility.
        if not (1 <= project_id <= len(state.next_task)):
            raise ValueError(f"project_id {project_id} out of range")
        if not (1 <= lab_id <= len(state.lab_owner)):
            raise ValueError(f"lab_id {lab_id} out of range")
        if lab_owner_l[j] != 0:
            raise ValueError(
                f"Lab {lab_id} is busy (owned by project {lab_owner_l[j]}); "
                f"cannot start project {project_id}."
            )
        if next_task_l[i] >= spec.projects[i].num_tasks():
            raise ValueError(
                f"Project {project_id} has no more tasks to start."
            )

        task_idx = next_task_l[i]
        project = spec.projects[i]
        task = project.tasks[task_idx]

        if task.lab != lab_id:
            raise ValueError(
                f"Task {task_idx} of project {project_id} requires lab "
                f"{task.lab}, not lab {lab_id}."
            )

        # Sample. prev_realization = state.z[i]; markov.sample_realization
        # ignores it when task_idx == 0 (uses PI instead).
        prev_r = state.z[i]
        r = sample_realization(rng, project, task_idx, prev_r)

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
# advance_to_next_event
# ---------------------------------------------------------------------------

def advance_to_next_event(
    state: State,
    spec: ProblemSpec,
) -> Tuple[State, Tuple[CompletedEvent, ...]]:
    """Advance the clock to the next task completion.

    Finds the busy lab(s) with minimum remaining time. Advances t by
    that Δt. For labs finishing exactly at that Δt:
      • Records the sampled realization into z[i].
      • Frees the lab (lab_owner=0, lab_realization=-1, lab_time_in_use=0).
    Other busy labs: just accumulate lab_time_in_use by Δt.

    Returns
    -------
    (new_state, completed_events) : Tuple[State, Tuple[CompletedEvent, ...]]
        completed_events is a tuple of (project_id, task_idx,
        realization_idx, completion_time) records — one per lab that
        just finished. reward.py will consume this.

    Raises ValueError if called with no lab busy (no event to advance to).
    """
    num_labs = len(state.lab_owner)

    # Compute remaining time on every busy lab.
    remaining_per_lab = {}  # j -> remaining int
    for j in range(num_labs):
        if state.lab_owner[j] == 0:
            continue
        project_id = state.lab_owner[j]
        i = project_id - 1
        task_idx = state.next_task[i] - 1  # in-flight task
        task = spec.projects[i].tasks[task_idx]
        r = state.lab_realization[j]
        duration = task.realizations[r][1]  # (result, duration, cost)
        remaining_per_lab[j] = duration - state.lab_time_in_use[j]

    if not remaining_per_lab:
        raise ValueError(
            "advance_to_next_event called with no lab busy — nothing to advance to."
        )

    delta_t = min(remaining_per_lab.values())
    new_t = state.t + delta_t

    lab_owner_l = list(state.lab_owner)
    lab_realization_l = list(state.lab_realization)
    lab_time_in_use_l = list(state.lab_time_in_use)
    z_l = list(state.z)

    completed_events: List[CompletedEvent] = []

    for j, remaining in remaining_per_lab.items():
        if remaining == delta_t:
            # This lab's task completes now.
            project_id = state.lab_owner[j]
            i = project_id - 1
            task_idx = state.next_task[i] - 1
            r = state.lab_realization[j]

            z_l[i] = r
            lab_owner_l[j] = 0
            lab_realization_l[j] = -1
            lab_time_in_use_l[j] = 0
            completed_events.append((project_id, task_idx, r, new_t))
        else:
            # Still running: accumulate elapsed time.
            lab_time_in_use_l[j] += delta_t

    new_state = State(
        next_task=state.next_task,
        z=tuple(z_l),
        lab_owner=tuple(lab_owner_l),
        lab_realization=tuple(lab_realization_l),
        lab_time_in_use=tuple(lab_time_in_use_l),
        t=new_t,
    )

    # Sort events for canonical form (helpful in reward.py and tests).
    completed_events.sort()
    return new_state, tuple(completed_events)