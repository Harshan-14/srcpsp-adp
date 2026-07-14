"""
State representation for the sRCPSP simulation.

A State is an immutable snapshot at a decision point — a moment when the
system is ready to potentially schedule new tasks. Two situations produce
a decision point:
    (a) t = 0 (initial state).
    (b) A task just completed on some lab (event-based transition).

Between events nothing observable happens: labs count time, that's all.
So the State only ever needs to represent the system AT event boundaries.

The State is a frozen dataclass with all-tuple fields, which makes it
hashable — required for use as a dict key in the confined state subset
during Bellman iteration.

Field semantics
---------------
next_task[i]        Index of project i's next unstarted task (0-indexed).
                    Increments the moment a task is started, not when it
                    completes. Equals num_tasks(i) when every task has
                    been started (project may still be running its last).

z[i]                Realization index of project i's most recently
                    COMPLETED task. -1 if no task of project i has
                    completed yet. Used by the Markov chain lookup.

lab_owner[j]        Project id (1-indexed, matches the paper) currently
                    running on lab j (0-indexed). 0 = lab is free.

lab_realization[j]  Realization index sampled for the in-progress task
                    on lab j. -1 if free. Sampled at task start so
                    duration is known; applied to z at task completion.

lab_time_in_use[j]  Time units lab j has been running its current task.
                    0 if free (or if task was just started this instant).
                    Corresponds to the paper's L_j.

t                   Current simulation time.

Sentinels
---------
lab_owner uses 0 for "free" (works because projects are 1-indexed).
lab_realization and z use -1 for "none." Keeping everything int makes
hashing cheap and avoids Optional overhead.
"""

from dataclasses import dataclass
from typing import Tuple

from src.models import ProblemSpec


# A Decision is a binary vector (delta_1, ..., delta_M).
# delta_i = 1 means "start project i's next task now."
Decision = Tuple[int, ...]


@dataclass(frozen=True)
class State:
    """
    Immutable snapshot of the system at a decision point.
    See module docstring for full field semantics.
    """
    next_task: Tuple[int, ...]
    z: Tuple[int, ...]
    lab_owner: Tuple[int, ...]
    lab_realization: Tuple[int, ...]
    lab_time_in_use: Tuple[int, ...]
    t: int


def initial_state(spec: ProblemSpec) -> State:
    """
    The state at t = 0 — no task started anywhere, every lab idle.
    """
    M = spec.num_projects()
    N = spec.num_labs
    return State(
        next_task=(0,) * M,
        z=(-1,) * M,
        lab_owner=(0,) * N,
        lab_realization=(-1,) * N,
        lab_time_in_use=(0,) * N,
        t=0,
    )


def project_running(state: State, spec: ProblemSpec, project_idx: int) -> bool:
    """
    True iff project project_idx (0-indexed) has a task currently
    in progress on some lab.
    """
    project_id = spec.projects[project_idx].project_id
    return project_id in state.lab_owner


def project_done(state: State, spec: ProblemSpec, project_idx: int) -> bool:
    """
    True iff project project_idx (0-indexed) has finished all its tasks.

    A project is done when every task has been started AND no lab is
    still running one of its tasks. Both conditions are necessary — a
    project that has started its last task but not yet completed it is
    NOT done.
    """
    proj = spec.projects[project_idx]
    if state.next_task[project_idx] < proj.num_tasks():
        return False
    return proj.project_id not in state.lab_owner


def lab_free(state: State, lab_idx: int) -> bool:
    """
    True iff lab lab_idx (0-indexed) is currently free.
    """
    return state.lab_owner[lab_idx] == 0


def is_terminal(state: State, spec: ProblemSpec) -> bool:
    """
    True iff every project has finished all tasks and every lab is free.

    Cheaper than looping project_done: check labs first (fast fail), then
    check every next_task counter reached num_tasks.
    """
    for owner in state.lab_owner:
        if owner != 0:
            return False
    for i, proj in enumerate(spec.projects):
        if state.next_task[i] < proj.num_tasks():
            return False
    return True