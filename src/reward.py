"""
Cost and reward accounting for the sRCPSP simulation.

Consumes completion events from transitions.advance_to_next_event and turns
each event into a (cost, reward) pair. The simulator accumulates these
into J, the cost-to-go.

Cost-minimization framing (locked from paper decoding):
    J = accumulated_cost - accumulated_reward
    smaller J  ⇒  larger expected profit
    optimal policy minimizes E[J]

Per-event contract:
    accrue(event, state_after, spec) -> (cost, reward)
    caller updates:  J += cost - reward

Design decisions locked:

 1. Both cost and reward handled at task COMPLETION events. No accrual at
    task start. Keeps this file focused on one event type; keeps
    apply_decision free of reward concerns.
 2. Project reward is piecewise exponential with deadline:
        k <= PD :  R(k) = R0 * exp(-alpha * k)
        k >  PD :  R(k) = R0 * exp(-alpha*PD - beta*(k - PD))
    Path B data has beta=0, so post-deadline reward flattens at
    R0 * exp(-alpha*PD). Piecewise form kept for paper-faithfulness.
 3. Path B: NO failure gating on completion reward. Any project that
    finishes all its tasks earns R(k) at the last completion time,
    regardless of individual realization results. Consistent with
    transitions.py decision 8. Documented deviation from paper's
    original termination-on-failure semantics.
 4. "Project complete" = next_task[i] == num_tasks(i) AND project_id
    NOT in state_after.lab_owner. Both required because next_task
    increments at start, so num_tasks() alone doesn't confirm the
    last task has actually finished.
 5. Per-event interface. No batch wrapper — the simulator iterates
    events itself.
"""
import math
from typing import Tuple

from .models import ProblemSpec
from .state import State
from .transitions import CompletedEvent


# ---------------------------------------------------------------------------
# task_cost
# ---------------------------------------------------------------------------

def task_cost(event: CompletedEvent, spec: ProblemSpec) -> float:
    """Cost of the just-completed task, read from its realization triple.

    Realization triples are (result, duration, cost). Cost is the third
    element and is always charged, regardless of the result string.
    """
    project_id, task_idx, realization_idx, _ = event
    project = spec.projects[project_id - 1]
    task = project.tasks[task_idx]
    return float(task.realizations[realization_idx][2])


# ---------------------------------------------------------------------------
# project_reward
# ---------------------------------------------------------------------------

def project_reward(
    project_id: int,
    completion_time: int,
    spec: ProblemSpec,
) -> float:
    """R_i(k) — piecewise exponential reward for project i completing at time k.

        k <= PD :  R0 * exp(-alpha * k)
        k >  PD :  R0 * exp(-alpha * PD - beta * (k - PD))

    Path B data has beta=0, so the post-deadline branch is flat at
    R0 * exp(-alpha * PD). Piecewise logic kept in for general beta.
    """
    project = spec.projects[project_id - 1]
    k = completion_time
    if k <= project.PD:
        return project.R0 * math.exp(-project.alpha * k)
    else:
        return project.R0 * math.exp(
            -project.alpha * project.PD - project.beta * (k - project.PD)
        )


# ---------------------------------------------------------------------------
# is_project_complete (private)
# ---------------------------------------------------------------------------

def _is_project_complete(
    project_id: int,
    state_after: State,
    spec: ProblemSpec,
) -> bool:
    """True iff project_id has no more tasks to start AND is not currently
    running on any lab.

    Both conditions required because next_task[i] increments at task
    START — reaching num_tasks(i) means the last task has been started,
    not that it has finished. The lab_owner check confirms it's finished.
    """
    i = project_id - 1
    all_started = state_after.next_task[i] == spec.projects[i].num_tasks()
    not_running = project_id not in state_after.lab_owner
    return all_started and not_running


# ---------------------------------------------------------------------------
# accrue
# ---------------------------------------------------------------------------

def accrue(
    event: CompletedEvent,
    state_after: State,
    spec: ProblemSpec,
) -> Tuple[float, float]:
    """For a single completion event, return (cost, reward).

    cost   — task cost from the realization triple. Always applicable.
    reward — R_i(k) if this event finishes the project's last task,
             else 0.0.

    Caller accumulates:  J += cost - reward.

    Handles simultaneous completions correctly when called in a loop
    over the events returned by advance_to_next_event: state_after
    already reflects all freeings and z-updates from the epoch, so the
    per-project completion check is consistent across events at the
    same time.
    """
    cost = task_cost(event, spec)
    project_id, _, _, completion_time = event

    reward = 0.0
    if _is_project_complete(project_id, state_after, spec):
        reward = project_reward(project_id, completion_time, spec)

    return cost, reward