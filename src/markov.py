"""
Markov chain sampling for task realizations.

Given a project, a task index, and the previous task's realization, sample
the next task's realization index according to the appropriate probability
vector:

    task 0            ->  project.PI                            (initial vector)
    task k (k > 0)    ->  project.PM[k-1][:, prev_realization]  (column of PM)

The returned index is a plain int in [0, num_realizations_of(task_idx)). To
get the corresponding (result, duration, cost) tuple, the caller does:

    project.tasks[task_idx].realizations[idx]

Randomness is externalized: the caller passes a numpy Generator. Same seed
produces the same sequence. Required for reproducible simulation runs.
"""

import numpy as np

from src.models import Project


def sample_realization(
    rng: np.random.Generator,
    project: Project,
    task_idx: int,
    prev_realization: int,
) -> int:
    """
    Sample the realization index for a task about to start.

    Parameters
    ----------
    rng
        numpy Generator. Advances by one draw per call.
    project
        The project whose task is starting.
    task_idx
        Index (0-based) of the task about to start.
    prev_realization
        Realization index (0-based) of the immediately previous task in
        this project. Ignored when task_idx == 0. Pass -1 or any value
        when sampling the first task.

    Returns
    -------
    int
        Index (0-based) into project.tasks[task_idx].realizations.
    """
    # Task index bounds. Cheap. Catches integration bugs early.
    assert 0 <= task_idx < project.num_tasks(), (
        f"task_idx {task_idx} out of range "
        f"[0, {project.num_tasks()}) for project {project.project_id}"
    )

    # Select the probability vector.
    if task_idx == 0:
        probs = project.PI
    else:
        prev_num_r = project.tasks[task_idx - 1].num_realizations()
        assert 0 <= prev_realization < prev_num_r, (
            f"prev_realization {prev_realization} out of range "
            f"[0, {prev_num_r}) for project {project.project_id}, "
            f"previous task {task_idx - 1}"
        )
        # Column `prev_realization` = P(next realization | previous = prev_realization).
        probs = project.PM[task_idx - 1][:, prev_realization]

    num_r = project.tasks[task_idx].num_realizations()
    # np.random.Generator.choice returns numpy.int64; cast for clean hashing.
    return int(rng.choice(num_r, p=probs))