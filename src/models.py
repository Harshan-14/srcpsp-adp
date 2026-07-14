"""
Data models for the sRCPSP problem instance.

Three objects:
    Task        - one atomic unit of work; knows its lab and possible realizations.
    Project     - an ordered chain of tasks + reward params + Markov chain data.
    ProblemSpec - the full problem: list of projects + number of labs.

All classes are frozen dataclasses so instances are immutable and hashable
where useful. The problem instance is static once loaded.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


# A single realization outcome for a task.
# (result, duration, cost) where result is a label like "success" / "fail" /
# "moderate" / "high", duration is an integer time-unit count, cost is a number.
Realization = Tuple[str, int, float]


@dataclass(frozen=True)
class Task:
    """
    One task in a project's chain.

    Fields
    ------
    task_id : str
        Human-readable id from the paper (e.g. "I1", "P3").
    lab : int
        Which lab this task must run on. 1-indexed to match the paper.
    realizations : tuple of (result, duration, cost)
        All possible outcomes for this task. Index into this tuple
        corresponds to the row/column index in the PI/PM matrices.
    """
    task_id: str
    lab: int
    realizations: Tuple[Realization, ...]

    def num_realizations(self) -> int:
        return len(self.realizations)


@dataclass(frozen=True)
class Project:
    """
    One project = one task chain + reward params + Markov chain data.

    Fields
    ------
    project_id : int
        1-indexed to match the paper.
    tasks : tuple of Task
        Ordered chain. tasks[0] runs first, tasks[-1] runs last.
    R0 : float
        Peak reward at t=0.
    alpha : float
        Reward decay rate. R(k) = R0 - exp(alpha * k).
    PD : int
        Project deadline. After PD, reward collapses to beta.
    beta : float
        Post-deadline floor reward.
    PI : np.ndarray
        Initial probability vector for task 0.
        Shape: (tasks[0].num_realizations,). Sums to 1.
    PM : tuple of np.ndarray
        Transition matrices between consecutive tasks.
        PM[n] is the transition matrix from task n to task n+1.
        Shape of PM[n]: (tasks[n+1].num_realizations, tasks[n].num_realizations).
        Column j of PM[n] = distribution over task n+1's realizations
        given that task n landed in realization j.
        Length of PM tuple: len(tasks) - 1.
    """
    project_id: int
    tasks: Tuple[Task, ...]
    R0: float
    alpha: float
    PD: int
    beta: float
    PI: np.ndarray
    PM: Tuple[np.ndarray, ...]

    def num_tasks(self) -> int:
        return len(self.tasks)


@dataclass(frozen=True)
class ProblemSpec:
    """
    The complete problem instance.

    Fields
    ------
    projects : tuple of Project
    num_labs : int
        Total resources available (labs are 1-indexed).
    """
    projects: Tuple[Project, ...]
    num_labs: int

    def num_projects(self) -> int:
        return len(self.projects)