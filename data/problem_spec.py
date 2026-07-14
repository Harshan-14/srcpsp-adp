"""
Paper's Section 6 problem instance — populated with representative numbers.

Path B: shapes and correlations match the paper's qualitative structure
(5 projects, 17 tasks, 2 labs; α=0.235, β=0; success-persistence Markov
transitions). Exact Table 1 values are placeholders. Swapping in the paper's
literal numbers later touches only this file — no downstream code changes.

Convention (mirrors src/models.py):
    PM[n] has shape (next_task_realizations, current_task_realizations).
    Column j of PM[n] is the probability distribution over the next task's
    realizations, given the current task landed in realization j.
"""

import numpy as np

from src.models import Task, Project, ProblemSpec


# ---------------------------------------------------------------------------
# Reusable Markov templates
# ---------------------------------------------------------------------------

# Initial distribution over (fail, moderate, high) for the first task of a
# project. Mildly optimistic — modal outcome is "moderate".
_PI_DEFAULT = np.array([0.2, 0.5, 0.3])

# Transition template. Rows index the NEXT task's realization, columns index
# the CURRENT task's realization.
#   Column 0 (prev = fail)     → likely to fail again.
#   Column 1 (prev = moderate) → moderate persists.
#   Column 2 (prev = high)     → high persists.
# This "success-persistence" shape matches the qualitative pattern in the paper.
_PM_DEFAULT = np.array([
    [0.5, 0.2, 0.1],   # next = fail
    [0.4, 0.5, 0.3],   # next = moderate
    [0.1, 0.3, 0.6],   # next = high
])


# ---------------------------------------------------------------------------
# Per-project builders
# ---------------------------------------------------------------------------

def _build_project_1() -> Project:
    """P1: I1(L1) → I2(L2) → P1(L1). R0=5000, PD=34."""
    tasks = (
        Task("I1", lab=1, realizations=(
            ("fail",     2, 100.0),
            ("moderate", 3, 150.0),
            ("high",     4, 200.0),
        )),
        Task("I2", lab=2, realizations=(
            ("fail",     2, 120.0),
            ("moderate", 4, 180.0),
            ("high",     5, 220.0),
        )),
        Task("P1", lab=1, realizations=(
            ("fail",     3, 150.0),
            ("moderate", 5, 250.0),
            ("high",     6, 300.0),
        )),
    )
    return Project(
        project_id=1,
        tasks=tasks,
        R0=5000.0, alpha=0.05, PD=34, beta=0.0,
        PI=_PI_DEFAULT.copy(),
        PM=(_PM_DEFAULT.copy(), _PM_DEFAULT.copy()),
    )


def _build_project_2() -> Project:
    """P2: I3(L1) → I4(L1) → P2(L2). R0=7000, PD=23 (tightest deadline)."""
    tasks = (
        Task("I3", lab=1, realizations=(
            ("fail",     2,  90.0),
            ("moderate", 3, 160.0),
            ("high",     4, 210.0),
        )),
        Task("I4", lab=1, realizations=(
            ("fail",     2, 100.0),
            ("moderate", 3, 170.0),
            ("high",     4, 220.0),
        )),
        Task("P2", lab=2, realizations=(
            ("fail",     3, 180.0),
            ("moderate", 4, 260.0),
            ("high",     5, 320.0),
        )),
    )
    return Project(
        project_id=2,
        tasks=tasks,
        R0=7000.0, alpha=0.05, PD=23, beta=0.0,
        PI=_PI_DEFAULT.copy(),
        PM=(_PM_DEFAULT.copy(), _PM_DEFAULT.copy()),
    )


def _build_project_3() -> Project:
    """P3: I5(L2) → I6(L1) → I7(L2) → P3(L1). R0=10000, PD=28."""
    tasks = (
        Task("I5", lab=2, realizations=(
            ("fail",     2, 110.0),
            ("moderate", 3, 180.0),
            ("high",     4, 230.0),
        )),
        Task("I6", lab=1, realizations=(
            ("fail",     2, 120.0),
            ("moderate", 3, 190.0),
            ("high",     4, 240.0),
        )),
        Task("I7", lab=2, realizations=(
            ("fail",     2, 130.0),
            ("moderate", 4, 200.0),
            ("high",     5, 260.0),
        )),
        Task("P3", lab=1, realizations=(
            ("fail",     3, 200.0),
            ("moderate", 5, 280.0),
            ("high",     6, 340.0),
        )),
    )
    return Project(
        project_id=3,
        tasks=tasks,
        R0=10000.0, alpha=0.05, PD=28, beta=0.0,
        PI=_PI_DEFAULT.copy(),
        PM=(_PM_DEFAULT.copy(), _PM_DEFAULT.copy(), _PM_DEFAULT.copy()),
    )


def _build_project_4() -> Project:
    """P4: I8(L2) → I9(L1) → I10(L1) → P4(L2). R0=11000, PD=37."""
    tasks = (
        Task("I8", lab=2, realizations=(
            ("fail",     2, 120.0),
            ("moderate", 3, 190.0),
            ("high",     4, 240.0),
        )),
        Task("I9", lab=1, realizations=(
            ("fail",     2, 130.0),
            ("moderate", 3, 200.0),
            ("high",     4, 250.0),
        )),
        Task("I10", lab=1, realizations=(
            ("fail",     3, 140.0),
            ("moderate", 4, 210.0),
            ("high",     5, 270.0),
        )),
        Task("P4", lab=2, realizations=(
            ("fail",     3, 220.0),
            ("moderate", 5, 300.0),
            ("high",     6, 360.0),
        )),
    )
    return Project(
        project_id=4,
        tasks=tasks,
        R0=11000.0, alpha=0.05, PD=37, beta=0.0,
        PI=_PI_DEFAULT.copy(),
        PM=(_PM_DEFAULT.copy(), _PM_DEFAULT.copy(), _PM_DEFAULT.copy()),
    )


def _build_project_5() -> Project:
    """P5: I11(L1) → I12(L2) → P5(L2). R0=13000 (largest), PD=32.

    Note: paper's Table 1 has apparent duplication between P4 and P5.
    Per prior decision, treated as intentionally similar. In this Path B
    encoding P5 uses lightly distinct numbers for clarity.
    """
    tasks = (
        Task("I11", lab=1, realizations=(
            ("fail",     2, 130.0),
            ("moderate", 3, 200.0),
            ("high",     4, 260.0),
        )),
        Task("I12", lab=2, realizations=(
            ("fail",     2, 140.0),
            ("moderate", 4, 220.0),
            ("high",     5, 280.0),
        )),
        Task("P5", lab=2, realizations=(
            ("fail",     3, 240.0),
            ("moderate", 5, 320.0),
            ("high",     6, 380.0),
        )),
    )
    return Project(
        project_id=5,
        tasks=tasks,
        R0=13000.0, alpha=0.05, PD=32, beta=0.0,
        PI=_PI_DEFAULT.copy(),
        PM=(_PM_DEFAULT.copy(), _PM_DEFAULT.copy()),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(spec: ProblemSpec) -> None:
    """Sanity-check probability shapes and normalization.

    Called by build_spec() at load time so any typo surfaces immediately —
    not silently during a downstream simulation.
    """
    tol = 1e-9

    for proj in spec.projects:
        # PI must be 1-D, sum to 1, and match task 0's realization count.
        assert proj.PI.ndim == 1, (
            f"Project {proj.project_id}: PI must be 1-D"
        )
        assert abs(proj.PI.sum() - 1.0) < tol, (
            f"Project {proj.project_id}: PI sums to {proj.PI.sum()}, not 1"
        )
        assert len(proj.PI) == proj.tasks[0].num_realizations(), (
            f"Project {proj.project_id}: PI length {len(proj.PI)} "
            f"!= task 0 realizations {proj.tasks[0].num_realizations()}"
        )

        # One PM matrix between each consecutive task pair.
        assert len(proj.PM) == proj.num_tasks() - 1, (
            f"Project {proj.project_id}: PM count {len(proj.PM)} "
            f"!= tasks-1 ({proj.num_tasks() - 1})"
        )

        # Each PM has correct shape and columns sum to 1.
        for n, pm in enumerate(proj.PM):
            next_r = proj.tasks[n + 1].num_realizations()
            prev_r = proj.tasks[n].num_realizations()
            assert pm.shape == (next_r, prev_r), (
                f"Project {proj.project_id} PM[{n}]: shape {pm.shape} "
                f"!= expected ({next_r}, {prev_r})"
            )
            col_sums = pm.sum(axis=0)
            for j, cs in enumerate(col_sums):
                assert abs(cs - 1.0) < tol, (
                    f"Project {proj.project_id} PM[{n}] col {j}: "
                    f"sums to {cs}, not 1"
                )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_spec() -> ProblemSpec:
    """
    Build the full paper instance: 5 projects, 17 tasks, 2 labs.

    Called once at pipeline startup. Runs validation before returning.
    """
    projects = (
        _build_project_1(),
        _build_project_2(),
        _build_project_3(),
        _build_project_4(),
        _build_project_5(),
    )
    spec = ProblemSpec(projects=projects, num_labs=2)
    _validate(spec)
    return spec