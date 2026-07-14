"""
Streamlit walkthrough for the sRCPSP ADP project.

Run from the repo root:
    streamlit run app.py

Three parts:
  A. What the problem looks like (schema, Markov, state).
  B. The ADP pipeline in three phases (simulation, Bellman, validation).
  C. One trajectory visualized as a Gantt chart.
"""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st

from data.problem_spec import build_spec
from src.bellman import bellman_iterate
from src.heuristic import heuristic_1
from src.policy import make_adp_policy
from src.simulator import build_subset_and_J0, run_trajectory
from src.state import initial_state, is_terminal
from src.transitions import (
    advance_to_next_event,
    apply_decision,
    enumerate_decisions,
)


st.set_page_config(page_title="sRCPSP ADP Walkthrough", layout="wide")


# ---------------------------------------------------------------------------
# Spec (cached so it isn't rebuilt on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_spec():
    return build_spec()


spec = load_spec()


# ---------------------------------------------------------------------------
# Helpers used by the Gantt section
# ---------------------------------------------------------------------------

def run_trajectory_with_events(spec_, policy, rng):
    """Run one trajectory under `policy`; return list of task executions
    with (start_time, end_time, project_id, task, lab, outcome) info."""
    s = initial_state(spec_)
    executions = []
    active = {}  # lab_id -> (project_id, task_idx, start_time)

    while not is_terminal(s, spec_):
        legal = enumerate_decisions(s, spec_)
        if not legal:
            break

        d = policy(s, spec_)
        start_time = s.t
        s = apply_decision(s, d, spec_, rng)

        for project_id, lab_id in d:
            i = project_id - 1
            task_idx = s.next_task[i] - 1
            active[lab_id] = (project_id, task_idx, start_time)

        if any(s.lab_owner):
            s, events = advance_to_next_event(s, spec_)
            for e in events:
                project_id, task_idx, realization_idx, completion_time = e
                for lab_id, (pid, tid, st_time) in list(active.items()):
                    if pid == project_id and tid == task_idx:
                        task = spec_.projects[pid - 1].tasks[tid]
                        executions.append({
                            "project_id": pid,
                            "task_idx": tid,
                            "task_name": task.task_id,
                            "lab_id": lab_id,
                            "start": st_time,
                            "end": completion_time,
                            "outcome": task.realizations[realization_idx][0],
                        })
                        del active[lab_id]
                        break

    return executions


def render_gantt(executions):
    """Draw a Gantt chart. Bars colored by task outcome."""
    if not executions:
        st.warning("No executions to display.")
        return

    outcome_color = {
        "fail":     "#c0392b",
        "moderate": "#f39c12",
        "high":     "#27ae60",
    }

    fig, ax = plt.subplots(figsize=(10, 3.2))
    for ex in executions:
        color = outcome_color[ex["outcome"]]
        width = ex["end"] - ex["start"]
        ax.barh(ex["lab_id"], width, left=ex["start"], height=0.55,
                color=color, edgecolor="black", linewidth=0.5)
        ax.text(ex["start"] + width / 2, ex["lab_id"],
                f"P{ex['project_id']}\u00b7{ex['task_name']}",
                ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Lab 1", "Lab 2"])
    ax.set_xlabel("Time")
    ax.set_xlim(0, max(ex["end"] for ex in executions) + 1)
    ax.set_ylim(0.4, 2.6)
    ax.invert_yaxis()

    legend = [Patch(facecolor=c, label=lbl) for lbl, c in outcome_color.items()]
    ax.legend(handles=legend, loc="upper right", title="Outcome", frameon=True)
    ax.grid(axis="x", alpha=0.3)

    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Stochastic RCPSP \u2014 ADP Walkthrough")
st.markdown(
    "This dashboard walks through an Approximate Dynamic Programming (ADP) "
    "solution to the Stochastic Resource-Constrained Project Scheduling "
    "Problem. Read the sections in order: problem definition first, then "
    "the three-phase pipeline, then a single trajectory visualization."
)
st.divider()


# ---------------------------------------------------------------------------
# Sidebar parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Pipeline parameters")
    train_trajs = st.number_input(
        "Training trajectories", min_value=100, max_value=10000,
        value=3000, step=100,
        help="Number of H1-driven trajectories used to build the confined subset.",
    )
    val_trajs = st.number_input(
        "Held-out trajectories", min_value=100, max_value=5000,
        value=1000, step=100,
        help="Number of fresh trajectories run under each policy for validation.",
    )
    train_seed = st.number_input("Training seed", 0, 99999, 42)
    val_seed = st.number_input("Validation seed", 0, 99999, 1337)
    tolerance = st.number_input(
        "Bellman tolerance", 0.001, 0.1, 0.01, 0.001, format="%.3f",
        help="Relative convergence threshold: max |\u0394\u0134| / max |\u0134|.",
    )

    st.divider()
    st.caption(
        "Default 3000/1000 finishes in ~30s. Change any parameter and "
        "click **Run pipeline** below to re-execute Phase 1\u20133."
    )


# ---------------------------------------------------------------------------
# PART A \u2014 Problem definition
# ---------------------------------------------------------------------------

st.header("Part A \u2014 What the problem looks like")

st.subheader("1. Projects and tasks")
st.markdown(
    "Five research projects compete for two shared labs. Each project has a "
    "fixed sequence of tasks. Each task must run on a specific lab and has "
    "three possible outcomes \u2014 *fail*, *moderate*, or *high* \u2014 each "
    "with its own duration and cost. Completing a project earns a "
    "time-decaying reward R(k) = R\u2080 \u00b7 exp(\u2212\u03b1 \u00b7 k)."
)

project_df = pd.DataFrame([
    {
        "Project": f"P{p.project_id}",
        "Tasks": p.num_tasks(),
        "Reward R\u2080": f"{p.R0:,.0f}",
        "Deadline PD": p.PD,
        "\u03b1 (decay)": p.alpha,
    }
    for p in spec.projects
])
st.dataframe(project_df, use_container_width=True, hide_index=True)

with st.expander("Inspect one project's task details"):
    pid = st.selectbox("Project", [1, 2, 3, 4, 5], key="proj_select")
    proj = spec.projects[pid - 1]
    rows = []
    for t in proj.tasks:
        for r in t.realizations:
            rows.append({
                "Task": t.task_id,
                "Lab": t.lab,
                "Outcome": r[0],
                "Duration": r[1],
                "Cost": f"{r[2]:.0f}",
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("2. Markov uncertainty over task outcomes")
st.markdown(
    "Task outcomes are stochastic and correlated. The **first** task of a "
    "project draws from an initial distribution PI. **Subsequent** tasks "
    "condition on the previous task's outcome via a transition matrix PM. "
    "The calibration has *success persistence*: a task that landed on "
    "*high* makes the next task more likely to also land on *high*."
)

p1 = spec.projects[0]
col1, col2 = st.columns(2)

with col1:
    st.markdown("**PI \u2014 initial distribution (P1's first task)**")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(["fail", "moderate", "high"], p1.PI, color="steelblue")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    for i, v in enumerate(p1.PI):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("**PM \u2014 transition matrix (P1's task 1 \u2192 task 2)**")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.imshow(p1.PM[0], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["fail", "moderate", "high"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["fail", "moderate", "high"])
    ax.set_xlabel("Previous outcome")
    ax.set_ylabel("Next outcome")
    for i in range(3):
        for j in range(3):
            v = p1.PM[0][i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.4 else "black", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)

st.caption(
    "Columns of PM sum to 1. Reading a column: given the previous task "
    "landed on that outcome, the probability distribution over the next "
    "task's outcome."
)

st.subheader("3. State encoding")
st.markdown(
    "A decision-epoch state is a frozen tuple of six fields \u2014 everything "
    "the policy needs to make a scheduling choice, and everything Bellman "
    "iterates over. Being a frozen tuple, it is hashable, so states become "
    "dictionary keys and set members without extra machinery."
)

s0 = initial_state(spec)
st.code(
    f"""State(
    next_task       = {s0.next_task},   # next unstarted task index, per project
    z               = {s0.z},                  # last realization, per project  (-1 = none yet)
    lab_owner       = {s0.lab_owner},                       # project on each lab              (0 = free)
    lab_realization = {s0.lab_realization},                       # realization sampled at task start
    lab_time_in_use = {s0.lab_time_in_use},                       # elapsed time on each lab
    t               = {s0.t},                             # current clock
)""",
    language="python",
)


# ---------------------------------------------------------------------------
# PART B \u2014 Pipeline
# ---------------------------------------------------------------------------

st.divider()
st.header("Part B \u2014 Run the ADP pipeline")
st.markdown(
    "Three phases: (1) simulate the H1 heuristic policy to build a confined "
    "subset of the state space and a Monte Carlo estimate \u0134\u2070; "
    "(2) iterate the Bellman operator on that subset to converge to \u0134*; "
    "(3) run fresh held-out trajectories under both H1 and the ADP policy "
    "and compare."
)

run_col1, run_col2 = st.columns([1, 3])
with run_col1:
    run_button = st.button("Run pipeline", type="primary", use_container_width=True)
with run_col2:
    st.caption(
        "With the default 3000 training / 1000 held-out trajectories, this "
        "takes roughly 30 seconds on a laptop."
    )

if run_button:
    with st.spinner(f"Phase 1 \u2014 running {train_trajs} H1 trajectories..."):
        t0 = time.time()
        rng = np.random.default_rng(train_seed)
        subset, J0 = build_subset_and_J0(spec, heuristic_1, train_trajs, rng)
        t_phase1 = time.time() - t0

    with st.spinner("Phase 2 \u2014 Bellman iteration..."):
        t0 = time.time()
        J_star, iters = bellman_iterate(subset, J0, spec,
                                        tolerance=tolerance, max_iters=100)
        t_phase2 = time.time() - t0

    with st.spinner(f"Phase 3 \u2014 {val_trajs} held-out trajectories per policy..."):
        t0 = time.time()
        rng_h1 = np.random.default_rng(val_seed)
        js_h1 = np.array([
            run_trajectory(spec, heuristic_1, rng_h1)[0][1]
            for _ in range(val_trajs)
        ])

        adp_fn = make_adp_policy(J_star, subset)
        rng_adp = np.random.default_rng(val_seed)
        js_adp = np.array([
            run_trajectory(spec, adp_fn, rng_adp)[0][1]
            for _ in range(val_trajs)
        ])
        t_phase3 = time.time() - t0

    st.session_state.update({
        "subset": subset,
        "J0": J0,
        "J_star": J_star,
        "iters": iters,
        "js_h1": js_h1,
        "js_adp": js_adp,
        "t_phase1": t_phase1,
        "t_phase2": t_phase2,
        "t_phase3": t_phase3,
    })


# ---------------------------------------------------------------------------
# Results (render only if pipeline has been run at least once)
# ---------------------------------------------------------------------------

if "J_star" in st.session_state:
    subset = st.session_state["subset"]
    J0 = st.session_state["J0"]
    J_star = st.session_state["J_star"]
    iters = st.session_state["iters"]
    js_h1 = st.session_state["js_h1"]
    js_adp = st.session_state["js_adp"]
    s0 = initial_state(spec)

    st.subheader("4. Phase 1 \u2014 Heuristic simulation \u2192 confined subset + \u0134\u2070")
    st.markdown(
        "H1 (highest-success-probability heuristic) drives each trajectory. "
        "Every decision-epoch state visited across the trajectories is added "
        "to the *confined subset* \u2014 the domain Bellman will iterate on. "
        "For each state, the mean of the observed cost-to-go tails becomes "
        "\u0134\u2070, the Monte Carlo starting estimate."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Subset size", f"{len(subset):,}",
              help="Unique decision-epoch states visited across all training trajectories.")
    c2.metric("\u0134\u2070(s\u2080)", f"{J0[s0]:,.0f}",
              help="Monte Carlo estimate of expected cost-to-go from the initial state "
                   "under H1. More negative = more profitable.")
    c3.metric("Time", f"{st.session_state['t_phase1']:.1f}s")

    st.subheader("5. Phase 2 \u2014 Bellman iteration \u2192 \u0134*")
    st.markdown(
        "For each state in the subset, iterate: "
        "`\u0134_new(s) = min over legal decisions d of E[j_delta + \u0134(s')]`. "
        "The expectation is exact \u2014 enumerated over the joint realization "
        "outcomes of tasks started by the decision. Missing next-states are "
        "handled by a renormalization + barrier hybrid: known outcomes are "
        "renormalized, decisions with no known outcomes are barred as +\u221e."
    )
    delta_J = J_star[s0] - J0[s0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Iterations to converge", iters)
    c2.metric("\u0134*(s\u2080)", f"{J_star[s0]:,.0f}",
              delta=f"{delta_J:+,.0f} vs \u0134\u2070",
              delta_color="inverse",
              help="Bellman-optimized cost-to-go. Negative delta = more negative J = "
                   "Bellman improved on the Monte Carlo estimate.")
    c3.metric("Time", f"{st.session_state['t_phase2']:.1f}s")

    st.subheader("6. Phase 3 \u2014 Held-out validation \u2192 H1 vs ADP")
    st.markdown(
        "New RNG seed, unseen trajectories. Both policies see the same seed "
        "(*common random numbers*) so the paired comparison has lower "
        "variance. Report the mean total J under each and overlay the full "
        "distributions."
    )
    mean_h1 = float(js_h1.mean())
    mean_adp = float(js_adp.mean())
    improvement_pct = (mean_h1 - mean_adp) / abs(mean_h1) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("H1 mean total J", f"{mean_h1:,.0f}",
              help="Baseline heuristic.")
    c2.metric("ADP mean total J", f"{mean_adp:,.0f}",
              delta=f"{improvement_pct:+.2f}% vs H1",
              help="Positive delta = ADP is more profitable than H1.")
    c3.metric("Time", f"{st.session_state['t_phase3']:.1f}s")

    fig, ax = plt.subplots(figsize=(9, 4))
    lo = min(js_h1.min(), js_adp.min())
    hi = max(js_h1.max(), js_adp.max())
    bins = np.linspace(lo, hi, 40)
    ax.hist(js_h1, bins=bins, alpha=0.55, label="H1",
            color="steelblue", edgecolor="none")
    ax.hist(js_adp, bins=bins, alpha=0.55, label="ADP",
            color="darkorange", edgecolor="none")
    ax.axvline(mean_h1, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"H1 mean = {mean_h1:,.0f}")
    ax.axvline(mean_adp, color="darkorange", linestyle="--", linewidth=1.5,
               label=f"ADP mean = {mean_adp:,.0f}")
    ax.set_xlabel("Total J per trajectory   (more negative = more profitable)")
    ax.set_ylabel("Number of trajectories")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The gap between the two dashed mean lines is the ADP improvement. "
        "Distribution overlap is expected \u2014 H1 is already a reasonable "
        "baseline."
    )

    st.divider()
    st.header("Part C \u2014 See it happen")
    st.subheader("7. One trajectory under the ADP policy")
    st.markdown(
        "Pick a trajectory seed to see one realization play out. Each bar is "
        "one task; its length is that task's duration; its color is the "
        "sampled outcome. This is what the ADP policy actually decided to do "
        "on this particular trajectory."
    )

    gantt_seed = st.number_input(
        "Trajectory seed (change to sample a different trajectory)",
        min_value=0, max_value=99999, value=1, key="gantt_seed",
    )
    executions = run_trajectory_with_events(
        spec,
        make_adp_policy(J_star, subset),
        np.random.default_rng(gantt_seed),
    )
    render_gantt(executions)

    with st.expander("Show execution log"):
        exec_df = pd.DataFrame(executions)[
            ["start", "end", "project_id", "task_name", "lab_id", "outcome"]
        ].rename(columns={
            "start": "Start", "end": "End",
            "project_id": "Project", "task_name": "Task",
            "lab_id": "Lab", "outcome": "Outcome",
        })
        st.dataframe(exec_df, use_container_width=True, hide_index=True)

else:
    st.info(
        "Click **Run pipeline** above to execute Phase 1\u20133 and reveal "
        "the results and trajectory visualization."
    )