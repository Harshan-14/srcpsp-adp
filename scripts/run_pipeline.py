"""
End-to-end pipeline: train the ADP policy, validate it against the H1
baseline on held-out trajectories.

Usage (from repo root):
    python -m scripts.run_pipeline
    python -m scripts.run_pipeline --train-trajs 2000 --val-trajs 500
    python -m scripts.run_pipeline --tolerance 0.005

Assumes data/problem_spec.py exposes a `spec` (ProblemSpec) at module
level. If your file uses a factory (e.g. `def get_spec()`), replace
the import at the top accordingly.
"""
import argparse
import time
from typing import Callable, List, Tuple

import numpy as np

from src.bellman import bellman_iterate
from src.heuristic import heuristic_1
from src.policy import make_adp_policy
from src.simulator import build_subset_and_J0, run_trajectory
from src.state import initial_state
from src.models import ProblemSpec
from src.state import State
from src.transitions import Decision


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

from data.problem_spec import build_spec

spec = build_spec()


PolicyFn = Callable[[State, ProblemSpec], Decision]


# ---------------------------------------------------------------------------
# Validation helper — mean total J over N held-out trajectories
# ---------------------------------------------------------------------------

def mean_total_J(
    policy: PolicyFn,
    spec: ProblemSpec,
    num_trajectories: int,
    seed: int,
) -> Tuple[float, List[float]]:
    """Run N trajectories under `policy` from a fresh RNG. Returns
    (mean total J, list of per-trajectory J)."""
    rng = np.random.default_rng(seed)
    per_traj_J: List[float] = []
    for _ in range(num_trajectories):
        traj = run_trajectory(spec, policy, rng)
        # tail_J at the initial state = the total J of the trajectory.
        per_traj_J.append(traj[0][1])
    return sum(per_traj_J) / len(per_traj_J), per_traj_J


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-trajs", type=int, default=1000,
                        help="Trajectories to build the confined subset and Ĵ⁰")
    parser.add_argument("--val-trajs", type=int, default=500,
                        help="Held-out trajectories per policy for validation")
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=1337)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Relative convergence threshold for Bellman")
    parser.add_argument("--max-iters", type=int, default=100)
    args = parser.parse_args()

    s0 = initial_state(spec)
    hr = "=" * 60

    # -----------------------------------------------------------------
    # Phase 1: Heuristic simulation → subset + Ĵ⁰
    # -----------------------------------------------------------------
    print(hr)
    print("PHASE 1 — Heuristic simulation (H1)")
    print(hr)
    print(f"  Trajectories : {args.train_trajs}")
    print(f"  Seed         : {args.train_seed}")

    t0 = time.time()
    rng_train = np.random.default_rng(args.train_seed)
    subset, J0 = build_subset_and_J0(
        spec, heuristic_1, args.train_trajs, rng_train
    )
    t_sim = time.time() - t0

    print(f"  |subset|     : {len(subset):,}")
    print(f"  Ĵ⁰(s0)       : {J0[s0]:.2f}")
    print(f"  Time         : {t_sim:.2f}s")

    # -----------------------------------------------------------------
    # Phase 2: Bellman iteration → Ĵ*
    # -----------------------------------------------------------------
    print()
    print(hr)
    print("PHASE 2 — Bellman iteration")
    print(hr)
    print(f"  Tolerance    : {args.tolerance}")
    print(f"  Max iters    : {args.max_iters}")

    t0 = time.time()
    J_star, iters = bellman_iterate(
        subset, J0, spec,
        tolerance=args.tolerance,
        max_iters=args.max_iters,
    )
    t_bellman = time.time() - t0

    print(f"  Iterations   : {iters}")
    print(f"  Ĵ*(s0)       : {J_star[s0]:.2f}")
    print(f"  ΔĴ vs Ĵ⁰     : {J_star[s0] - J0[s0]:+.2f}")
    print(f"  Time         : {t_bellman:.2f}s")

    # -----------------------------------------------------------------
    # Phase 3: Held-out validation — H1 vs ADP
    # -----------------------------------------------------------------
    print()
    print(hr)
    print("PHASE 3 — Held-out validation")
    print(hr)
    print(f"  Trajectories : {args.val_trajs} per policy")
    print(f"  Seed         : {args.val_seed} (common random numbers)")
    print()

    t0 = time.time()
    mean_h1, _ = mean_total_J(
        heuristic_1, spec, args.val_trajs, args.val_seed
    )
    t_h1 = time.time() - t0
    print(f"  H1  mean total J : {mean_h1:.2f}   (time: {t_h1:.2f}s)")

    adp_policy_fn = make_adp_policy(J_star, subset)
    t0 = time.time()
    mean_adp, _ = mean_total_J(
        adp_policy_fn, spec, args.val_trajs, args.val_seed
    )
    t_adp = time.time() - t0
    print(f"  ADP mean total J : {mean_adp:.2f}   (time: {t_adp:.2f}s)")

    # Improvement: with cost-min framing, more negative = better.
    # (H1 - ADP) / |H1| gives % improvement (positive = ADP wins).
    improvement_pct = (mean_h1 - mean_adp) / abs(mean_h1) * 100.0 if mean_h1 != 0 else 0.0

    print()
    print(hr)
    print("SUMMARY")
    print(hr)
    print(f"  |subset|                  : {len(subset):,}")
    print(f"  Bellman iterations        : {iters}")
    print(f"  Ĵ*(s0)                    : {J_star[s0]:.2f}")
    print(f"  H1  mean total J          : {mean_h1:.2f}")
    print(f"  ADP mean total J          : {mean_adp:.2f}")
    print(f"  ADP improvement over H1   : {improvement_pct:+.2f}%")
    print(f"  Total wall time           : {t_sim + t_bellman + t_h1 + t_adp:.1f}s")


if __name__ == "__main__":
    main()