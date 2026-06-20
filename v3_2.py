"""
v3_2: Synthetic STRESS TEST + GRID ABLATION for the tolerance-order certificate.

Built on verify_tolerance_order_v3.py (difference estimator + shared base mask +
rho=0 total-energy constraint).  Two studies:

  STRESS TEST (stress_test)
    Scenarios:
      - pure-1     : K*=1; MUST NOT false-certify K=1 insufficient.
      - deg-2      : K*=2 with a clean pairwise block.
      - deg-3-tail : K*=2 (or 3) with a thin high-degree tail above target.
      - rare-int   : a single high-order interaction carrying small energy.
    Sweeps: alpha in {1%,5%,10%}, sigma_obs, d, and total query budget Q.
    Reports, per cell:
      feas         true-spectrum feasibility rate  (>= 1-delta)
      Kbar=K*      exact order-recovery rate
      K1-insuf     K=1 certified-insufficient rate (resolvable direction)
      FALSE-insuf  K=1 certified insuff but truly E(1)<=alpha  (MUST be ~0)
      FALSE-suf    any K certified suff but truly E(K)>alpha    (MUST be ~0)

  GRID ABLATION (grid_ablation)
    Fixes a TOTAL query budget Q and, for each candidate grid, sets
        N0 = Q // (L + 2)
    so every grid spends the same Q.  Compares L=4,6,9, always keeping rho=0,
    plus variants that push points near rho=1.  Picks the grid maximizing the
    K=1-insufficient rate at fixed Q while keeping FALSE rates ~0.

Run it yourself:   python v3_2.py
(uses torch / GPU if present; LP layer is scipy on CPU)
"""

import numpy as np
import torch
from verify_tolerance_order_v3 import (
    WalshFunction, run_pilot_batched, certify_order, _build_polytope,
    certified_Kbar, Kstar, true_residual_curve, pilot_queries,
    SUFFICIENT, INSUFFICIENT, UNRESOLVED, DEVICE,
)

# ----------------------------------------------------------------------
# Shared evaluation: run pilot for one (g, grid, N0) and tally verdicts
# ----------------------------------------------------------------------
def evaluate(g, rhos, N0, sigma_obs, delta, alpha, n_trials, Kmax, seed):
    """Returns a dict of rates over n_trials.  E_true computed to a depth that
    covers the true support so false-rate checks are exact."""
    rhos = np.asarray(rhos, dtype=np.float64)
    depth = min(len(g.a_true) - 1, Kmax)
    E_true, T_true = true_residual_curve(g.a_true, max(depth, 1))
    Kst = Kstar(g.a_true, alpha)

    s_hat, eta_s = run_pilot_batched(g, rhos, N0, sigma_obs, n_trials,
                                     seed=seed, delta=delta)

    # true spectrum over a_1..a_Kmax for feasibility test
    a_star = np.zeros(Kmax)
    upto = min(len(g.a_true) - 1, Kmax)
    a_star[:upto] = g.a_true[1:1 + upto]

    feas = 0
    kbar_eq = 0
    k1_insuf = 0
    false_insuf = 0      # K=1 insuff but E(1) <= alpha  (must be 0)
    false_suf = 0        # any K suff but E(K) > alpha    (must be 0)
    resolved = 0

    def E_at(K):
        return E_true[K] if K < len(E_true) else 0.0

    for t in range(n_trials):
        A_ub, b_ub, _, _ = _build_polytope(s_hat[t], rhos, Kmax, eta_s[t])
        feas += int(np.all(A_ub @ a_star <= b_ub + 1e-9))

        verdict, _, _ = certify_order(s_hat[t], rhos, Kmax, eta_s[t], alpha)
        Kbar, _ = certified_Kbar(verdict, Kmax)
        if Kbar is not None:
            resolved += 1
            kbar_eq += int(Kbar == Kst)

        v1 = verdict.get(1)
        if v1 == INSUFFICIENT:
            k1_insuf += 1
            if E_at(1) <= alpha:
                false_insuf += 1
        # any false-sufficient across all K
        for K, vv in verdict.items():
            if vv == SUFFICIENT and E_at(K) > alpha:
                false_suf += 1
                break

    n = n_trials
    return dict(
        feas=feas / n, kbar_eq=kbar_eq / n, k1_insuf=k1_insuf / n,
        false_insuf=false_insuf / n, false_suf=false_suf / n,
        Kst=Kst, E1=E_at(1), Q=pilot_queries(len(rhos), N0), N0=N0, L=len(rhos),
    )


# ----------------------------------------------------------------------
# Scenario builders
# ----------------------------------------------------------------------
def make_scenario(name, d, seed=0):
    """Returns (WalshFunction, label). Mean fixed at 0.5 (a_0 carries no info)."""
    if name == "pure-1":
        energies = {1: 1.0}
    elif name == "deg-2":
        energies = {1: 1.0, 2: 0.35}
    elif name == "deg-3-tail":
        energies = {1: 1.0, 2: 0.35, 3: 0.04}
    elif name == "rare-int":
        # most energy first order, a single thin high-order interaction at deg 5
        energies = {1: 1.0, 2: 0.15, 5: 0.05}
    else:
        raise ValueError(name)
    return WalshFunction(d, energies, mean=0.5, seed=seed), name


# ----------------------------------------------------------------------
# STRESS TEST
# ----------------------------------------------------------------------
def stress_test(n_trials=120, delta=0.1, seed=7):
    print("\n" + "=" * 78)
    print("STRESS TEST  (honest Kmax=d; false rates MUST stay ~0)")
    print("=" * 78)

    header = (f"{'scenario':>12} {'d':>4} {'a':>5} {'sig':>5} {'N0':>7} {'Q':>9} "
              f"{'feas':>6} {'Kbar=K*':>8} {'K1-insuf':>9} {'FALSE-ins':>10} "
              f"{'FALSE-suf':>10} {'K*':>3} {'E1':>6}")

    # --- (1) scenario sweep at a fixed reference budget ---
    print("\n[1] Scenario sweep  (d=49, alpha=5%, sigma=0.1, N0=16k)")
    print(header)
    for name in ["pure-1", "deg-2", "deg-3-tail", "rare-int"]:
        g, _ = make_scenario(name, d=49, seed=seed)
        r = evaluate(g, RHOS_DEFAULT, 16000, 0.1, delta, 0.05, n_trials,
                     Kmax=49, seed=3100)
        _row(name, 49, 0.05, 0.1, r)

    # --- (2) alpha sweep on deg-3-tail ---
    print("\n[2] alpha sweep  (deg-3-tail, d=49, sigma=0.1, N0=16k)")
    print(header)
    for alpha in [0.01, 0.05, 0.10]:
        g, _ = make_scenario("deg-3-tail", d=49, seed=seed)
        r = evaluate(g, RHOS_DEFAULT, 16000, 0.1, delta, alpha, n_trials,
                     Kmax=49, seed=3200)
        _row("deg-3-tail", 49, alpha, 0.1, r)

    # --- (3) noise sweep on deg-3-tail ---
    print("\n[3] sigma sweep  (deg-3-tail, d=49, alpha=5%, N0=16k)")
    print(header)
    for sig in [0.05, 0.1, 0.2, 0.4]:
        g, _ = make_scenario("deg-3-tail", d=49, seed=seed)
        r = evaluate(g, RHOS_DEFAULT, 16000, sig, delta, 0.05, n_trials,
                     Kmax=49, seed=3300)
        _row("deg-3-tail", 49, 0.05, sig, r)

    # --- (4) d sweep (cost decoupling check) ---
    print("\n[4] d sweep  (deg-3-tail, alpha=5%, sigma=0.1, N0=16k)")
    print(header)
    for d in [25, 49, 100, 196]:
        g, _ = make_scenario("deg-3-tail", d=d, seed=seed)
        r = evaluate(g, RHOS_DEFAULT, 16000, 0.1, delta, 0.05, n_trials,
                     Kmax=d, seed=3400)
        _row("deg-3-tail", d, 0.05, 0.1, r)

    # --- (5) budget sweep: how much Q to certify K=1 insuff on deg-3-tail ---
    print("\n[5] budget sweep  (deg-3-tail, d=49, alpha=5%, sigma=0.1)")
    print(header)
    for N0 in [2000, 4000, 8000, 16000, 32000, 64000]:
        g, _ = make_scenario("deg-3-tail", d=49, seed=seed)
        r = evaluate(g, RHOS_DEFAULT, N0, 0.1, delta, 0.05, n_trials,
                     Kmax=49, seed=3500)
        _row("deg-3-tail", 49, 0.05, 0.1, r)

    print("\nReadout: FALSE-ins and FALSE-suf must be ~0 everywhere (bounded by "
          "delta). pure-1 must show K1-insuf=0 AND FALSE-ins=0.")


def _row(name, d, alpha, sig, r):
    print(f"{name:>12} {d:>4} {alpha:>5.2f} {sig:>5.2f} {r['N0']:>7} {r['Q']:>9,} "
          f"{r['feas']:>6.3f} {r['kbar_eq']:>8.3f} {r['k1_insuf']:>9.3f} "
          f"{r['false_insuf']:>10.3f} {r['false_suf']:>10.3f} {r['Kst']:>3} "
          f"{r['E1']:>6.3f}")


# ----------------------------------------------------------------------
# GRID ABLATION  (matched total Q)
# ----------------------------------------------------------------------
def _grid(name):
    """Candidate overlap grids, all including rho=0."""
    g = {
        "L4":        np.r_[0.0, 0.5, 0.8, 0.97],
        "L6":        np.r_[0.0, np.linspace(0.5, 0.97, 5)],
        "L9":        np.r_[0.0, np.linspace(0.5, 0.97, 8)],
        "L6-near1":  np.r_[0.0, 0.5, 0.7, 0.85, 0.95, 0.99],
        "L9-near1":  np.r_[0.0, np.linspace(0.5, 0.9, 5), 0.95, 0.98, 0.995],
    }
    return g[name]


def grid_ablation(Q_total=200_000, n_trials=120, delta=0.1, seed=7):
    print("\n" + "=" * 78)
    print(f"GRID ABLATION  (matched total Q={Q_total:,}; N0=Q//(L+2); rho=0 always)")
    print("=" * 78)
    print("Objective: maximize K=1-insuff rate on deg-3-tail at fixed Q, FALSE~0.")
    print(f"{'grid':>10} {'L':>3} {'N0':>7} {'Q_real':>9} {'feas':>6} "
          f"{'Kbar=K*':>8} {'K1-insuf':>9} {'FALSE-ins':>10} {'FALSE-suf':>10} "
          f"{'meanRho':>8}")

    g_fn, _ = make_scenario("deg-3-tail", d=49, seed=seed)
    best = None
    for name in ["L4", "L6", "L9", "L6-near1", "L9-near1"]:
        rhos = _grid(name)
        L = len(rhos)
        N0 = Q_total // (L + 2)
        r = evaluate(g_fn, rhos, N0, 0.1, delta, 0.05, n_trials,
                     Kmax=49, seed=4100)
        print(f"{name:>10} {L:>3} {N0:>7} {r['Q']:>9,} {r['feas']:>6.3f} "
              f"{r['kbar_eq']:>8.3f} {r['k1_insuf']:>9.3f} "
              f"{r['false_insuf']:>10.3f} {r['false_suf']:>10.3f} "
              f"{rhos.mean():>8.3f}")
        ok = (r['false_insuf'] < 0.02 and r['false_suf'] < 0.02)
        if ok and (best is None or r['k1_insuf'] > best[1]):
            best = (name, r['k1_insuf'])
    if best:
        print(f"\n  -> best grid at Q={Q_total:,}: {best[0]} "
              f"(K1-insuf={best[1]:.3f}, FALSE rates within delta)")
    print("  (Grids pushing nearer rho=1 sharpen high-degree separation but lose "
          "SNR there; matched-Q picks the winner empirically.)")


RHOS_DEFAULT = np.r_[0.0, np.linspace(0.5, 0.97, 8)]


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"device = {DEVICE}")
    stress_test()
    grid_ablation(Q_total=200_000)
    grid_ablation(Q_total=400_000)