"""
v3_3: (1) HELD-OUT synthetic validation on L4 (NO grid tuning) and
      (2) CLEAN sigma=0 SCREENING harness for the real-model step.

L4 was selected by the matched-Q grid ablation (v3_2).  Here it is FROZEN:
    L4 = [0.0, 0.5, 0.8, 0.97]
and evaluated on fresh spectra / seeds NOT used in selection -- especially a
weak pairwise block and a lone high-degree interaction term.  No grid is changed.

Framing (locked):  CERTIFIED SCREENING.  The deliverable is a three-way verdict
on first order --
    K=1 INSUFFICIENT  : provably E(1) > alpha            (first order not enough)
    K=1 SUFFICIENT    : provably E(1) <= alpha            (first order enough)
    UNRESOLVED        : neither, at this pilot budget     (honest abstention)
Exact K* recovery is NOT claimed (honest Kmax=d defers it); do not report Kbar=K*.

Real setup (real_screen):  deterministic model output in [0,1] (e.g. a class
probability), sigma=0, so the self-pair noise query is dropped automatically
(skip_selfpair).  Pilot cost is (L+1)*N0 queries; we report queries-per-input
and the three verdict fractions over inputs.

Run it yourself:   python v3_3.py
"""

import numpy as np
import torch
from verify_tolerance_order_v3 import (
    WalshFunction, run_pilot_batched, certify_order,
    SUFFICIENT, INSUFFICIENT, UNRESOLVED, DEVICE, PILOT_DTYPE,
)

# FROZEN grid from the v3_2 matched-Q ablation.  Do not retune.
L4 = np.array([0.0, 0.5, 0.8, 0.97])


def pilot_queries_clean(L, N0, skip_selfpair):
    """Query accounting.
       skip_selfpair (sigma=0): shared base z (N0) + L coupled (L*N0) = (L+1)*N0.
       else: + N0 self-pair re-read = (L+2)*N0."""
    return (L + 1) * N0 if skip_selfpair else (L + 2) * N0


def screen_first_order(s_hat, eta_s, rhos, Kmax, alpha):
    """Three-way verdict on K=1 only (screening), from the K=1 entry of certify_order."""
    verdict, _, _ = certify_order(s_hat, rhos, Kmax, eta_s, alpha)
    v = verdict.get(1, UNRESOLVED)
    return v


# ----------------------------------------------------------------------
# (1) HELD-OUT SYNTHETIC  (L4 frozen)
# ----------------------------------------------------------------------
def heldout_synthetic(n_trials=160, delta=0.1, alpha=0.05, sigma_obs=0.1, d=49,
                      N0=16000):
    print("\n" + "=" * 76)
    print(f"HELD-OUT SYNTHETIC  (L4 FROZEN={list(L4)}; d={d}, alpha={alpha}, "
          f"sigma={sigma_obs}, N0={N0})")
    print("=" * 76)
    Kmax = d
    Q = pilot_queries_clean(len(L4), N0, skip_selfpair=(sigma_obs == 0))
    print(f"pilot queries Q = {Q:,}   trials = {n_trials}   (NO grid tuning here)")
    print(f"{'scenario':>14} {'seed':>5} {'K*':>3} {'E1':>6} {'insuf':>7} "
          f"{'suf':>6} {'unres':>7} {'FALSE-ins':>10} {'FALSE-suf':>10}")

    # fresh spectra + seeds NOT used in v3_2 selection
    cases = [
        ("weak-pair-.10", {1: 1.0, 2: 0.10}, 21),   # E1~0.091 > a: K*=2, insuff CORRECT
        ("weak-pair-.04", {1: 1.0, 2: 0.04}, 22),   # E1~0.038 <= a=0.05: K*=1, must NOT insuff
        ("lone-deg4",     {1: 1.0, 4: 0.12}, 23),   # single high-order term, no deg2/3
        ("lone-deg6",     {1: 1.0, 6: 0.10}, 24),   # even higher lone interaction
        ("deg-2-clean",   {1: 1.0, 2: 0.35}, 25),   # easy reference, K*=2
        ("pure-1-holdout",{1: 1.0},          26),   # K*=1 gate on a new seed
    ]
    for name, energies, seed in cases:
        g = WalshFunction(d, energies, mean=0.5, seed=seed)
        T = g.a_true[1:].sum()
        E1 = g.a_true[2:].sum() / T if T > 0 else 0.0
        Kst = 1 if E1 <= alpha else _kstar(g.a_true, alpha)

        s_hat, eta_s = run_pilot_batched(g, L4, N0, sigma_obs, n_trials,
                                         seed=9000 + seed, delta=delta)
        ins = suf = unr = f_ins = f_suf = 0
        for t in range(n_trials):
            v = screen_first_order(s_hat[t], eta_s[t], L4, Kmax, alpha)
            if v == INSUFFICIENT:
                ins += 1
                if E1 <= alpha:
                    f_ins += 1
            elif v == SUFFICIENT:
                suf += 1
                if E1 > alpha:
                    f_suf += 1
            else:
                unr += 1
        n = n_trials
        print(f"{name:>14} {seed:>5} {Kst:>3} {E1:>6.3f} {ins/n:>7.3f} "
              f"{suf/n:>6.3f} {unr/n:>7.3f} {f_ins/n:>10.3f} {f_suf/n:>10.3f}")
    print("\nGate: FALSE-ins and FALSE-suf must be ~0.  pure-1-holdout & weak-pair "
          "with E1<=alpha must NOT certify insufficient.")


def _kstar(a_true, alpha):
    T = a_true[1:].sum()
    if T <= 0:
        return 1
    for K in range(1, len(a_true)):
        if a_true[K + 1:].sum() / T <= alpha:
            return K
    return len(a_true) - 1


# ----------------------------------------------------------------------
# (2) REAL SCREENING HARNESS  (sigma=0, deterministic output in [0,1])
# ----------------------------------------------------------------------
class DeterministicProbModel:
    """Stand-in for a real classifier's masked class-probability g(z) in [0,1].
    REPLACE .eval with the real forward pass: take a mask batch z in {0,1}^d,
    build the masked input, run the model, return the target-class probability.
    Here we synthesize a deterministic [0,1] response with known interaction
    structure so the harness is exercisable end-to-end before wiring a real net.
    The interface (.eval(Z)->(...,) float in [0,1], .d, .device) matches what
    run_pilot_batched expects from WalshFunction."""

    def __init__(self, d, energies, mean=0.5, seed=0, device=DEVICE):
        self._w = WalshFunction(d, energies, mean=mean, seed=seed, device=device)
        self.d = d
        self.device = device
        self.a_true = self._w.a_true        # for reporting only; real model has none
        self.member = self._w.member
        self.beta = self._w.beta
        self.mean = self._w.mean

    @torch.no_grad()
    def eval(self, Z):
        raw = self._w.eval(Z)
        return torch.clamp(raw, 0.0, 1.0)   # deterministic, bounded in [0,1]


def real_screen(n_inputs=50, delta=0.1, alpha=0.05, d=49, N0=16000, seed=31):
    """Clean sigma=0 screening over many 'inputs' (each = a fresh masked response).
    Reports ONLY the three screening fractions + queries/input.  No K* claim."""
    print("\n" + "=" * 76)
    print(f"REAL SCREENING (sigma=0, output in [0,1], L4 frozen)  "
          f"d={d}, alpha={alpha}, N0={N0}")
    print("=" * 76)
    sigma_obs = 0.0
    Kmax = d
    skip = True
    Qpi = pilot_queries_clean(len(L4), N0, skip_selfpair=skip)
    print(f"queries per input Q = {Qpi:,}   (= (L+1)*N0, self-pair dropped at sigma=0)")
    print(f"inputs = {n_inputs}")

    # heterogeneous population of 'inputs': some genuinely first-order, some not
    rng = np.random.default_rng(seed)
    pops = {
        "mostly-1st (E1<=a)":  lambda s: {1: 1.0, 2: float(rng.uniform(0.0, 0.04))},
        "borderline":          lambda s: {1: 1.0, 2: float(rng.uniform(0.04, 0.09))},
        "needs-2nd (E1>a)":    lambda s: {1: 1.0, 2: float(rng.uniform(0.15, 0.45))},
        "high-order tail":     lambda s: {1: 1.0, 2: 0.20, 3: float(rng.uniform(0.05, 0.2))},
    }
    print(f"{'population':>20} {'insuf':>7} {'suf':>6} {'unres':>7} {'Q/input':>9}")
    for label, gen_e in pops.items():
        ins = suf = unr = 0
        for i in range(n_inputs):
            energies = gen_e(i)
            model = DeterministicProbModel(d, energies, mean=0.5, seed=1000 + i)
            s_hat, eta_s = run_pilot_batched(model, L4, N0, sigma_obs, 1,
                                             seed=70000 + i, delta=delta,
                                             skip_selfpair=skip)
            v = screen_first_order(s_hat[0], eta_s[0], L4, Kmax, alpha)
            ins += (v == INSUFFICIENT); suf += (v == SUFFICIENT); unr += (v == UNRESOLVED)
        n = n_inputs
        print(f"{label:>20} {ins/n:>7.3f} {suf/n:>6.3f} {unr/n:>7.3f} {Qpi:>9,}")
    print("\nDeliverable per input: {K=1 insufficient | sufficient | unresolved}.")
    print("Wire a real net by replacing DeterministicProbModel.eval with its")
    print("masked forward pass returning the target-class probability in [0,1].")


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"device = {DEVICE}")
    heldout_synthetic()
    real_screen()
