"""
v3_3: (1) HELD-OUT synthetic validation on L4 (NO grid tuning) and
      (2) CLEAN sigma=0 SCREENING harness for the real-model step.

L4 was selected on a DEVELOPMENT synthetic suite that optimized the K=1-INSUFFICIENT
rate (not balanced three-way accuracy, not the 'sufficient' direction), at a
development budget (Q=200k/400k).  It is FROZEN here before the disjoint held-out and
real evaluations -- which run at different budgets (Q~96k / 80k), so this is NOT a
'matched-Q grid for the deployed budget':
    L4 = [0.0, 0.5, 0.8, 0.97]
Evaluated on fresh spectra / seeds NOT used in selection -- a weak pairwise block and
a lone high-degree interaction term.  No grid is changed here.

Framing (locked):  CERTIFIED SCREENING, and specifically a CONSERVATIVE ONE-SIDED
DETECTOR of when first order is INSUFFICIENT.  Three verdicts are emitted --
    K=1 INSUFFICIENT  : provably E(1) > alpha            (first order not enough)
    K=1 SUFFICIENT    : provably E(1) <= alpha            (first order enough; LOW POWER
                                                           at this budget -- rare)
    UNRESOLVED        : neither, at this pilot budget     (honest abstention; common)
This is NOT a balanced three-way classifier: the 'sufficient' direction is weak (in
held-out, pure first order certifies sufficient only a few percent of the time).
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
    WalshFunction, run_pilot_batched, certify_order, audit_determinism,
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
    print("radius_mode=empirical (sigma>0 Gaussian noise is unbounded; this is a "
          "calibration, not a theorem-exact certificate -- see real_screen for the "
          "theorem-exact sigma=0 path).")
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
class BoundedWalshProb:
    """Synthetic probability response in [0.5 - hw, 0.5 + hw] with EXACT known spectrum.

    The map is AFFINE:  g(z) = 0.5 + scale * w(z),  w a zero-mean Walsh function.
    Affine transforms do NOT create new interactions (unlike clamp): the per-degree
    energies all scale by scale^2, so the residual ratio E(K) is INVARIANT and a_true
    stays valid for the response the pilot actually sees.  scale = hw / ||beta||_1
    guarantees |w| <= ||beta||_1 hence g in [0.5-hw, 0.5+hw].

    NOTE: ||beta||_1 is a worst-case L1 bound; for spectra with many terms it makes
    scale tiny and shrinks the difference statistic (lower SNR -> more 'unresolved').
    That is a property of demanding a hard [0,1] envelope, not a certificate fault.
    We print scale and true E1 so this is visible, never silent."""

    def __init__(self, d, energies, hw=0.45, seed=0, device=DEVICE):
        self._w = WalshFunction(d, energies, mean=0.0, seed=seed, device=device)
        self.d = d
        self.device = device
        self.member = self._w.member
        self.beta = self._w.beta
        self.mean = 0.5

        l1 = float(self._w.beta.abs().sum().item()) if self._w.beta is not None else 0.0
        self.scale = hw / max(l1, 1e-12)

        # exact post-affine spectrum: a_j scale by scale^2 for j>=1; a_0 = mean^2 = 0.25
        self.a_true = self._w.a_true.copy()
        self.a_true[1:] *= self.scale ** 2
        self.a_true[0] = 0.25

    @torch.no_grad()
    def eval(self, Z):
        return 0.5 + self.scale * self._w.eval(Z)   # affine, bounded, deterministic

    def E1(self):
        T = self.a_true[1:].sum()
        return self.a_true[2:].sum() / T if T > 0 else 0.0


def real_screen(n_inputs=50, delta=0.1, alpha=0.05, d=49, N0=16000, seed=31, hw=0.45):
    """Clean sigma=0 screening over many synthetic 'inputs' (affine bounded Walsh,
    exact spectrum).  radius_mode='theorem' (range of (1/2)(y-y')^2 is 0.5*1^2 for
    outputs in [0,1]) so reported verdicts are theorem-exact, not calibration.
    Reports the three screening fractions + FALSE rates (we DO know true E1 here, so
    we check them) + queries/input.  No K* / Kbar claim."""
    print("\n" + "=" * 76)
    print(f"REAL SCREENING (sigma=0, affine output in [{0.5-hw:.2f},{0.5+hw:.2f}], "
          f"L4 frozen, theorem radius)  d={d}, alpha={alpha}, N0={N0}")
    print("=" * 76)
    sigma_obs = 0.0
    Kmax = d
    skip = True

    # --- determinism audit: skip_selfpair only valid if model is exactly repeatable ---
    probe = BoundedWalshProb(d, {1: 1.0, 2: 0.2}, hw=hw, seed=0)
    dd = audit_determinism(probe)
    print(f"determinism audit  max|y1-y2| = {dd:.2e}  "
          f"-> skip_selfpair {'OK' if dd == 0.0 else 'UNSAFE (fall back to self-pair!)'}")
    if dd != 0.0:
        print("  WARNING: output not bit-exact repeatable; disabling skip_selfpair.")
        skip = False
        sigma_obs = 0.0  # still no injected noise, but keep self-pair accounting

    Qpi = pilot_queries_clean(len(L4), N0, skip_selfpair=skip)
    print(f"queries per input Q = {Qpi:,}   "
          f"({'(L+1)*N0' if skip else '(L+2)*N0'})   inputs = {n_inputs}")

    # populations labelled by TRUE post-affine E1 (threshold x*=0.0526 for a1=1):
    rng = np.random.default_rng(seed)
    pops = {
        "mostly-1st (E1<a)": lambda: {1: 1.0, 2: float(rng.uniform(0.0, 0.04))},
        "just-below":        lambda: {1: 1.0, 2: float(rng.uniform(0.030, 0.045))},
        "just-above":        lambda: {1: 1.0, 2: float(rng.uniform(0.060, 0.090))},
        "needs-2nd (E1>a)":  lambda: {1: 1.0, 2: float(rng.uniform(0.15, 0.45))},
        "high-order tail":   lambda: {1: 1.0, 2: 0.20, 3: float(rng.uniform(0.05, 0.2))},
    }
    print(f"{'population':>20} {'meanE1':>7} {'insuf':>7} {'suf':>6} {'unres':>7} "
          f"{'FALSE-ins':>10} {'FALSE-suf':>10} {'Q/input':>9}")
    for label, gen_e in pops.items():
        ins = suf = unr = f_ins = f_suf = 0
        e1s = []
        for i in range(n_inputs):
            model = BoundedWalshProb(d, gen_e(), hw=hw, seed=1000 + i)
            E1 = model.E1(); e1s.append(E1)
            s_hat, eta_s = run_pilot_batched(model, L4, N0, sigma_obs, 1,
                                             seed=70000 + i, delta=delta,
                                             skip_selfpair=skip,
                                             radius_mode="theorem", out_range=1.0)
            v = screen_first_order(s_hat[0], eta_s[0], L4, Kmax, alpha)
            if v == INSUFFICIENT:
                ins += 1; f_ins += (E1 <= alpha)
            elif v == SUFFICIENT:
                suf += 1; f_suf += (E1 > alpha)
            else:
                unr += 1
        n = n_inputs
        print(f"{label:>20} {np.mean(e1s):>7.3f} {ins/n:>7.3f} {suf/n:>6.3f} "
              f"{unr/n:>7.3f} {f_ins/n:>10.3f} {f_suf/n:>10.3f} {Qpi:>9,}")
    print("\nDeliverable per input: {K=1 insufficient | sufficient | unresolved}.")
    print("FALSE-ins / FALSE-suf must be ~0 (now meaningful: affine keeps E1 exact).")
    print("Wire a real net by replacing BoundedWalshProb with an adapter whose .eval")
    print("returns the masked target-class probability in [0,1].")


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"device = {DEVICE}")
    heldout_synthetic()
    real_screen()
