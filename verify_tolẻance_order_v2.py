"""
Synthetic verification of the tolerance-order certificate (Sections 3-5), v2.

This is a corrected rewrite addressing the four blockers raised in review:

  (#1) CERTIFICATE IS A VALID CERTIFICATE.  The polytope A(eta) of Eq. (8) is now
       implemented as interval constraints, exactly as the Theorem-1 proof requires:
           |a_0 - a0_hat|              <= eta_0
           |sum_{j>=1} a_j - T_hat|    <= eta_T
           |sum_j a_j rho_l^j - c(rho_l)| <= eta_l   for each l
       The OLD code pinned a_0 = a0_hat and sum a_j = T_hat as EQUALITIES (A_eq),
       so the true (noisy-estimate-mismatched) spectrum a* was almost surely
       INFEASIBLE -> the LP bounds were not certificates.  Fixed: no A_eq at all.

       And we do NOT certify by tail/T_hat (a ratio with an uncertain denominator).
       We certify the SIGN of the homogeneous functional
           h_K(a) = sum_{j>K} a_j  -  alpha * sum_{j>=1} a_j
       over A(eta):
           max_{a in A} h_K(a) <= 0   => K certified SUFFICIENT
           min_{a in A} h_K(a) >  0   => K certified INSUFFICIENT
           otherwise                  => UNRESOLVED
       h_K is linear and scale-homogeneous in a, so the uncertain T never appears
       in a denominator.  This is the clean form.

  (#2) NO ORACLE SUPPORT.  The honest default sets Kmax = d (the true Walsh degree
       can be up to d), and the synthetic spectrum carries a real tail at degrees
       ABOVE the target order.  Runs with Kmax = support are kept but LABELLED
       "oracle-support ablation".

  (#3) QUERY ACCOUNTING.  Every table reports total model queries
           Q = 2*L*N0  (coupled pairs)  +  2*N0 (c(1) paired re-query)  +  N0 (a0 mean)
       and, where a degree-K dense fit is well posed, the baseline fit cost p_K so
       the reader can see when the pilot is actually cheaper.

  (#4) HONEST RADIUS.  eta is an empirical-Bernstein radius for the sub-exponential
       product y*y':  variance term PLUS a (range * log/N0) correction, with SEPARATE
       intervals for a_0 = (E g)^2 and for c(1).  No clamp-to-zero inside the band.

Precision: float32 in the pilot tensor work; float64 from c_hat onward (LP inputs).
GPU-batched across trials.  Run it yourself:  python verify_tolerance_order_v2.py
"""

import numpy as np
import torch
from itertools import combinations
from math import comb
from scipy.optimize import linprog

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PILOT_DTYPE = torch.float32
LP_DTYPE = np.float64

# verdict codes
SUFFICIENT, INSUFFICIENT, UNRESOLVED = "suff", "insuff", "unres"


# ----------------------------------------------------------------------
# Ground-truth Walsh function (torch eval) -- now supports a high-degree tail
# ----------------------------------------------------------------------
class WalshFunction:
    """g(z) = mean + sum_S beta_S * prod_{i in S}(2 z_i - 1).
    energies: dict {degree j -> per-degree energy a_j}.  Degrees may exceed any
    Kmax used in the certificate, so a genuine tail above the target order exists."""

    def __init__(self, d, energies, mean=0.0, seed=0, device=DEVICE):
        rng = np.random.default_rng(seed)
        self.d = d
        self.mean = float(mean)
        self.device = device
        max_deg = max(energies) if energies else 0
        self.a_true = np.zeros(max_deg + 1)
        self.a_true[0] = mean ** 2

        subsets, betas = [], []
        for j, a_j in energies.items():
            if a_j <= 0 or j < 1 or j > d:
                continue
            n_all = comb(d, j)
            k = min(n_all, 8)          # spread a_j over up to 8 random subsets of size j
            # sample k distinct subsets of size j without enumerating all C(d,j)
            chosen = set()
            while len(chosen) < k:
                chosen.add(tuple(sorted(rng.choice(d, size=j, replace=False).tolist())))
            chosen = [list(s) for s in chosen]
            raw = rng.standard_normal(k)
            raw = raw / np.sqrt(np.sum(raw ** 2))
            b = raw * np.sqrt(a_j)
            for S, bb in zip(chosen, b):
                subsets.append(S)
                betas.append(float(bb))
            self.a_true[j] = a_j

        self.n_terms = len(subsets)
        if self.n_terms > 0:
            M = torch.zeros((self.n_terms, d), dtype=PILOT_DTYPE, device=device)
            for t, S in enumerate(subsets):
                if S:
                    M[t, S] = 1.0
            self.member = M
            self.beta = torch.tensor(betas, dtype=PILOT_DTYPE, device=device)
        else:
            self.member = None
            self.beta = None

    @torch.no_grad()
    def eval(self, Z):
        """Z: (..., d) in {0,1}. Memory-flat fold over feature columns; peak tensor
        is (..., n_terms), never (..., n_terms, d)."""
        signed = (2.0 * Z - 1.0).to(PILOT_DTYPE)
        lead = Z.shape[:-1]
        out = torch.full(lead, self.mean, dtype=PILOT_DTYPE, device=Z.device)
        if self.member is None:
            return out
        n_terms = self.member.shape[0]
        chi = torch.ones((*lead, n_terms), dtype=PILOT_DTYPE, device=Z.device)
        for i in range(self.d):
            mem_i = self.member[:, i]
            if not bool(torch.any(mem_i)):
                continue
            s_i = signed[..., i].unsqueeze(-1)
            chi = chi * (1.0 - mem_i * (1.0 - s_i))
        return out + (chi * self.beta).sum(dim=-1)


# ----------------------------------------------------------------------
# Vectorized pilot across trials
# ----------------------------------------------------------------------
@torch.no_grad()
def _coupled(d, rho, shape_lead, gen, device):
    z = torch.randint(0, 2, (*shape_lead, d), generator=gen, device=device, dtype=PILOT_DTYPE)
    keep = torch.rand((*shape_lead, d), generator=gen, device=device) < rho
    fresh = torch.randint(0, 2, (*shape_lead, d), generator=gen, device=device, dtype=PILOT_DTYPE)
    return z, torch.where(keep, z, fresh)


def _auto_chunk(n_trials, N0, n_terms, d, device, budget_gib=2.0):
    if str(device) == "cpu":
        return n_trials
    per_trial = N0 * max(n_terms, d) * 4 * 8
    chunk = max(1, int(budget_gib * (1024 ** 3) // max(per_trial, 1)))
    return min(chunk, n_trials)


@torch.no_grad()
def run_pilot_batched(g, rhos, N0, sigma_obs, n_trials, seed, delta, B=None, chunk=None):
    n_terms = 0 if g.member is None else g.member.shape[0]
    if chunk is None:
        chunk = _auto_chunk(n_trials, N0, n_terms, g.d, g.device)
    parts = {"c": [], "a0": [], "T": [], "ec": [], "e0": [], "eT": []}
    start = 0
    while start < n_trials:
        m = min(chunk, n_trials - start)
        out = _run_pilot_core(g, rhos, N0, sigma_obs, m, seed + start, delta, B)
        for k, v in zip(parts, out):
            parts[k].append(v)
        start += m
    return tuple(np.concatenate(parts[k], 0) for k in parts)


@torch.no_grad()
def _run_pilot_core(g, rhos, N0, sigma_obs, n_trials, seed, delta, B):
    """Returns (per-trial, float64):
        c_hat (n,L), a0_hat (n,), T_hat (n,),
        eta_c (n,L)   per-overlap radius,
        eta_0 (n,)    radius on a_0,
        eta_T (n,)    radius on T.
    Empirical-Bernstein with sub-exponential correction (#4)."""
    device = g.device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    L = len(rhos)
    lead = (n_trials, N0)

    # union bound over L overlaps + 2 endpoints; two-sided
    M_terms = L + 2
    log_term = float(np.log(2.0 * M_terms / delta))
    # range of the product y*y' (sub-exponential).  If B unknown, estimate from data.
    def bernstein(std, rng):  # std,(rng) tensors -> radius tensor
        var = std ** 2
        return torch.sqrt(2.0 * var * log_term / N0) + (rng * log_term) / (3.0 * N0)

    c_hat = torch.empty((n_trials, L), dtype=PILOT_DTYPE, device=device)
    c_std = torch.empty((n_trials, L), dtype=PILOT_DTYPE, device=device)
    c_rng = torch.empty((n_trials, L), dtype=PILOT_DTYPE, device=device)
    for l, rho in enumerate(rhos):
        z, z2 = _coupled(g.d, float(rho), lead, gen, device)
        y = g.eval(z) + sigma_obs * torch.randn(lead, generator=gen, device=device)
        yp = g.eval(z2) + sigma_obs * torch.randn(lead, generator=gen, device=device)
        prod = y * yp
        c_hat[:, l] = prod.mean(dim=1)
        c_std[:, l] = prod.std(dim=1)
        c_rng[:, l] = prod.amax(dim=1) - prod.amin(dim=1)

    # c(1) via paired re-query of the SAME mask under independent noise
    z, _ = _coupled(g.d, 1.0, lead, gen, device)
    gz = g.eval(z)
    y1 = gz + sigma_obs * torch.randn(lead, generator=gen, device=device)
    y2 = gz + sigma_obs * torch.randn(lead, generator=gen, device=device)
    prod1 = y1 * y2
    c1_hat = prod1.mean(dim=1)
    c1_std = prod1.std(dim=1)
    c1_rng = prod1.amax(dim=1) - prod1.amin(dim=1)

    # a0 = (E g)^2, debias the (1/N0) sigma^2 inflation (kept), but DON'T clamp inside band
    zm, _ = _coupled(g.d, 1.0, lead, gen, device)
    ym = g.eval(zm) + sigma_obs * torch.randn(lead, generator=gen, device=device)
    mean_est = ym.mean(dim=1)
    g_std = ym.std(dim=1)
    g_rng = ym.amax(dim=1) - ym.amin(dim=1)
    a0_hat = mean_est ** 2 - (sigma_obs ** 2) / N0   # may be slightly <0; band handles it

    T_hat = c1_hat - a0_hat

    # radii
    eta_c = bernstein(c_std, c_rng)                                   # (n,L)
    rad_mean = bernstein(g_std, g_rng)                               # radius on E[g]
    # a0=(E g)^2: |a0 - a0_hat| <= 2|mean|*rad_mean + rad_mean^2  (exact propagation, no delta-method)
    eta_0 = 2.0 * mean_est.abs() * rad_mean + rad_mean ** 2          # (n,)
    rad_c1 = bernstein(c1_std, c1_rng)                              # radius on c(1)
    eta_T = rad_c1 + eta_0                                           # T = c1 - a0

    to64 = lambda t: t.detach().to("cpu").double().numpy()
    return (to64(c_hat), to64(a0_hat), to64(T_hat),
            to64(eta_c), to64(eta_0), to64(eta_T))


# ----------------------------------------------------------------------
# (#1) Certificate by feasibility -- interval polytope A(eta), sign-of-h_K objective
# ----------------------------------------------------------------------
def _build_polytope(c_hat, a0_hat, T_hat, rhos, Kmax, eta_c, eta_0, eta_T):
    """A(eta) = { a >= 0 :
        |a_0 - a0_hat| <= eta_0,
        |sum_{j>=1} a_j - T_hat| <= eta_T,
        |sum_j a_j rho_l^j - c_hat_l| <= eta_c_l  for all l }.
    Returns (A_ub, b_ub, bounds, nvar).  NO equality constraints (#1)."""
    nvar = Kmax + 1
    rhos = np.asarray(rhos, dtype=LP_DTYPE)
    eta_c = np.broadcast_to(np.asarray(eta_c, dtype=LP_DTYPE), (len(rhos),))
    powers = np.array([[rho ** j for j in range(nvar)] for rho in rhos], dtype=LP_DTYPE)  # (L,nvar)

    e0 = np.zeros(nvar); e0[0] = 1.0                # picks a_0
    eT = np.zeros(nvar); eT[1:] = 1.0               # picks sum_{j>=1} a_j

    A_ub = np.vstack([
        powers, -powers,                            # curve, two-sided
        e0[None, :], -e0[None, :],                  # a_0, two-sided
        eT[None, :], -eT[None, :],                  # T,   two-sided
    ])
    b_ub = np.concatenate([
        c_hat + eta_c, -(c_hat - eta_c),
        [a0_hat + eta_0], [-(a0_hat - eta_0)],
        [T_hat + eta_T], [-(T_hat - eta_T)],
    ]).astype(LP_DTYPE)
    bounds = [(0, None)] * nvar
    return A_ub, b_ub, bounds, nvar


def certify_order(c_hat, a0_hat, T_hat, rhos, Kmax, eta_c, eta_0, eta_T, alpha):
    """For each K in 1..Kmax, certify the SIGN of
        h_K(a) = sum_{j>K} a_j - alpha * sum_{j>=1} a_j
    over A(eta).  Returns dict K -> verdict in {SUFFICIENT, INSUFFICIENT, UNRESOLVED},
    plus (h_lo, h_hi) arrays for inspection.  No division by T_hat (#1)."""
    A_ub, b_ub, bounds, nvar = _build_polytope(
        c_hat, a0_hat, T_hat, rhos, Kmax, eta_c, eta_0, eta_T)

    eT = np.zeros(nvar); eT[1:] = 1.0
    verdict = {}
    h_lo = np.full(Kmax + 1, np.nan)
    h_hi = np.full(Kmax + 1, np.nan)
    for K in range(1, Kmax + 1):
        tail = np.zeros(nvar); tail[K + 1:] = 1.0
        obj = tail - alpha * eT                      # coefficients of h_K
        r_min = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        r_max = linprog(-obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not (r_min.success and r_max.success):
            verdict[K] = UNRESOLVED            # infeasible/unbounded -> defer, never assume
            continue
        lo, hi = r_min.fun, -r_max.fun
        h_lo[K], h_hi[K] = lo, hi
        if hi <= 0:
            verdict[K] = SUFFICIENT
        elif lo > 0:
            verdict[K] = INSUFFICIENT
        else:
            verdict[K] = UNRESOLVED
    return verdict, h_lo, h_hi


def certified_Kbar(verdict, Kmax):
    """Three-way read-off (#1 output semantics): smallest K certified SUFFICIENT.
    Returns (Kbar or None, 'highest K certified insufficient-through')."""
    Kbar = next((K for K in range(1, Kmax + 1) if verdict.get(K) == SUFFICIENT), None)
    insuff_through = 0
    for K in range(1, Kmax + 1):
        if verdict.get(K) == INSUFFICIENT:
            insuff_through = K
        else:
            break
    return Kbar, insuff_through


# ----------------------------------------------------------------------
# Truth utilities and query accounting (#3)
# ----------------------------------------------------------------------
def true_residual_curve(a_true, Kmax):
    a = np.zeros(Kmax + 1)
    a[:min(len(a_true), Kmax + 1)] = a_true[:Kmax + 1]
    T = a_true[1:].sum()
    E = np.array([a_true[K + 1:].sum() / T if T > 0 else 0.0 for K in range(Kmax + 1)])
    return E, T


def Kstar(a_true, alpha):
    T = a_true[1:].sum()
    if T <= 0:
        return 1
    for K in range(1, len(a_true)):
        if a_true[K + 1:].sum() / T <= alpha:
            return K
    return len(a_true) - 1


def pilot_queries(L, N0):
    """Total model queries the pilot spends (#3)."""
    return 2 * L * N0 + 2 * N0 + N0     # coupled pairs + c(1) re-query + a0 mean


def dense_fit_queries(d, K):
    """Well-posedness floor for a degree-K dense surrogate: p_K = sum_{k<=K} C(d,k)."""
    return sum(comb(d, k) for k in range(0, K + 1))


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
RHOS = np.linspace(0.5, 0.97, 8)


def exp_coverage(n_trials=200, d=49, N0=16000, sigma_obs=0.1, delta=0.1, seed=1):
    """(a) HONEST default Kmax=d, spectrum has a real degree-3 tail above target.
    Reports: feasibility of a* (not residual-band), coverage of h_K sign, the
    INSUFFICIENT-K=1 rate with soundness, total queries Q vs baseline fit cost."""
    L = len(RHOS)
    Kmax = d                                          # (#2) no oracle support
    alpha = 0.05
    # deg 1,2 plus a thin deg-3 tail -> K* honest under Kmax=d
    energies = {1: 1.0, 2: 0.35, 3: 0.04}
    g = WalshFunction(d, energies, mean=0.5, seed=seed)
    Kst = Kstar(g.a_true, alpha)
    E_true, T_true = true_residual_curve(g.a_true, min(5, Kmax))

    print(f"\n=== (a) COVERAGE (honest Kmax=d)  d={d} N0={N0} sigma={sigma_obs} "
          f"delta={delta} dev={DEVICE} ===")
    print(f"    spectrum a_1..a_3 = {np.round(g.a_true[1:4],4)}   T={T_true:.4f}")
    print(f"    true residual E(1..3) = {np.round(E_true[1:4],4)}   K*_5% = {Kst}")
    Q = pilot_queries(L, N0)
    print(f"    pilot total queries Q = {Q:,}   |  dense-fit floor: "
          f"p1={dense_fit_queries(d,1)}, p2={dense_fit_queries(d,2)}, "
          f"p3={dense_fit_queries(d,3):,}")

    c, a0, T, ec, e0, eT = run_pilot_batched(g, RHOS, N0, sigma_obs, n_trials,
                                             seed=1000, delta=delta)
    # true spectrum vector over 0..Kmax for feasibility test
    a_star = np.zeros(Kmax + 1)
    a_star[:len(g.a_true)] = g.a_true

    feasible = 0
    h1_insuff = 0           # K=1 certified insufficient
    h1_insuff_sound = 0     # and truly insufficient
    h1_false_suff = 0       # K=1 wrongly certified sufficient (must be 0)
    kbar_correct = 0
    used = 0
    for t in range(n_trials):
        A_ub, b_ub, bounds, nvar = _build_polytope(
            c[t], a0[t], T[t], RHOS, Kmax, ec[t], e0[t], eT[t])
        # (#1 gate) does the TRUE spectrum satisfy A(eta)?  residual <= b_ub
        feas = np.all(A_ub @ a_star <= b_ub + 1e-9)
        feasible += feas

        verdict, h_lo, h_hi = certify_order(
            c[t], a0[t], T[t], RHOS, Kmax, ec[t], e0[t], eT[t], alpha)
        used += 1
        Kbar, insuff_through = certified_Kbar(verdict, Kmax)
        if verdict.get(1) == INSUFFICIENT:
            h1_insuff += 1
            if E_true[1] > alpha:
                h1_insuff_sound += 1
        if verdict.get(1) == SUFFICIENT and E_true[1] > alpha:
            h1_false_suff += 1
        if Kbar == Kst:
            kbar_correct += 1

    dd = max(used, 1)
    print(f"    --- (#1) true-spectrum feasibility rate = {feasible/dd:.3f} "
          f"(target >= {1-delta:.2f}) ---")
    print(f"    K=1 certified INSUFFICIENT rate = {h1_insuff/dd:.3f}  "
          f"(soundness = {h1_insuff_sound/max(h1_insuff,1):.3f}; "
          f"FALSE-sufficient = {h1_false_suff})")
    print(f"    certified Kbar == K* (={Kst}) rate = {kbar_correct/dd:.3f}")


def exp_budget_law(d=49, N0_grid=(2000, 4000, 8000, 16000, 32000),
                   sigma_obs=0.1, delta=0.1, n_trials=80, seed=1):
    """(b) Budget law for the resolvable direction: K=1 INSUFFICIENT rate -> 1 as N0
    grows. Honest Kmax=d, real deg-3 tail. Reports Q per row (#3)."""
    L = len(RHOS)
    Kmax = d
    alpha = 0.05
    energies = {1: 1.0, 2: 0.35, 3: 0.04}
    g = WalshFunction(d, energies, mean=0.5, seed=seed)
    E_true, T_true = true_residual_curve(g.a_true, 3)
    print(f"\n=== (b) BUDGET LAW: K=1 insufficiency vs N0 (honest Kmax=d) "
          f"E_true(1)={E_true[1]:.4f} margin={E_true[1]-alpha:.4f} ===")
    print(f"{'N0':>8} {'Q':>12} {'mean eta_c':>11} {'K=1 insuff':>11} {'false-suff':>11}")
    for N0 in N0_grid:
        c, a0, T, ec, e0, eT = run_pilot_batched(g, RHOS, N0, sigma_obs, n_trials,
                                                 seed=2000, delta=delta)
        insuff = 0; false_suff = 0
        for t in range(n_trials):
            verdict, _, _ = certify_order(c[t], a0[t], T[t], RHOS, Kmax,
                                          ec[t], e0[t], eT[t], alpha)
            insuff += (verdict.get(1) == INSUFFICIENT)
            false_suff += (verdict.get(1) == SUFFICIENT and E_true[1] > alpha)
        Q = pilot_queries(L, N0)
        print(f"{N0:>8} {Q:>12,} {ec.mean():>11.4f} "
              f"{insuff/n_trials:>11.3f} {false_suff:>11d}")
    print(f"    (false-suff must stay 0; insuff -> 1.  Compare Q to dense p2="
          f"{dense_fit_queries(d,2)}, p3={dense_fit_queries(d,3):,}.)")


def exp_oracle_ablation(d=49, N0=16000, sigma_obs=0.1, delta=0.1, n_trials=80, seed=3):
    """(c) ORACLE-SUPPORT ABLATION (appendix-grade, explicitly labelled #2).
    Here Kmax is set to the true support so the MINIMAL K* is recoverable. We sweep
    a deg-3 tail fraction and show: with ceiling=support the pilot finds K*, and the
    soundness bound Kbar(suff) never undershoots K*."""
    L = len(RHOS)
    alpha = 0.05
    print(f"\n=== (c) ORACLE-SUPPORT ABLATION (Kmax = support; not the honest setting) ===")
    print(f"    Tail-fraction sweep; ceiling set to true max degree. (#2: appendix only.)")
    print(f"{'tail_a3':>8} {'K*':>3} {'Kmax':>5} {'Kbar==K*':>9} {'Kbar>=K*':>9} "
          f"{'unresolved':>11} {'Q':>10}")
    for a3 in [0.30, 0.10, 0.04, 0.0]:
        energies = {1: 1.0, 2: 0.35}
        if a3 > 0:
            energies[3] = a3
        support = max(energies)
        Kmax = support
        g = WalshFunction(d, energies, mean=0.5, seed=seed)
        Kst = Kstar(g.a_true, alpha)
        c, a0, T, ec, e0, eT = run_pilot_batched(g, RHOS, N0, sigma_obs, n_trials,
                                                 seed=4000, delta=delta)
        exact = 0; sound = 0; unres = 0
        for t in range(n_trials):
            verdict, _, _ = certify_order(c[t], a0[t], T[t], RHOS, Kmax,
                                          ec[t], e0[t], eT[t], alpha)
            Kbar, _ = certified_Kbar(verdict, Kmax)
            if Kbar is None:
                unres += 1
            else:
                exact += (Kbar == Kst)
                sound += (Kbar >= Kst)
        Q = pilot_queries(L, N0)
        # sound counts only resolved trials; report over resolved
        res = n_trials - unres
        print(f"{a3:>8.3f} {Kst:>3} {Kmax:>5} {exact/n_trials:>9.3f} "
              f"{(sound/max(res,1)):>9.3f} {unres/n_trials:>11.3f} {Q:>10,}")
    print(f"    (Kbar>=K* among resolved == 1.000 => never undershoots; unresolved is "
          f"honest abstention, not error.)")


def exp_independence_pK(N0=16000, sigma_obs=0.1, delta=0.1, n_trials=40, seed=4):
    """(d) Cost vs p_K as d grows, with HONEST accounting (#3). Pilot Q is fixed
    while the dense-fit floor p2 explodes -> the crossover where the pilot wins is
    shown explicitly, instead of asserting 'independent of p_K' without the totals."""
    L = len(RHOS)
    alpha = 0.05
    print(f"\n=== (d) PILOT Q vs DENSE-FIT FLOOR as d grows (fixed N0={N0}) ===")
    print(f"{'d':>5} {'p2':>9} {'p3':>11} {'pilot Q':>10} {'K=1 insuff':>11} "
          f"{'pilot<p3?':>10}")
    Q = pilot_queries(L, N0)
    for d in [25, 49, 100, 196]:
        Kmax = d
        energies = {1: 1.0, 2: 0.35, 3: 0.04}
        g = WalshFunction(d, energies, mean=0.5, seed=seed)
        E_true, _ = true_residual_curve(g.a_true, 3)
        c, a0, T, ec, e0, eT = run_pilot_batched(g, RHOS, N0, sigma_obs, n_trials,
                                                 seed=5000, delta=delta)
        insuff = 0
        for t in range(n_trials):
            verdict, _, _ = certify_order(c[t], a0[t], T[t], RHOS, Kmax,
                                          ec[t], e0[t], eT[t], alpha)
            insuff += (verdict.get(1) == INSUFFICIENT)
        p2 = dense_fit_queries(d, 2)
        p3 = dense_fit_queries(d, 3)
        print(f"{d:>5} {p2:>9,} {p3:>11,} {Q:>10,} {insuff/n_trials:>11.3f} "
              f"{str(Q < p3):>10}")
    print(f"    (Pilot Q is FIXED; p3 grows ~d^3. The pilot only *wins* on cost once "
          f"p_K overtakes Q -- shown, not assumed.)")


if __name__ == "__main__":
    torch.manual_seed(0)
    exp_coverage()
    exp_budget_law()
    exp_oracle_ablation()
    exp_independence_pK()