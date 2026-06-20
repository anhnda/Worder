"""
Synthetic verification of the tolerance-order certificate (Sections 3-5).

We never touch a real model. We *construct* a ground-truth Walsh spectrum {a_j}
directly, which fixes K*_alpha exactly. Then we simulate the pilot estimator:
  - draw rho-coupled mask pairs, query a function whose per-degree energies are a_j,
  - form noisy correlations c_hat(rho_l),
  - measure c(1) and a0 by paired re-query / centered mean,
  - run the two LP families (9) to get the certified band [E_lo(K), E_hi(K)],
  - read off Kbar_alpha.

To make per-degree energies controllable, we realize the spectrum on a small set
of d units with an *explicit* Walsh function g(z) = sum_S beta_S chi_S(z), where
within each degree j we place the energy a_j on randomly chosen subsets S, |S|=j,
with random signs and magnitudes scaled to hit a_j exactly. This g has EXACT
per-degree energies a_j, so K*_alpha and the true residual curve E(K) are known.

Claims checked:
  (a) coverage: P(true E(K) in [E_lo,E_hi] for all K) >= 1 - delta
  (b) exactness: Kbar_alpha == K*_alpha when band width < margin
  (c) budget law: N0 ~ 1/Delta_alpha^2, and pilot cost independent of p_K as d grows
"""

import numpy as np
from itertools import combinations
from scipy.optimize import linprog

rng_global = np.random.default_rng(0)


# ----------------------------------------------------------------------
# Ground-truth function with an exact Walsh spectrum
# ----------------------------------------------------------------------
class WalshFunction:
    """g(z) = sum_S beta_S chi_S(z), chi_S(z) = prod_{i in S} (2 z_i - 1)/... 
    using centered chars: chi_S(z) = prod_{i in S} (2*zbar_i), zbar = z - 1/2 => +-1.
    Per-degree energy a_j = sum_{|S|=j} beta_S^2 is set exactly."""

    def __init__(self, d, energies, mean=0.0, seed=0):
        """energies: list a_1..a_K (degree>=1). mean sets a0 = mean^2 (beta_empty=mean)."""
        rng = np.random.default_rng(seed)
        self.d = d
        self.mean = mean
        self.terms = []  # list of (frozenset S, beta_S)
        self.a_true = [mean ** 2]  # a0
        for j, a_j in enumerate(energies, start=1):
            if a_j <= 0:
                self.a_true.append(0.0)
                continue
            all_S = list(combinations(range(d), j))
            # pick a handful of subsets to carry this degree's energy
            k = min(len(all_S), max(1, min(8, len(all_S))))
            idx = rng.choice(len(all_S), size=k, replace=False)
            chosen = [all_S[i] for i in idx]
            raw = rng.standard_normal(k)
            raw = raw / np.sqrt(np.sum(raw ** 2))  # unit-norm coefficient vector
            betas = raw * np.sqrt(a_j)              # now sum betas^2 = a_j exactly
            for S, b in zip(chosen, betas):
                self.terms.append((frozenset(S), float(b)))
            self.a_true.append(float(a_j))
        self.a_true = np.array(self.a_true)

    def eval(self, Z):
        """Z: (n, d) binary array -> g values (n,)."""
        zc = 2.0 * Z - 1.0  # +-1
        out = np.full(Z.shape[0], self.mean)
        for S, b in self.terms:
            if len(S) == 0:
                continue
            idx = list(S)
            out = out + b * np.prod(zc[:, idx], axis=1)
        return out


# ----------------------------------------------------------------------
# Pilot estimator
# ----------------------------------------------------------------------
def sample_coupled(d, rho, n, rng):
    z = rng.integers(0, 2, size=(n, d))
    keep = rng.random(size=(n, d)) < rho
    z2 = np.where(keep, z, rng.integers(0, 2, size=(n, d)))
    return z, z2


def run_pilot(g, rhos, N0, sigma_obs, rng):
    """Return c_hat(rho_l), c1_hat (= E[g^2]), a0_hat, T_hat."""
    c_hat = np.empty(len(rhos))
    for l, rho in enumerate(rhos):
        z, z2 = sample_coupled(g.d, rho, N0, rng)
        y = g.eval(z) + sigma_obs * rng.standard_normal(N0)
        yp = g.eval(z2) + sigma_obs * rng.standard_normal(N0)
        c_hat[l] = np.mean(y * yp)

    # c(1) = E[g^2] via paired re-query of the SAME mask under independent noise
    z, _ = sample_coupled(g.d, 1.0, N0, rng)
    gz = g.eval(z)
    y1 = gz + sigma_obs * rng.standard_normal(N0)
    y2 = gz + sigma_obs * rng.standard_normal(N0)
    c1_hat = np.mean(y1 * y2)

    # a0 = (E[g])^2, debias the (1/N0) sigma^2 inflation of the squared sample mean
    zm, _ = sample_coupled(g.d, 1.0, N0, rng)
    ym = g.eval(zm) + sigma_obs * rng.standard_normal(N0)
    mean_est = np.mean(ym)
    a0_hat = mean_est ** 2 - (sigma_obs ** 2) / N0
    a0_hat = max(a0_hat, 0.0)

    T_hat = max(c1_hat - a0_hat, 1e-12)
    return c_hat, c1_hat, a0_hat, T_hat


def certified_band(c_hat, a0_hat, T_hat, rhos, Kmax, eta):
    """Solve the LP pair (9) for each K, returning E_lo[K], E_hi[K], K=1..Kmax.
    Variables: a = (a_0,...,a_Kmax) >= 0.
    Constraints:
      a_0 = a0_hat
      sum_{j>=1} a_j = T_hat
      | sum_j a_j rho_l^j - c_hat_l | <= eta   for all l
    Objective for residual at K: minimize/maximize sum_{j>K} a_j, divide by T_hat.
    """
    nvar = Kmax + 1
    powers = np.array([[rho ** j for j in range(nvar)] for rho in rhos])  # (L, nvar)

    # Equality: a0 fixed, sum_{j>=1} a_j = T_hat
    A_eq = np.zeros((2, nvar))
    A_eq[0, 0] = 1.0
    b_eq = np.array([a0_hat, T_hat])
    A_eq[1, 1:] = 1.0

    # Inequalities for the curve band: powers @ a <= c+eta ; -(powers@a) <= -(c-eta)
    A_ub = np.vstack([powers, -powers])
    b_ub = np.concatenate([c_hat + eta, -(c_hat - eta)])

    bounds = [(0, None)] * nvar

    E_lo = np.empty(Kmax + 1)
    E_hi = np.empty(Kmax + 1)
    for K in range(0, Kmax + 1):
        c_obj = np.zeros(nvar)
        c_obj[K + 1:] = 1.0  # sum_{j>K} a_j
        # min
        r_min = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method="highs")
        # max  (minimize -obj)
        r_max = linprog(-c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method="highs")
        if not (r_min.success and r_max.success):
            E_lo[K], E_hi[K] = np.nan, np.nan
        else:
            E_lo[K] = r_min.fun / T_hat
            E_hi[K] = (-r_max.fun) / T_hat
    return E_lo, E_hi


def true_residual_curve(a_true, Kmax):
    a = a_true.copy()
    T = a[1:].sum()
    E = np.empty(Kmax + 1)
    for K in range(0, Kmax + 1):
        E[K] = a[K + 1:].sum() / T if T > 0 else 0.0
    return E


def Kstar(E_true, alpha):
    for K in range(1, len(E_true)):
        if E_true[K] <= alpha:
            return K
    return len(E_true) - 1


def eta_radius(B, sigma_obs, L, N0, delta, c0=1.0):
    return c0 * (B ** 2 + sigma_obs ** 2) * np.sqrt(np.log(2 * (L + 1) / delta) / N0)


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------
def make_spectrum(d, top_frac, seed):
    """A1-main + a2-pairwise + small higher-order block. Returns energies a_1..a_3."""
    a1 = 1.0
    a2 = 0.35
    a3 = top_frac  # tunable high-order tail
    return [a1, a2, a3], d, seed


def estimate_B(g, rng, n=20000):
    z, _ = sample_coupled(g.d, 1.0, n, rng)
    return float(np.max(np.abs(g.eval(z)))) + abs(g.mean)


def exp_coverage(n_trials=200, alpha_list=(0.05, 0.01), d=49,
                 top_frac=0.04, N0=4000, sigma_obs=0.1, delta=0.1, seed=1):
    print(f"\n=== (a) COVERAGE  d={d} N0={N0} sigma={sigma_obs} delta={delta} ===")
    energies, d, sseed = make_spectrum(d, top_frac, seed)
    g = WalshFunction(d, energies, mean=0.5, seed=sseed)
    rng = np.random.default_rng(seed)
    B = estimate_B(g, rng)
    rhos = np.linspace(0.2, 0.9, 6)
    Kmax = 6
    E_true = true_residual_curve(g.a_true, Kmax)
    Kst = {a: Kstar(E_true, a) for a in alpha_list}
    print(f"true per-degree energies a_1..  = "
          f"{np.round(g.a_true[1:len(energies)+1],4)}")
    print(f"true residual curve E(1..3)     = {np.round(E_true[1:4],4)}")
    print(f"K*_alpha: " + ", ".join(f"{a:.0%}->{Kst[a]}" for a in alpha_list))

    covered = 0
    kbar_hits = {a: 0 for a in alpha_list}
    band_widths = []
    for t in range(n_trials):
        r = np.random.default_rng(1000 + t)
        c_hat, c1, a0_hat, T_hat = run_pilot(g, rhos, N0, sigma_obs, r)
        eta = eta_radius(B, sigma_obs, len(rhos), N0, delta, c0=1.0)
        E_lo, E_hi = certified_band(c_hat, a0_hat, T_hat, rhos, Kmax, eta)
        if np.any(np.isnan(E_lo)):
            continue
        ok = np.all((E_true[1:] >= E_lo[1:] - 1e-9) &
                    (E_true[1:] <= E_hi[1:] + 1e-9))
        covered += ok
        band_widths.append(np.max(E_hi[1:4] - E_lo[1:4]))
        for a in alpha_list:
            # Kbar = smallest K with E_hi(K) <= alpha
            kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= a), Kmax)
            if kbar == Kst[a]:
                kbar_hits[a] += 1

    print(f"empirical coverage (all K)      = {covered/n_trials:.3f}  "
          f"(target >= {1-delta:.2f})")
    print(f"mean band width (K=1..3)        = {np.mean(band_widths):.4f}")
    for a in alpha_list:
        print(f"  Kbar==K* rate @alpha={a:.0%}      = {kbar_hits[a]/n_trials:.3f}")


def exp_exactness_vs_margin(d=49, N0=8000, sigma_obs=0.1, delta=0.1, seed=2):
    print(f"\n=== (b) EXACTNESS vs MARGIN (sweep top-order energy) d={d} ===")
    rhos = np.linspace(0.2, 0.9, 6)
    Kmax = 6
    alpha = 0.05
    print(f"{'top_frac':>9} {'E(2)true':>9} {'margin':>8} "
          f"{'K*':>3} {'Kbar':>5} {'bandW(2)':>9} {'exact?':>7}")
    for top_frac in [0.20, 0.10, 0.06, 0.045, 0.03]:
        energies = [1.0, 0.35, top_frac]
        g = WalshFunction(d, energies, mean=0.5, seed=seed)
        rng = np.random.default_rng(seed)
        B = estimate_B(g, rng)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        margin = abs(E_true[2] - alpha)  # near the crossing
        r = np.random.default_rng(7777)
        c_hat, c1, a0_hat, T_hat = run_pilot(g, rhos, N0, sigma_obs, r)
        eta = eta_radius(B, sigma_obs, len(rhos), N0, delta, c0=1.0)
        E_lo, E_hi = certified_band(c_hat, a0_hat, T_hat, rhos, Kmax, eta)
        kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
        bw2 = E_hi[2] - E_lo[2]
        exact = (kbar == Kst)
        print(f"{top_frac:>9.3f} {E_true[2]:>9.4f} {margin:>8.4f} "
              f"{Kst:>3} {kbar:>5} {bw2:>9.4f} {str(exact):>7}")


def exp_budget_law(d=49, sigma_obs=0.1, delta=0.1, seed=3, n_trials=40):
    print(f"\n=== (c1) BUDGET LAW: N0 vs margin (resolve K*) d={d} ===")
    rhos = np.linspace(0.2, 0.9, 6)
    Kmax = 6
    alpha = 0.05
    print(f"{'margin':>8} {'N0_needed':>10} {'N0*margin^2':>12}")
    for top_frac in [0.12, 0.08, 0.055]:
        energies = [1.0, 0.35, top_frac]
        g = WalshFunction(d, energies, mean=0.5, seed=seed)
        rng = np.random.default_rng(seed)
        B = estimate_B(g, rng)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        margin = abs(E_true[2] - alpha)
        # find smallest N0 (grid) achieving Kbar==K* in >=90% of trials
        N0_needed = None
        for N0 in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            hits = 0
            for t in range(n_trials):
                r = np.random.default_rng(20000 + t)
                c_hat, c1, a0_hat, T_hat = run_pilot(g, rhos, N0, sigma_obs, r)
                eta = eta_radius(B, sigma_obs, len(rhos), N0, delta, c0=1.0)
                E_lo, E_hi = certified_band(c_hat, a0_hat, T_hat, rhos, Kmax, eta)
                kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
                hits += (kbar == Kst)
            if hits / n_trials >= 0.9:
                N0_needed = N0
                break
        if N0_needed:
            print(f"{margin:>8.4f} {N0_needed:>10} {N0_needed*margin**2:>12.2f}")
        else:
            print(f"{margin:>8.4f} {'>64000':>10} {'--':>12}")


def exp_independence_pK(N0=4000, sigma_obs=0.1, delta=0.1, seed=4):
    print(f"\n=== (c2) COST INDEPENDENT of p_K as d grows (fixed N0) ===")
    rhos = np.linspace(0.2, 0.9, 6)
    Kmax = 6
    alpha = 0.05
    print(f"{'d':>4} {'p_3=C(d,<=3)':>12} {'Kbar':>5} {'K*':>3} {'bandW(2)':>9}")
    for d in [25, 49, 100, 196]:
        energies = [1.0, 0.35, 0.04]
        g = WalshFunction(d, energies, mean=0.5, seed=seed)
        rng = np.random.default_rng(seed)
        B = estimate_B(g, rng)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        r = np.random.default_rng(33333)
        c_hat, c1, a0_hat, T_hat = run_pilot(g, rhos, N0, sigma_obs, r)
        eta = eta_radius(B, sigma_obs, len(rhos), N0, delta, c0=1.0)
        E_lo, E_hi = certified_band(c_hat, a0_hat, T_hat, rhos, Kmax, eta)
        kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
        pK = sum(int(np.math.comb(d, k)) for k in range(1, 4))
        print(f"{d:>4} {pK:>12} {kbar:>5} {Kst:>3} {E_hi[2]-E_lo[2]:>9.4f}")
    print("  (N0 fixed across d while p_3 grows ~ d^3 => pilot cost decoupled from p_K)")


if __name__ == "__main__":
    exp_coverage()
    exp_exactness_vs_margin()
    exp_budget_law()
    exp_independence_pK()