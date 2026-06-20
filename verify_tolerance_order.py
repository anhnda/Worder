"""
Synthetic verification of the tolerance-order certificate (Sections 3-5),
torch-accelerated pilot + scipy LP certificate.

Changes vs the original numpy script:
  * PILOT is fully vectorized across trials in torch (float32, GPU if available):
    all n_trials pilots for a given (rho-grid, N0) run as ONE batched tensor op.
    Shapes flow as (n_trials, N0, d). The expensive part -- g.eval over big mask
    batches and the coupled-pair correlations -- is where torch helps; the LPs are
    tiny and stay on CPU in scipy.
  * eta is now an EMPIRICAL Bernstein radius measured from std(y*y'), union-bounded
    over the L overlaps + 2 endpoint estimators. The old plug-in c0*(B^2+sigma^2)
    fed worst-case B^2 (~19 here) into eta, giving eta>T and a polytope = whole
    simplex => band width pinned at exactly 1.0000. The empirical radius removes
    that artifact.
  * Kmax is threaded through as an explicit argument. The upper LP can always hide
    energy at the top available degree where rho^Kmax is tiny, so an over-large
    Kmax inflates E_hi(K); the smallest defensible ceiling gives the tightest band.
  * Verdicts read Kbar off the UPPER band E_hi per Eq. (10): Kbar = min{K: E_hi(K)<=a}.

Precision policy (per request): float32 in the pilot tensor work; everything from
c_hat onward (the LP inputs) is float64.

Run it yourself:  python verify_tolerance_order_torch.py
"""

import numpy as np
import torch
from itertools import combinations
from scipy.optimize import linprog

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PILOT_DTYPE = torch.float32   # pilot tensor work
LP_DTYPE = np.float64         # c_hat / a0 / T / LP inputs


# ----------------------------------------------------------------------
# Ground-truth function with an exact Walsh spectrum (torch eval)
# ----------------------------------------------------------------------
class WalshFunction:
    """g(z) = mean + sum_S beta_S * prod_{i in S}(2 z_i - 1).
    Per-degree energy a_j = sum_{|S|=j} beta_S^2 set exactly.
    Stores terms as a ragged list but evaluates in batched torch."""

    def __init__(self, d, energies, mean=0.0, seed=0, device=DEVICE):
        rng = np.random.default_rng(seed)
        self.d = d
        self.mean = float(mean)
        self.device = device
        self.a_true = [mean ** 2]  # a0
        subsets = []   # list of index-lists
        betas = []
        for j, a_j in enumerate(energies, start=1):
            if a_j <= 0:
                self.a_true.append(0.0)
                continue
            all_S = list(combinations(range(d), j))
            k = min(len(all_S), max(1, min(8, len(all_S))))
            idx = rng.choice(len(all_S), size=k, replace=False)
            chosen = [all_S[i] for i in idx]
            raw = rng.standard_normal(k)
            raw = raw / np.sqrt(np.sum(raw ** 2))
            b = raw * np.sqrt(a_j)
            for S, bb in zip(chosen, b):
                subsets.append(list(S))
                betas.append(float(bb))
            self.a_true.append(float(a_j))
        self.a_true = np.array(self.a_true)

        # Build a (T_terms, d) {0,1} membership mask and a beta vector for batched eval.
        self.n_terms = len(subsets)
        if self.n_terms > 0:
            M = torch.zeros((self.n_terms, d), dtype=PILOT_DTYPE, device=device)
            for t, S in enumerate(subsets):
                if S:
                    M[t, S] = 1.0
            self.member = M                          # (n_terms, d)
            self.beta = torch.tensor(betas, dtype=PILOT_DTYPE, device=device)  # (n_terms,)
        else:
            self.member = None
            self.beta = None

    @torch.no_grad()
    def eval(self, Z):
        """Z: (..., d) float tensor in {0,1}. Returns (...) tensor.
        chi_S(z) = prod_{i in S}(2 z_i - 1). For each term we need the product over
        its members. We compute it as exp(sum_{i in S} log|s_i|) * sign, but s_i = +-1
        so |s_i| = 1 and the product is just the parity of (-1) factors among members.
        => prod = prod_i (s_i)^{member} = exp over members of log(s_i) is undefined for
        s_i=-1; instead use: signed = (2z-1); term_prod = prod_i signed_i^{member_i}.
        Implement via: log not safe -> use cumulative product through masking:
        term_prod = prod over members of signed; do it as
        prod = exp( member @ 0 ) adjusted -> we use the identity
        signed^member = 1 - member*(1 - signed) only works for member in {0,1} and
        signed in {-1,+1}: 1 - member*(1-signed) = signed if member=1 else 1.  Exactly. """
        signed = (2.0 * Z - 1.0).to(PILOT_DTYPE)     # (..., d), entries +-1
        out = torch.full(Z.shape[:-1], self.mean, dtype=PILOT_DTYPE, device=Z.device)
        if self.member is None:
            return out
        # For each term t: factor_i = signed_i if member_{t,i}=1 else 1.
        # factor = 1 - member*(1 - signed). Then prod over i (last dim).
        # Z: (..., d) ; member: (n_terms, d). Broadcast over a new term axis.
        signed_e = signed.unsqueeze(-2)              # (..., 1, d)
        member_e = self.member                       # (n_terms, d)
        factor = 1.0 - member_e * (1.0 - signed_e)   # (..., n_terms, d)
        chi = factor.prod(dim=-1)                     # (..., n_terms)
        contrib = chi * self.beta                     # (..., n_terms)
        out = out + contrib.sum(dim=-1)
        return out


# ----------------------------------------------------------------------
# Vectorized pilot across trials (one tensor op per overlap)
# ----------------------------------------------------------------------
@torch.no_grad()
def _coupled(d, rho, shape_lead, gen, device):
    """Return z, z2 each of shape (*shape_lead, d), z2 a rho-coupled copy of z."""
    z = torch.randint(0, 2, (*shape_lead, d), generator=gen, device=device,
                      dtype=PILOT_DTYPE)
    keep = torch.rand((*shape_lead, d), generator=gen, device=device) < rho
    fresh = torch.randint(0, 2, (*shape_lead, d), generator=gen, device=device,
                          dtype=PILOT_DTYPE)
    z2 = torch.where(keep, z, fresh)
    return z, z2


@torch.no_grad()
def run_pilot_batched(g, rhos, N0, sigma_obs, n_trials, seed, delta):
    """Run n_trials pilots at once. Returns numpy arrays (float64):
       c_hat (n_trials, L), a0_hat (n_trials,), T_hat (n_trials,), eta (n_trials,).
    eta is the per-trial empirical Bernstein radius (max over estimators)."""
    device = g.device
    gen = torch.Generator(device=device).manual_seed(int(seed))
    L = len(rhos)
    lead = (n_trials, N0)

    c_hat = torch.empty((n_trials, L), dtype=PILOT_DTYPE, device=device)
    c_std = torch.empty((n_trials, L), dtype=PILOT_DTYPE, device=device)
    for l, rho in enumerate(rhos):
        z, z2 = _coupled(g.d, float(rho), lead, gen, device)
        y = g.eval(z) + sigma_obs * torch.randn(lead, generator=gen, device=device)
        yp = g.eval(z2) + sigma_obs * torch.randn(lead, generator=gen, device=device)
        prod = y * yp                                  # (n_trials, N0)
        c_hat[:, l] = prod.mean(dim=1)
        c_std[:, l] = prod.std(dim=1)

    # c(1) via paired re-query of SAME mask under independent noise
    z, _ = _coupled(g.d, 1.0, lead, gen, device)
    gz = g.eval(z)
    y1 = gz + sigma_obs * torch.randn(lead, generator=gen, device=device)
    y2 = gz + sigma_obs * torch.randn(lead, generator=gen, device=device)
    prod1 = y1 * y2
    c1_hat = prod1.mean(dim=1)
    c1_std = prod1.std(dim=1)

    # a0 = (E[g])^2, debias the (1/N0) sigma^2 inflation
    zm, _ = _coupled(g.d, 1.0, lead, gen, device)
    ym = g.eval(zm) + sigma_obs * torch.randn(lead, generator=gen, device=device)
    mean_est = ym.mean(dim=1)
    a0_hat = torch.clamp(mean_est ** 2 - (sigma_obs ** 2) / N0, min=0.0)
    a0_std = ym.std(dim=1)  # std of single query; mean concentrates at a0_std/sqrt(N0)

    T_hat = torch.clamp(c1_hat - a0_hat, min=1e-12)

    # Empirical Bernstein radius: z_score * std / sqrt(N0), union bound over L+2 terms.
    zsc = float(np.sqrt(2.0 * np.log(2.0 * (L + 2) / delta)))
    rad_curve = zsc * c_std / np.sqrt(N0)              # (n_trials, L)
    rad_c1 = zsc * c1_std / np.sqrt(N0)                # (n_trials,)
    # a0 enters as (mean)^2; radius on a0 ~ 2|mean|*(a0_std/sqrt(N0))
    rad_a0 = zsc * 2.0 * mean_est.abs() * a0_std / np.sqrt(N0)
    eta = torch.maximum(rad_curve.max(dim=1).values,
                        torch.maximum(rad_c1, rad_a0))  # (n_trials,)

    to64 = lambda t: t.detach().to("cpu").double().numpy()
    return to64(c_hat), to64(a0_hat), to64(T_hat), to64(eta)


# ----------------------------------------------------------------------
# Certificate by feasibility (scipy LPs, CPU, float64)
# ----------------------------------------------------------------------
def certified_band(c_hat, a0_hat, T_hat, rhos, Kmax, eta):
    """LP pair (9) per K. Returns E_lo[0..Kmax], E_hi[0..Kmax] (relative residuals)."""
    nvar = Kmax + 1
    rhos = np.asarray(rhos, dtype=LP_DTYPE)
    powers = np.array([[rho ** j for j in range(nvar)] for rho in rhos], dtype=LP_DTYPE)

    A_eq = np.zeros((2, nvar), dtype=LP_DTYPE)
    A_eq[0, 0] = 1.0
    A_eq[1, 1:] = 1.0
    b_eq = np.array([a0_hat, T_hat], dtype=LP_DTYPE)

    A_ub = np.vstack([powers, -powers])
    b_ub = np.concatenate([c_hat + eta, -(c_hat - eta)])

    bounds = [(0, None)] * nvar
    E_lo = np.empty(Kmax + 1); E_hi = np.empty(Kmax + 1)
    for K in range(0, Kmax + 1):
        c_obj = np.zeros(nvar, dtype=LP_DTYPE); c_obj[K + 1:] = 1.0
        r_min = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method="highs")
        r_max = linprog(-c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                        bounds=bounds, method="highs")
        if not (r_min.success and r_max.success):
            E_lo[K], E_hi[K] = np.nan, np.nan
        else:
            E_lo[K] = r_min.fun / T_hat
            E_hi[K] = (-r_max.fun) / T_hat
    return E_lo, E_hi


def true_residual_curve(a_true, Kmax):
    a = a_true.copy(); T = a[1:].sum()
    E = np.empty(Kmax + 1)
    for K in range(0, Kmax + 1):
        E[K] = a[K + 1:].sum() / T if T > 0 else 0.0
    return E


def Kstar(E_true, alpha):
    for K in range(1, len(E_true)):
        if E_true[K] <= alpha:
            return K
    return len(E_true) - 1


# ----------------------------------------------------------------------
# Experiments (verdicts read off E_hi per Eq. 10)
# ----------------------------------------------------------------------
def exp_coverage(n_trials=200, alpha_list=(0.05, 0.01), d=49,
                 top_frac=0.04, N0=4000, sigma_obs=0.1, delta=0.1, seed=1,
                 Kmax=3):
    print(f"\n=== (a) COVERAGE  d={d} N0={N0} sigma={sigma_obs} delta={delta} "
          f"Kmax={Kmax} dev={DEVICE} ===")
    energies = [1.0, 0.35, top_frac]
    g = WalshFunction(d, energies, mean=0.5, seed=seed)
    E_true = true_residual_curve(g.a_true, Kmax)
    Kst = {a: Kstar(E_true, a) for a in alpha_list}
    print(f"true per-degree energies a_1.. = {np.round(g.a_true[1:4],4)}")
    print(f"true residual curve E(1..3)    = {np.round(E_true[1:4],4)}")
    print(f"K*_alpha: " + ", ".join(f"{a:.0%}->{Kst[a]}" for a in alpha_list))

    rhos = np.linspace(0.5, 0.97, 8)
    c_hat, a0_hat, T_hat, eta = run_pilot_batched(
        g, rhos, N0, sigma_obs, n_trials, seed=1000, delta=delta)

    covered = 0; kbar_hits = {a: 0 for a in alpha_list}; widths = []; used = 0
    for t in range(n_trials):
        E_lo, E_hi = certified_band(c_hat[t], a0_hat[t], T_hat[t], rhos, Kmax, eta[t])
        if np.any(np.isnan(E_lo)):
            continue
        used += 1
        ok = np.all((E_true[1:] >= E_lo[1:] - 1e-9) &
                    (E_true[1:] <= E_hi[1:] + 1e-9))
        covered += ok
        widths.append(np.max(E_hi[1:Kmax+1] - E_lo[1:Kmax+1]))
        for a in alpha_list:
            kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= a), Kmax)
            if kbar == Kst[a]:
                kbar_hits[a] += 1

    denom = max(used, 1)
    print(f"mean empirical eta             = {eta.mean():.4f}  (eta/T~{eta.mean()/T_hat.mean():.3f})")
    print(f"empirical coverage (all K)     = {covered/denom:.3f}  (target >= {1-delta:.2f})")
    print(f"mean band width (K=1..{Kmax})       = {np.mean(widths):.4f}")
    for a in alpha_list:
        print(f"  Kbar==K* rate @alpha={a:.0%}     = {kbar_hits[a]/denom:.3f}")


def exp_exactness_vs_margin(d=49, N0=8000, sigma_obs=0.1, delta=0.1, seed=2,
                            Kmax=3, n_trials=1):
    print(f"\n=== (b) EXACTNESS vs MARGIN (sweep top-order energy) d={d} Kmax={Kmax} ===")
    rhos = np.linspace(0.5, 0.97, 8)
    alpha = 0.05
    print(f"{'top_frac':>9} {'E(2)true':>9} {'margin':>8} "
          f"{'K*':>3} {'Kbar':>5} {'bandW(2)':>9} {'exact?':>7}")
    for top_frac in [0.20, 0.10, 0.06, 0.045, 0.03]:
        energies = [1.0, 0.35, top_frac]
        g = WalshFunction(d, energies, mean=0.5, seed=seed)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        margin = abs(E_true[2] - alpha)
        c_hat, a0_hat, T_hat, eta = run_pilot_batched(
            g, rhos, N0, sigma_obs, n_trials, seed=7777, delta=delta)
        E_lo, E_hi = certified_band(c_hat[0], a0_hat[0], T_hat[0], rhos, Kmax, eta[0])
        kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
        bw2 = E_hi[2] - E_lo[2]
        print(f"{top_frac:>9.3f} {E_true[2]:>9.4f} {margin:>8.4f} "
              f"{Kst:>3} {kbar:>5} {bw2:>9.4f} {str(kbar==Kst):>7}")


def exp_budget_law(d=49, sigma_obs=0.1, delta=0.1, Kmax=3, n_trials=40):
    print(f"\n=== (c1) BUDGET LAW: N0 vs margin (resolve K*) d={d} Kmax={Kmax} ===")
    rhos = np.linspace(0.5, 0.97, 8)
    alpha = 0.05
    print(f"{'margin':>8} {'N0_needed':>10} {'N0*margin^2':>12}")
    for top_frac in [0.12, 0.08, 0.055]:
        energies = [1.0, 0.35, top_frac]
        g = WalshFunction(d, energies, mean=0.5, seed=3)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        margin = abs(E_true[2] - alpha)
        N0_needed = None
        for N0 in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            c_hat, a0_hat, T_hat, eta = run_pilot_batched(
                g, rhos, N0, sigma_obs, n_trials, seed=20000, delta=delta)
            hits = 0
            for t in range(n_trials):
                E_lo, E_hi = certified_band(c_hat[t], a0_hat[t], T_hat[t], rhos, Kmax, eta[t])
                kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
                hits += (kbar == Kst)
            if hits / n_trials >= 0.9:
                N0_needed = N0
                break
        if N0_needed:
            print(f"{margin:>8.4f} {N0_needed:>10} {N0_needed*margin**2:>12.2f}")
        else:
            print(f"{margin:>8.4f} {'>64000':>10} {'--':>12}")


def exp_independence_pK(N0=4000, sigma_obs=0.1, delta=0.1, Kmax=3, n_trials=1):
    print(f"\n=== (c2) COST INDEPENDENT of p_K as d grows (fixed N0) Kmax={Kmax} ===")
    rhos = np.linspace(0.5, 0.97, 8)
    alpha = 0.05
    print(f"{'d':>4} {'p_3=C(d,<=3)':>12} {'Kbar':>5} {'K*':>3} {'bandW(2)':>9}")
    for d in [25, 49, 100, 196]:
        energies = [1.0, 0.35, 0.04]
        g = WalshFunction(d, energies, mean=0.5, seed=4)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        c_hat, a0_hat, T_hat, eta = run_pilot_batched(
            g, rhos, N0, sigma_obs, n_trials, seed=33333, delta=delta)
        E_lo, E_hi = certified_band(c_hat[0], a0_hat[0], T_hat[0], rhos, Kmax, eta[0])
        kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
        from math import comb
        pK = sum(comb(d, k) for k in range(1, 4))
        print(f"{d:>4} {pK:>12} {kbar:>5} {Kst:>3} {E_hi[2]-E_lo[2]:>9.4f}")
    print("  (N0 fixed across d while p_3 grows ~ d^3 => pilot cost decoupled from p_K)")


if __name__ == "__main__":
    torch.manual_seed(0)
    exp_coverage()
    exp_exactness_vs_margin()
    exp_budget_law()
    exp_independence_pK()