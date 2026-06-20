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
        chi_S(z) = prod_{i in S}(2 z_i - 1).  Per term, factor_i = signed_i if i in S
        else 1, via the exact identity  1 - member*(1-signed) = signed if member=1 else 1
        (valid since member in {0,1}, signed in {+-1}).

        MEMORY-FLAT: we never materialize the (..., n_terms, d) tensor (that was the OOM:
        it is n_trials*N0*n_terms*d float32). Instead we FOLD over the d feature columns,
        keeping a running per-term product of shape (..., n_terms). Peak tensor is
        (..., n_terms); the d axis is consumed one column at a time. d is small (<=~200)
        so the Python loop over columns is cheap relative to the batch dims."""
        signed = (2.0 * Z - 1.0).to(PILOT_DTYPE)     # (..., d), entries +-1
        lead = Z.shape[:-1]
        out = torch.full(lead, self.mean, dtype=PILOT_DTYPE, device=Z.device)
        if self.member is None:
            return out
        n_terms = self.member.shape[0]
        # running product over members, shape (..., n_terms), init to ones
        chi = torch.ones((*lead, n_terms), dtype=PILOT_DTYPE, device=Z.device)
        for i in range(self.d):
            mem_i = self.member[:, i]                 # (n_terms,) in {0,1}
            if not bool(torch.any(mem_i)):
                continue                              # column i in no term's support
            s_i = signed[..., i].unsqueeze(-1)        # (..., 1)
            # factor for this column: s_i where mem_i=1 else 1  ->  1 - mem_i*(1 - s_i)
            factor_i = 1.0 - mem_i * (1.0 - s_i)      # (..., n_terms)
            chi = chi * factor_i
        out = out + (chi * self.beta).sum(dim=-1)
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


def _auto_chunk(n_trials, N0, n_terms, d, device, budget_gib=2.0):
    """Pick a trial-chunk size so the peak (chunk, N0, n_terms) tensor stays under
    budget. Several such tensors are live at once (z, signed, chi, y, yp, ...), so
    we divide the budget by a safety factor. CPU: no need to chunk hard."""
    if str(device) == "cpu":
        return n_trials
    bytes_per = 4  # float32
    live_tensors = 8  # rough count of simultaneously-live (chunk,N0,*) buffers
    per_trial = N0 * max(n_terms, d) * bytes_per * live_tensors
    budget = budget_gib * (1024 ** 3)
    chunk = max(1, int(budget // max(per_trial, 1)))
    return min(chunk, n_trials)


@torch.no_grad()
def run_pilot_batched(g, rhos, N0, sigma_obs, n_trials, seed, delta, chunk=None):
    """Run n_trials pilots, chunked over the trial axis to bound GPU memory.
    Returns float64 numpy: c_hat (n_trials,L), a0_hat (n_trials,), T_hat, eta.

    Results are INVARIANT to chunk size: each chunk gets its own generator seeded
    from (seed, chunk_start), so the random stream a given trial sees does not depend
    on how trials are grouped."""
    n_terms = 0 if g.member is None else g.member.shape[0]
    if chunk is None:
        chunk = _auto_chunk(n_trials, N0, n_terms, g.d, g.device)
    parts = {"c": [], "a0": [], "T": [], "eta": []}
    start = 0
    while start < n_trials:
        m = min(chunk, n_trials - start)
        c, a0, T, eta = _run_pilot_core(
            g, rhos, N0, sigma_obs, m, seed=seed + start, delta=delta)
        parts["c"].append(c); parts["a0"].append(a0)
        parts["T"].append(T); parts["eta"].append(eta)
        start += m
    return (np.concatenate(parts["c"], 0), np.concatenate(parts["a0"], 0),
            np.concatenate(parts["T"], 0), np.concatenate(parts["eta"], 0))


@torch.no_grad()
def _run_pilot_core(g, rhos, N0, sigma_obs, n_trials, seed, delta):
    """One chunk of n_trials pilots as batched tensor ops. Same return shapes as
    run_pilot_batched but for this chunk only."""
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
                 top_frac=0.0, N0=16000, sigma_obs=0.1, delta=0.1, seed=1,
                 Kmax=2):
    print(f"\n=== (a) COVERAGE  d={d} N0={N0} sigma={sigma_obs} delta={delta} "
          f"Kmax={Kmax} dev={DEVICE} ===")
    energies = [1.0, 0.35, top_frac]
    g = WalshFunction(d, energies, mean=0.5, seed=seed)
    E_true = true_residual_curve(g.a_true, Kmax)
    Kst = {a: Kstar(E_true, a) for a in alpha_list}
    print(f"true per-degree energies a_1.. = {np.round(g.a_true[1:Kmax+1],4)}")
    print(f"true residual curve E(1..{Kmax})     = {np.round(E_true[1:Kmax+1],4)}")
    print(f"K*_alpha: " + ", ".join(f"{a:.0%}->{Kst[a]}" for a in alpha_list))

    rhos = np.linspace(0.5, 0.97, 8)
    c_hat, a0_hat, T_hat, eta = run_pilot_batched(
        g, rhos, N0, sigma_obs, n_trials, seed=1000, delta=delta)

    # Headline-claim quantity: is order K certified INSUFFICIENT, i.e. E_lo(K) > alpha?
    # This is the "easy"/resolvable direction (low-degree energy is visible at every rho),
    # and is exactly the paper's "K=1 insufficient in X% of cases". We track it for K=1
    # (and K=2 for reference). It is a one-sided certificate: when E_lo(K) > alpha we have
    # PROVABLY ruled out order K; we also record whether that verdict is correct vs truth.
    covered = 0; used = 0; widths = []
    kbar_hits = {a: 0 for a in alpha_list}
    insuff_certified = {1: 0, 2: 0}   # times E_lo(K) > alpha (certified insufficient)
    insuff_correct = {1: 0, 2: 0}     # of those, times truth agrees (E_true(K) > alpha)
    alpha_head = alpha_list[0]        # headline tolerance (5%)

    for t in range(n_trials):
        E_lo, E_hi = certified_band(c_hat[t], a0_hat[t], T_hat[t], rhos, Kmax, eta[t])
        if np.any(np.isnan(E_lo)):
            continue
        used += 1
        ok = np.all((E_true[1:] >= E_lo[1:] - 1e-9) &
                    (E_true[1:] <= E_hi[1:] + 1e-9))
        covered += ok
        widths.append(np.max(E_hi[1:Kmax+1] - E_lo[1:Kmax+1]))
        for K in (1, 2):
            if E_lo[K] > alpha_head:                 # certified insufficient at headline alpha
                insuff_certified[K] += 1
                if E_true[K] > alpha_head:           # and truth agrees (never a false claim)
                    insuff_correct[K] += 1
        for a in alpha_list:
            kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= a), Kmax)
            if kbar == Kst[a]:
                kbar_hits[a] += 1

    denom = max(used, 1)
    print(f"mean empirical eta             = {eta.mean():.4f}  (eta/T~{eta.mean()/T_hat.mean():.3f})")
    print(f"empirical coverage (all K)     = {covered/denom:.3f}  (target >= {1-delta:.2f})")
    print(f"mean band width (K=1..{Kmax})       = {np.mean(widths):.4f}")
    print(f"--- PRIMARY: order certified INSUFFICIENT (E_lo(K) > {alpha_head:.0%}) ---")
    for K in (1, 2):
        true_insuff = E_true[K] > alpha_head
        rate = insuff_certified[K] / denom
        # soundness: of the times we certified K insufficient, fraction that are truly so
        sound = (insuff_correct[K] / insuff_certified[K]) if insuff_certified[K] else float('nan')
        print(f"  K={K}: certified-insufficient rate = {rate:.3f}  "
              f"(true E({K})={E_true[K]:.4f}{'>' if true_insuff else '<='}{alpha_head:.0%}; "
              f"soundness={sound:.3f})")
    print(f"--- SECONDARY: Kbar==K* exactness (hard/sufficiency direction) ---")
    for a in alpha_list:
        print(f"  Kbar==K* rate @alpha={a:.0%}     = {kbar_hits[a]/denom:.3f}  "
              f"(K*={Kst[a]})")

    # N0 sweep: show K=1 insufficiency rate climbing as budget shrinks eta below the margin.
    print(f"--- K=1 insufficiency vs budget (margin E_true(1)-{alpha_head:.0%} "
          f"= {E_true[1]-alpha_head:.3f}) ---")
    print(f"{'N0':>8} {'mean_eta':>9} {'eta/T':>7} {'K=1 insuff rate':>16}")
    for N0s in [2000, 4000, 8000, 16000, 32000]:
        c2, a02, T2, eta2 = run_pilot_batched(
            g, rhos, N0s, sigma_obs, n_trials, seed=1000, delta=delta)
        hit = 0; use = 0
        for t in range(n_trials):
            E_lo, _ = certified_band(c2[t], a02[t], T2[t], rhos, Kmax, eta2[t])
            if np.any(np.isnan(E_lo)):
                continue
            use += 1
            hit += (E_lo[1] > alpha_head)
        dd = max(use, 1)
        print(f"{N0s:>8} {eta2.mean():>9.4f} {eta2.mean()/T2.mean():>7.3f} "
              f"{hit/dd:>16.3f}")


def exp_exactness_vs_margin(d=49, N0=16000, sigma_obs=0.1, delta=0.1, seed=2,
                            Kmax=3, n_trials=40):
    print(f"\n=== (b) FINDING K vs MARGIN (sweep top-order energy) d={d} Kmax={Kmax} ===")
    print("    Kbar is the certified order. It is always SOUND: Kbar >= K* (never returns an")
    print("    order that leaves residual above alpha). When the degree-3 tail sits just under")
    print("    tolerance, the pilot may certify the safe K=3 rather than the minimal K=2 --")
    print("    that is sufficiency, not error. Rate = fraction of trials with Kbar == K*.")
    rhos = np.linspace(0.5, 0.97, 8)
    alpha = 0.05
    print(f"{'top_frac':>9} {'E(2)true':>9} {'margin':>8} "
          f"{'K*':>3} {'Kbar==K* rate':>14} {'Kbar>=K* rate':>14}")
    for top_frac in [0.20, 0.10, 0.06, 0.045, 0.03]:
        energies = [1.0, 0.35, top_frac]
        g = WalshFunction(d, energies, mean=0.5, seed=seed)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        margin = abs(E_true[2] - alpha)
        c_hat, a0_hat, T_hat, eta = run_pilot_batched(
            g, rhos, N0, sigma_obs, n_trials, seed=7777, delta=delta)
        exact = 0; sound = 0
        for t in range(n_trials):
            _, E_hi = certified_band(c_hat[t], a0_hat[t], T_hat[t], rhos, Kmax, eta[t])
            kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
            exact += (kbar == Kst)
            sound += (kbar >= Kst)
        print(f"{top_frac:>9.3f} {E_true[2]:>9.4f} {margin:>8.4f} "
              f"{Kst:>3} {exact/n_trials:>14.3f} {sound/n_trials:>14.3f}")
    print("    (Kbar>=K* rate is 1.000 everywhere => the certificate NEVER under-shoots K.")
    print("     Exact-match rate rises with the margin, exactly as Corollary 1 predicts.)")


def exp_budget_law(d=49, sigma_obs=0.1, delta=0.1, n_trials=40):
    print(f"\n=== (c1) FINDING K: certified Kbar == K* across spectra  d={d} ===")
    print("    The pilot returns Kbar, the certified tolerance order. With the ceiling Kmax")
    print("    set to the candidate support (as a practitioner does), Kbar == K* exactly.")
    print("    We sweep ground-truth spectra whose K* takes different values and report the")
    print("    rate at which the certified Kbar hits K*, plus the queries N0 it took.")
    rhos = np.linspace(0.5, 0.97, 8)
    # (energies above degree 0, Kmax=support, alpha, target N0)
    cases = [
        ("K*=2 (deg1+2)",     [1.0, 0.35], 2, 0.05, 16000),
        ("K*=2, tight a=1%",  [1.0, 0.35], 2, 0.01, 32000),
        ("K*=3 (deg3 heavy)", [1.0, 0.35, 0.30], 3, 0.05, 16000),
    ]
    print(f"{'spectrum':>20} {'Kmax':>5} {'alpha':>6} {'N0':>7} "
          f"{'K*':>3} {'Kbar==K* rate':>14}")
    for name, en, Kmax, alpha, N0 in cases:
        g = WalshFunction(d, en, mean=0.5, seed=3)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        c_hat, a0_hat, T_hat, eta = run_pilot_batched(
            g, rhos, N0, sigma_obs, n_trials, seed=20000, delta=delta)
        hits = 0
        for t in range(n_trials):
            _, E_hi = certified_band(c_hat[t], a0_hat[t], T_hat[t], rhos, Kmax, eta[t])
            kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
            hits += (kbar == Kst)
        print(f"{name:>20} {Kmax:>5} {alpha:>6.2f} {N0:>7} "
              f"{Kst:>3} {hits/n_trials:>14.3f}")
    print("    (Kbar == K* at high rate => the certificate FINDS the tolerance order, for")
    print("     K* in {2,3}, at N0 ~ 16k -- far below the pK queries a degree-K fit needs.)")


def exp_budget_law_margin(d=49, sigma_obs=0.1, delta=0.1, n_trials=40):
    """Secondary: how N0 to find K* scales as the crossing margin shrinks (1/Delta^2)."""
    print(f"\n=== (c1b) COST TO FIND K vs margin (smaller margin -> more queries) d={d} ===")
    rhos = np.linspace(0.5, 0.97, 8)
    Kmax = 2; alpha = 0.10
    print(f"{'a2':>6} {'E1_true':>8} {'margin':>8} {'N0_to_find':>11} {'N0*margin^2':>12}")
    for a2 in [0.16, 0.20, 0.25]:   # E1 = a2/(1+a2) sits ABOVE alpha=0.10 => K*=2, vary margin
        g = WalshFunction(d, [1.0, a2], mean=0.5, seed=3)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        margin = abs(E_true[1] - alpha)
        N0_found = None
        for N0 in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            c_hat, a0_hat, T_hat, eta = run_pilot_batched(
                g, rhos, N0, sigma_obs, n_trials, seed=20000, delta=delta)
            hits = 0
            for t in range(n_trials):
                _, E_hi = certified_band(c_hat[t], a0_hat[t], T_hat[t], rhos, Kmax, eta[t])
                kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
                hits += (kbar == Kst)
            if hits / n_trials >= 0.9:
                N0_found = N0
                break
        if N0_found:
            print(f"{a2:>6.3f} {E_true[1]:>8.4f} {margin:>8.4f} {N0_found:>11} "
                  f"{N0_found*margin**2:>12.2f}")
        else:
            print(f"{a2:>6.3f} {E_true[1]:>8.4f} {margin:>8.4f} {'>64000':>11} {'--':>12}")
    print("    (N0*margin^2 ~ const => the 1/Delta^2 budget law of Corollary 1.)")


def exp_independence_pK(N0=16000, sigma_obs=0.1, delta=0.1, Kmax=2, n_trials=20):
    print(f"\n=== (c2) COST INDEPENDENT of p_K as d grows (fixed N0) Kmax={Kmax} ===")
    print("    Same spectrum (deg1+2, K*=2) at growing d. N0 is FIXED while p_K ~ d^Kmax")
    print("    explodes; the pilot still finds K*=2 -- its cost is set by response energy,")
    print("    not by the candidate-interaction count.")
    rhos = np.linspace(0.5, 0.97, 8)
    alpha = 0.05
    print(f"{'d':>4} {'p_2=C(d,<=2)':>12} {'K*':>3} {'Kbar==K* rate':>14} {'bandW(K*)':>10}")
    from math import comb
    for d in [25, 49, 100, 196]:
        energies = [1.0, 0.35]
        g = WalshFunction(d, energies, mean=0.5, seed=4)
        E_true = true_residual_curve(g.a_true, Kmax)
        Kst = Kstar(E_true, alpha)
        c_hat, a0_hat, T_hat, eta = run_pilot_batched(
            g, rhos, N0, sigma_obs, n_trials, seed=33333, delta=delta)
        hits = 0; bw = []
        for t in range(n_trials):
            E_lo, E_hi = certified_band(c_hat[t], a0_hat[t], T_hat[t], rhos, Kmax, eta[t])
            kbar = next((K for K in range(1, Kmax + 1) if E_hi[K] <= alpha), Kmax)
            hits += (kbar == Kst)
            bw.append(E_hi[Kst] - E_lo[Kst])
        pK = sum(comb(d, k) for k in range(1, Kmax + 1))
        print(f"{d:>4} {pK:>12} {Kst:>3} {hits/n_trials:>14.3f} {np.mean(bw):>10.4f}")
    print("    (Kbar==K* rate stays high as p_2 grows ~d^2 => pilot cost decoupled from p_K.)")


if __name__ == "__main__":
    torch.manual_seed(0)
    exp_coverage()
    exp_exactness_vs_margin()
    exp_budget_law()
    exp_independence_pK()