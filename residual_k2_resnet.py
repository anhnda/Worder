"""
Direct point-estimate of the held-out residual E(K) for K=1 and K=2 dense Walsh
surrogates on a ResNet-50 masked-probability response over a 7x7 grid (d=49).

NO certificate, NO LP, NO adversary.  Just:
  1. draw N_fit masks, query g(z) = P(target class | masked image)
  2. build the degree-<=K Walsh design matrix (K=1: 1+49 cols; K=2: +C(49,2)=1176 cols)
  3. least-squares fit beta
  4. on a SEPARATE held-out mask set, compute
         E(K) = ||g - g_hat||^2 / ||g - mean(g)||^2     (variance-normalized residual)
  5. report E(1), E(2); 'acceptable at 5%' iff E(K) <= 0.05.

This answers exactly: "with K=2 on a 7x7 grid, what is the residual, and is it
under 5%?"  It is an ESTIMATE with standard error, not a certified bound.

Run:  python residual_k2_resnet.py --images benchmark_50/ --reference black --n-img 10
"""

import argparse, os, glob, itertools
import numpy as np
import torch
from verify_tolerance_order_v3 import DEVICE, PILOT_DTYPE
from phase_a_resnet import ResNet50Adapter


def design_matrix(Z, K, d):
    """Z: (N,d) in {0,1} -> centered Walsh features chi_S for |S|<=K.
    chi_i = 2 z_i - 1.  Returns (N, p_K) including the constant column.
    Supports K up to 3 (degree-3 block = C(d,3) cols)."""
    s = (2.0 * Z - 1.0)                                  # (N,d) in {-1,+1}
    N = Z.shape[0]
    cols = [np.ones((N, 1)), s]                          # deg 0 + deg 1
    if K >= 2:
        idx2 = list(itertools.combinations(range(d), 2))
        pair = np.empty((N, len(idx2)))
        for c, (i, j) in enumerate(idx2):
            pair[:, c] = s[:, i] * s[:, j]
        cols.append(pair)
    if K >= 3:
        idx3 = list(itertools.combinations(range(d), 3))
        trip = np.empty((N, len(idx3)))
        for c, (i, j, k) in enumerate(idx3):
            trip[:, c] = s[:, i] * s[:, j] * s[:, k]
        cols.append(trip)
    if K >= 4:
        raise ValueError("K>=4 not supported by dense fit (p_4 ~ 2e5 cols); use sparse.")
    return np.concatenate(cols, axis=1)


def p_K(d, K):
    from math import comb
    return sum(comb(d, k) for k in range(0, K + 1))


@torch.inference_mode()
def query(adapter, N, seed, micro=1024):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    Z = torch.randint(0, 2, (N, adapter.d), generator=g, device=DEVICE, dtype=PILOT_DTYPE)
    y = adapter.eval(Z)
    return Z.detach().cpu().numpy().astype(np.float64), y.detach().cpu().numpy().astype(np.float64)


def fit_eval(adapter, d, max_k, N_fit, N_test, seed, ridge=0.0):
    Zf, yf = query(adapter, N_fit, seed=seed)
    Zt, yt = query(adapter, N_test, seed=seed + 1)       # independent held-out masks
    denom = np.sum((yt - yt.mean()) ** 2)                # variance of held-out response
    out = {}
    for K in range(1, max_k + 1):
        Xf = design_matrix(Zf, K, d)
        if ridge > 0.0:
            # ridge: (X'X + lam I) beta = X'y, don't penalize the constant col
            p = Xf.shape[1]
            A = Xf.T @ Xf
            reg = ridge * np.eye(p); reg[0, 0] = 0.0
            beta = np.linalg.solve(A + reg, Xf.T @ yf)
        else:
            beta, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
        Xt = design_matrix(Zt, K, d)
        resid = yt - Xt @ beta
        out[K] = float(np.sum(resid ** 2) / denom)       # held-out normalized residual
    return out, denom


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--reference", default="black",
                    choices=["white", "black", "blur", "mean"])
    ap.add_argument("--n-img", type=int, default=10)
    ap.add_argument("--max-k", type=int, default=2, choices=[1, 2, 3],
                    help="fit E(1)..E(max_k). K=3 needs large n-fit (p3=18,424).")
    ap.add_argument("--n-fit", type=int, default=8000,
                    help="must exceed p_K: p2=1226, p3=18,424. K=3 -> use >=25000.")
    ap.add_argument("--n-test", type=int, default=8000)
    ap.add_argument("--ridge", type=float, default=0.0,
                    help="ridge lambda (recommended for K=3 to stabilize the fit, e.g. 1.0)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    import torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    model = torchvision.models.resnet50(weights=weights).to(DEVICE).eval()
    ResNet50Adapter.model = model
    from torchvision.io import read_image

    d = 49
    pk = p_K(d, args.max_k)
    if args.n_fit < pk:
        print(f"!! WARNING: n_fit={args.n_fit} < p_{args.max_k}={pk:,}. Fit is "
              f"under-determined; raise --n-fit (and consider --ridge).")
    paths = sorted(glob.glob(os.path.join(args.images, "*")))[:args.n_img]
    print(f"ResNet-50  ref={args.reference}  7x7 grid (d=49)  max_k={args.max_k}  "
          f"N_fit={args.n_fit} N_test={args.n_test}  p_{args.max_k}={pk:,}  "
          f"ridge={args.ridge}  alpha={args.alpha}")
    hdr = "{:>4} {:>5}".format("img", "tgt") + \
          "".join(f"{'E('+str(K)+')':>8}" for K in range(1, args.max_k + 1)) + \
          f"{'K*':>4}"
    print(hdr)

    Ek_all = {K: [] for K in range(1, args.max_k + 1)}
    kstar_list = []
    with torch.inference_mode():
        for k, p in enumerate(paths):
            x = preprocess(read_image(p)).to(DEVICE)
            tgt = int(model(x.unsqueeze(0)).argmax(1).item())
            ad = ResNet50Adapter(x, tgt, reference=args.reference)
            E, var = fit_eval(ad, d, args.max_k, args.n_fit, args.n_test,
                              seed=1000 + k, ridge=args.ridge)
            # K* = smallest K with E(K) <= alpha among those fit; '>max_k' if none
            kstar = next((K for K in range(1, args.max_k + 1) if E[K] <= args.alpha), None)
            kstar_str = str(kstar) if kstar else f">{args.max_k}"
            kstar_list.append(kstar)
            row = f"{k:>4} {tgt:>5}" + "".join(f"{E[K]:>8.3f}" for K in range(1, args.max_k + 1)) + \
                  f"{kstar_str:>4}"
            print(row)
            for K in range(1, args.max_k + 1):
                Ek_all[K].append(E[K])

    print("\nmedian " + "  ".join(f"E({K})={np.median(Ek_all[K]):.3f}"
                                  for K in range(1, args.max_k + 1)))
    resolved = [k for k in kstar_list if k is not None]
    print(f"images with K* found within max_k={args.max_k}: {len(resolved)}/{len(paths)}")
    if resolved:
        print(f"  among those, K* distribution: " +
              ", ".join(f"K*={v}:{resolved.count(v)}" for v in sorted(set(resolved))))
    if len(resolved) < len(paths):
        print(f"  {len(paths)-len(resolved)} image(s): E(max_k) still > {args.alpha} "
              f"-> K* > {args.max_k} (raise --max-k or accept dense limit at K=3).")
    print("E(K) = held-out variance-normalized residual of a degree-<=K dense fit "
          "(point estimate, not a certified bound).")