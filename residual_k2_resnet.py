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
    chi_i = 2 z_i - 1.  Returns (N, p_K) including the constant column."""
    s = (2.0 * Z - 1.0)                                  # (N,d) in {-1,+1}
    N = Z.shape[0]
    cols = [np.ones((N, 1))]                              # degree 0
    cols.append(s)                                       # degree 1: d cols
    if K >= 2:
        idx = list(itertools.combinations(range(d), 2))
        pair = np.empty((N, len(idx)))
        for c, (i, j) in enumerate(idx):
            pair[:, c] = s[:, i] * s[:, j]
        cols.append(pair)                                # degree 2: C(d,2) cols
    return np.concatenate(cols, axis=1)


@torch.inference_mode()
def query(adapter, N, seed, micro=1024):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    Z = torch.randint(0, 2, (N, adapter.d), generator=g, device=DEVICE, dtype=PILOT_DTYPE)
    y = adapter.eval(Z)
    return Z.detach().cpu().numpy().astype(np.float64), y.detach().cpu().numpy().astype(np.float64)


def fit_eval(adapter, d, N_fit, N_test, seed):
    Zf, yf = query(adapter, N_fit, seed=seed)
    Zt, yt = query(adapter, N_test, seed=seed + 1)       # independent held-out masks
    denom = np.sum((yt - yt.mean()) ** 2)                # variance of held-out response
    out = {}
    for K in (1, 2):
        Xf = design_matrix(Zf, K, d)
        beta, *_ = np.linalg.lstsq(Xf, yf, rcond=None)   # LS fit on fit set
        Xt = design_matrix(Zt, K, d)
        resid = yt - Xt @ beta
        E = float(np.sum(resid ** 2) / denom)            # held-out normalized residual
        out[K] = E
    return out, denom, len(yt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--reference", default="black",
                    choices=["white", "black", "blur", "mean"])
    ap.add_argument("--n-img", type=int, default=10)
    ap.add_argument("--n-fit", type=int, default=8000,
                    help=">= p_2 = 1+49+1176 = 1226 for K=2 to be well-posed; 8000 gives headroom")
    ap.add_argument("--n-test", type=int, default=8000)
    args = ap.parse_args()

    import torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    model = torchvision.models.resnet50(weights=weights).to(DEVICE).eval()
    ResNet50Adapter.model = model
    from torchvision.io import read_image

    paths = sorted(glob.glob(os.path.join(args.images, "*")))[:args.n_img]
    print(f"ResNet-50  ref={args.reference}  7x7 grid (d=49)  "
          f"N_fit={args.n_fit} N_test={args.n_test}  (p2=1226)")
    print(f"{'img':>4} {'tgt':>5} {'E(1)':>7} {'E(2)':>7} {'K=2<=5%?':>9} {'var_test':>9}")
    e1s, e2s, ok2 = [], [], 0
    with torch.inference_mode():
        for k, p in enumerate(paths):
            x = preprocess(read_image(p)).to(DEVICE)
            tgt = int(model(x.unsqueeze(0)).argmax(1).item())
            ad = ResNet50Adapter(x, tgt, reference=args.reference)
            E, var, n = fit_eval(ad, ad.d, args.n_fit, args.n_test, seed=1000 + k)
            acc = "YES" if E[2] <= 0.05 else "no"
            ok2 += (E[2] <= 0.05)
            e1s.append(E[1]); e2s.append(E[2])
            print(f"{k:>4} {tgt:>5} {E[1]:>7.3f} {E[2]:>7.3f} {acc:>9} {var:>9.4f}")
    print(f"\nmedian E(1)={np.median(e1s):.3f}  median E(2)={np.median(e2s):.3f}")
    print(f"images with E(2)<=5%: {ok2}/{len(paths)}")
    print("E(K) = held-out variance-normalized residual of a degree-<=K dense fit.")
    print("This is a point estimate (has sampling error), not a certified bound.")