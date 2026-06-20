"""
Diagnostic before the primary run: is T_hat=s(0) really capturing the response
variance?  We compare three independent estimates of Var(g) on the SAME adapter:

  (1) s(0) = (1/2) E[(g(z)-g(z'))^2] at rho=0   -> the pilot's T_hat
  (2) direct sample variance of g(z) over fresh masks
  (3) (pmax - pmin) and the full histogram of g(z)

If (1) ~ (2), T_hat is honest and the small value is a real property (the masked
response genuinely has little total interaction energy -> first order has little
to explain, and what little exists is high-degree).  If (1) << (2), s(0) is
under-estimating and the rho=0 column / difference estimator needs inspection.

Also reports the per-degree-1 lower bound: how much energy COULD be first order.
This tells us whether 'insufficient' is driven by genuine high-degree mass or by
a tiny T where alpha*T is below the noise floor.

Run:  python probe_T_diagnostic.py --images DIR --reference black
"""
import argparse, os, glob
import numpy as np
import torch
from verify_tolerance_order_v3 import DEVICE, PILOT_DTYPE
from phase_a_resnet import ResNet50Adapter

@torch.inference_mode()
def direct_stats(ad, N=20000, micro=512, seed=0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randint(0, 2, (N, ad.d), generator=g, device=DEVICE, dtype=PILOT_DTYPE)
    vals = ad.eval(z)                          # (N,)
    return vals

@torch.inference_mode()
def s0_estimate(ad, N=20000, seed=1):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z  = torch.randint(0, 2, (N, ad.d), generator=g, device=DEVICE, dtype=PILOT_DTYPE)
    zp = torch.randint(0, 2, (N, ad.d), generator=g, device=DEVICE, dtype=PILOT_DTYPE)  # rho=0: independent
    y, yp = ad.eval(z), ad.eval(zp)
    return float((0.5 * (y - yp) ** 2).mean())

@torch.inference_mode()
def first_order_energy(ad, N=20000, seed=2):
    """Unbiased estimate of a_1 = sum_i beta_{i}^2 via the rho-coupled slope, but
    cheaply: a_1 >= correlation drop from rho=1 to rho=0 attributable to degree 1.
    Here we just estimate per-coordinate effect: Var of E[g | z_i=1] - E[g | z_i=0].
    Returns a lower-bound proxy on degree-1 energy."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    z = torch.randint(0, 2, (N, ad.d), generator=g, device=DEVICE, dtype=PILOT_DTYPE)
    y = ad.eval(z)
    a1 = 0.0
    for i in range(ad.d):
        zi = z[:, i].bool()
        if zi.any() and (~zi).any():
            diff = y[zi].mean() - y[~zi].mean()   # 2*beta_i estimate
            a1 += float((0.5 * diff) ** 2)
    return a1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--reference", default="black")
    args = ap.parse_args()

    import torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    model = torchvision.models.resnet50(weights=weights).to(DEVICE).eval()
    ResNet50Adapter.model = model

    from torchvision.io import read_image
    paths = sorted(glob.glob(os.path.join(args.images, "*")))[:5]

    print(f"{'img':>4} {'tgt':>5} {'mean_g':>7} {'var_g':>8} {'s0(=T)':>8} "
          f"{'s0/var':>7} {'a1_lb':>8} {'a1/var':>7} {'pmin':>6} {'pmax':>6}")
    with torch.inference_mode():
        for p in paths:
            x = preprocess(read_image(p)).to(DEVICE)
            tgt = int(model(x.unsqueeze(0)).argmax(1).item())
            ad = ResNet50Adapter(x, tgt, reference=args.reference)
            vals = direct_stats(ad)
            var_g = float(vals.var())
            s0 = s0_estimate(ad)
            a1 = first_order_energy(ad)
            print(f"{paths.index(p):>4} {tgt:>5} {float(vals.mean()):>7.3f} "
                  f"{var_g:>8.4f} {s0:>8.4f} {s0/max(var_g,1e-9):>7.3f} "
                  f"{a1:>8.4f} {a1/max(var_g,1e-9):>7.3f} "
                  f"{float(vals.min()):>6.3f} {float(vals.max()):>6.3f}")
    print("\nRead: s0/var should be ~1 (s(0) IS the variance up to a0). If a1/var is")
    print("small, degree-1 genuinely carries little -> insufficient is REAL, not")
    print("a scale artifact. If s0/var << 1, the rho=0 estimator under-reads T.")