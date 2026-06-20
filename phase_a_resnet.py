"""
Phase A -- ResNet-50 / ImageNet adapter + SMOKE TEST (5 images, excluded from study).

Locks the three pre-real constraints:
  (1) MULTIPLICITY: confidence_scope in {"per_input","simultaneous"}.  For
      "simultaneous" over n inputs we set delta_per = delta_global / n inside the
      pilot and label the report accordingly.  We never print a bare "90% certified"
      for a multi-image run.
  (2) DETERMINISM AUDIT on the REAL adapter (eval() in inference_mode, fixed
      preprocessing), re-querying the SAME mask batch.  Audit queries are counted
      separately and can be folded into the budget.
  (3) NO FALSE-ins/FALSE-suf on real outputs -- true E1 is unknown.  Phase C
      (independent K=1/K=2 fit on held-out masks) is the validation, stubbed here.

Adapter contract (matches run_pilot_batched's expectation of WalshFunction):
  .d         : int (= 49 for a 7x7 grid)
  .device    : torch device
  .member    : None  (so _auto_chunk treats n_terms=0)
  .eval(Z)   : Z (..., d) in {0,1} -> tensor (...,) of target-class probability in [0,1]
               internally MICROBATCHES so 16k masked 224x224 images are never all
               materialized at once.

Run (needs torch+torchvision+GPU and a few ImageNet samples):
    python phase_a_resnet.py --images /path/to/5_imagenet_samples
This file is import-safe without torchvision; the smoke test guards on availability.
"""

import argparse
import numpy as np
import torch

from verify_tolerance_order_v3 import (
    run_pilot_batched, certify_order, audit_determinism, DEVICE, PILOT_DTYPE,
)

L4 = np.array([0.0, 0.5, 0.8, 0.97])      # frozen grid


# ----------------------------------------------------------------------
# Query counter
# ----------------------------------------------------------------------
class QueryCounter:
    def __init__(self):
        self.pilot = 0
        self.audit = 0
        self.diag = 0          # sanity-probe queries (range checks), not part of certificate

    def total(self):
        return self.pilot + self.audit + self.diag


# ----------------------------------------------------------------------
# ResNet-50 masked-probability adapter
# ----------------------------------------------------------------------
class ResNet50Adapter:
    """g(z) = P_model(target_class | mask z applied over a 7x7 grid of the image).
    Masked-out cells are replaced by a fixed reference (white/black/blur/mean).
    Deterministic: model in eval()/inference_mode, fixed preprocessing."""

    GRID = 7  # 7x7 -> d=49

    def __init__(self, image_chw, target_class, reference="white",
                 micro=512, counter=None, device=DEVICE):
        """image_chw: preprocessed float tensor (3,224,224) already normalized.
        reference: 'white'|'black'|'blur'|'mean' -- precomputed reference image."""
        self.d = self.GRID * self.GRID
        self.device = device
        self.member = None                       # _auto_chunk: n_terms=0
        self.micro = micro
        self.counter = counter
        self.target = int(target_class)

        self.img = image_chw.to(device).to(torch.float32)        # (3,224,224)
        self.ref = self._make_reference(self.img, reference).to(device)
        # precompute per-cell pixel slices for the 7x7 grid over 224x224 (32px cells)
        self.cell = 224 // self.GRID

    @staticmethod
    def _make_reference(img, kind):
        if kind == "white":
            return torch.ones_like(img)
        if kind == "black":
            return torch.zeros_like(img)
        if kind == "mean":
            return img.mean(dim=(1, 2), keepdim=True).expand_as(img).clone()
        if kind == "blur":
            # cheap separable box blur as a stand-in; replace with Gaussian if desired
            k = 15
            pad = k // 2
            x = img.unsqueeze(0)
            w = torch.ones(3, 1, k, k, device=img.device) / (k * k)
            x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
            return torch.nn.functional.conv2d(x, w, groups=3).squeeze(0)
        raise ValueError(kind)

    def _compose(self, z_rows):
        """z_rows: (B, d) in {0,1} -> (B,3,224,224) masked images.
        Vectorized: expand the 7x7 cell mask to a 224x224 pixel mask via
        repeat_interleave, then blend original/reference in one op (no 49-cell loop)."""
        B = z_rows.shape[0]
        G, c = self.GRID, self.cell
        m = z_rows.view(B, 1, G, G).to(self.img.dtype)            # (B,1,7,7)
        m = m.repeat_interleave(c, dim=2).repeat_interleave(c, dim=3)  # (B,1,224,224)
        # handle 224 not divisible cleanly (224/7=32 exact, but guard anyway)
        if m.shape[2] != 224 or m.shape[3] != 224:
            m = torch.nn.functional.pad(m, (0, 224 - m.shape[3], 0, 224 - m.shape[2]),
                                        value=0.0)
        img = self.img.unsqueeze(0)                                # (1,3,224,224)
        ref = self.ref.unsqueeze(0)
        return m * img + (1.0 - m) * ref                           # (B,3,224,224)

    @torch.inference_mode()
    def eval(self, Z):
        """Z (..., d) in {0,1} -> (...,) target-class probability. Microbatched."""
        lead = Z.shape[:-1]
        flat = Z.reshape(-1, self.d).to(self.device)
        n = flat.shape[0]
        probs = torch.empty(n, device=self.device, dtype=PILOT_DTYPE)
        for s in range(0, n, self.micro):
            chunk = flat[s:s + self.micro]
            imgs = self._compose(chunk)                           # (b,3,224,224)
            logits = self.model(imgs)                             # (b,1000)
            p = torch.softmax(logits, dim=1)[:, self.target]
            probs[s:s + chunk.shape[0]] = p.to(PILOT_DTYPE)
            if self.counter is not None:
                self.counter.pilot += chunk.shape[0]
        return probs.reshape(lead)

    # model is attached externally (shared across adapters to save memory)
    model = None


# ----------------------------------------------------------------------
# Multiplicity-aware pilot wrapper
# ----------------------------------------------------------------------
def screen_one(adapter, alpha, N0, delta_global, n_inputs_total, confidence_scope,
               seed, Kmax=49):
    """Returns (verdict, diagnostics).  Applies the multiplicity correction."""
    if confidence_scope == "simultaneous":
        delta_per = delta_global / max(n_inputs_total, 1)
    elif confidence_scope == "per_input":
        delta_per = delta_global
    else:
        raise ValueError(confidence_scope)

    s_hat, eta_s = run_pilot_batched(
        adapter, L4, N0, sigma_obs=0.0, n_trials=1, seed=seed, delta=delta_per,
        skip_selfpair=True, radius_mode="theorem", out_range=1.0,
    )
    verdict, h_lo, h_hi = certify_order(s_hat[0], L4, Kmax, eta_s[0], alpha)
    # s(0) is the rho=0 column -> direct T_hat
    T_hat = float(s_hat[0][0])
    diag = dict(T_hat=T_hat, h1_lo=float(h_lo[1]), h1_hi=float(h_hi[1]),
                eta_mean=float(np.mean(eta_s[0])), delta_per=delta_per)
    return verdict.get(1), diag


# ----------------------------------------------------------------------
# Smoke test (5 images, EXCLUDED from study)
# ----------------------------------------------------------------------
def smoke_test(images_chw, targets, reference="white", N0=16000, alpha=0.05,
               delta_global=0.1, confidence_scope="per_input"):
    print("=" * 78)
    print(f"PHASE A SMOKE TEST  ResNet-50  ref={reference}  N0={N0}  alpha={alpha}")
    print(f"  confidence_scope={confidence_scope}  delta_global={delta_global}")
    print("  (these 5 images are EXCLUDED from the primary study)")
    print("=" * 78)

    import torchvision
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights).to(DEVICE).eval()
    ResNet50Adapter.model = model

    n = len(images_chw)
    print(f"{'img':>4} {'tgt':>5} {'pmin':>7} {'pmax':>7} {'audit|y1-y2|':>13} "
          f"{'feas':>6} {'verdict':>14} {'T_hat':>7} {'h1_lo':>8} {'h1_hi':>8} {'Q':>10}")
    for i in range(n):
        ctr = QueryCounter()
        ad = ResNet50Adapter(images_chw[i], targets[i], reference=reference,
                             counter=ctr)

        # (2) determinism audit on the REAL adapter; count separately
        before = ctr.pilot
        dd = audit_determinism(ad, n=256, seed=i)
        ctr.audit += (ctr.pilot - before); ctr.pilot = before   # reclassify as audit

        # probability range sanity (diagnostic queries, kept out of certificate budget)
        before = ctr.pilot
        zr = torch.randint(0, 2, (512, ad.d), device=DEVICE, dtype=PILOT_DTYPE)
        pr = ad.eval(zr)
        ctr.diag += (ctr.pilot - before); ctr.pilot = before
        pmin, pmax = float(pr.min()), float(pr.max())

        skip_ok = (dd == 0.0)
        verdict, diag = screen_one(
            ad, alpha, N0, delta_global, n_inputs_total=n,
            confidence_scope=confidence_scope, seed=1000 + i)

        # feasibility: did the pilot produce a non-empty polytope? (LP succeeded)
        feas = "ok" if not np.isnan(diag["h1_lo"]) else "EMPTY"
        print(f"{i:>4} {targets[i]:>5} {pmin:>7.3f} {pmax:>7.3f} {dd:>13.2e} "
              f"{feas:>6} {str(verdict):>14} {diag['T_hat']:>7.3f} "
              f"{diag['h1_lo']:>8.3f} {diag['h1_hi']:>8.3f} {ctr.total():>10,}")

    print("\nChecks: pmin>=0 & pmax<=1; audit|y1-y2|==0 (else skip_selfpair UNSAFE);")
    print("feas=ok; Q ~ (L+1)*N0 + audit. If audit!=0, fall back to self-pair branch.")
    print("Smoke test only -- results NOT reported in the study.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, default=None,
                    help="dir of >=5 ImageNet sample images (jpg/png)")
    ap.add_argument("--reference", type=str, default="white",
                    choices=["white", "black", "blur", "mean"])
    ap.add_argument("--scope", type=str, default="per_input",
                    choices=["per_input", "simultaneous"])
    args = ap.parse_args()

    if args.images is None:
        print("Provide --images DIR with 5 ImageNet samples. Adapter/contract is")
        print("import-safe; this guard avoids running without data.")
        raise SystemExit(0)

    import os, glob
    import torchvision
    from torchvision.io import read_image
    from torchvision.transforms.functional import resize, center_crop

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    paths = sorted(glob.glob(os.path.join(args.images, "*")))[:5]
    if len(paths) < 1:
        print("No images found."); raise SystemExit(1)

    # build preprocessed tensors + predicted (target) class on the ORIGINAL image
    model = torchvision.models.resnet50(weights=weights).to(DEVICE).eval()
    imgs, tgts = [], []
    with torch.inference_mode():
        for p in paths:
            x = read_image(p)
            x = preprocess(x).to(DEVICE)                 # (3,224,224) normalized
            logit = model(x.unsqueeze(0))
            tgt = int(logit.argmax(1).item())            # predicted class = target
            imgs.append(x); tgts.append(tgt)
    smoke_test(imgs, tgts, reference=args.reference,
               confidence_scope=args.scope)
