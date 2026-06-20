"""
Phase B -- PRIMARY real-model study.

Models   : ResNet-50, ViT-B/16        (both masked on a 7x7 grid -> d=49)
References: white, black, blur        (all three; robustness is the finding)
Scopes   : per_input (1-delta each) AND simultaneous (delta/n)  -- both reported
Images   : N_IMG predicted-correct ImageNet samples, FIXED SEED, NOT confidence-ranked
Pilot    : alpha=0.05, L4 frozen, N0=16k, Kmax=49, sigma=0, theorem radius (out_range=1)

Reports (per model x reference x scope):
  insufficient / sufficient / unresolved fractions
  feasibility rate
  distribution of T_hat = s(0)  (small in practice -- shown explicitly)
  distribution of the certified [h1_lo, h1_hi] interval
  queries per input (= (L+1)*N0 + audit) and total

Writes per-image rows to CSV (model, ref, scope, img, tgt, T_hat, h1_lo, h1_hi,
verdict, eta_mean, pmin, pmax) for Phase C independent validation.

NO false-rate columns: true E1 is unknown on real models (that is Phase C's job).

Run:
  python phase_b_primary.py --images DIR --n-img 50 --out primary.csv
  [--models resnet50 vit_b_16] [--refs white black blur] [--share-samples]
"""

import argparse, os, glob, csv, time
import numpy as np
import torch

from verify_tolerance_order_v3 import (
    run_pilot_batched, certify_order, audit_determinism, DEVICE, PILOT_DTYPE,
)
from phase_a_resnet import ResNet50Adapter, QueryCounter, L4

INSUFFICIENT = "insuff"; SUFFICIENT = "suff"; UNRESOLVED = "unres"


# ----------------------------------------------------------------------
# ViT adapter: same masking/compose as ResNet, different forward
# ----------------------------------------------------------------------
class ViTAdapter(ResNet50Adapter):
    """Inherits _compose / eval from ResNet50Adapter; only .model differs (set
    externally).  7x7 grid over 224x224 -> each cell spans 2x2 ViT patches."""
    model = None


def build_model(name):
    import torchvision
    if name == "resnet50":
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        m = torchvision.models.resnet50(weights=w)
    elif name == "vit_b_16":
        w = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        m = torchvision.models.vit_b_16(weights=w)
    else:
        raise ValueError(name)
    return m.to(DEVICE).eval(), w


def adapter_for(name):
    return ViTAdapter if name == "vit_b_16" else ResNet50Adapter


# ----------------------------------------------------------------------
# Image selection: predicted-correct, fixed seed, NOT confidence-ranked
# (helpers _select_paths / _load_with_targets are defined below in run_primary)
# ----------------------------------------------------------------------
def _label_from_name(path):
    base = os.path.basename(path)
    head = base.split("_")[0]
    return int(head) if head.isdigit() else None


# ----------------------------------------------------------------------
# One screening result
# ----------------------------------------------------------------------
def screen(adapter, alpha, N0, delta, seed, Kmax=49):
    s_hat, eta_s = run_pilot_batched(
        adapter, L4, N0, sigma_obs=0.0, n_trials=1, seed=seed,
        delta=delta, skip_selfpair=True, radius_mode="theorem", out_range=1.0,
    )
    verdict, h_lo, h_hi = certify_order(s_hat[0], L4, Kmax, eta_s[0], alpha)
    return dict(
        verdict=verdict.get(1), T_hat=float(s_hat[0][0]),
        h1_lo=float(h_lo[1]), h1_hi=float(h_hi[1]),
        eta_mean=float(np.mean(eta_s[0])),
    )


# ----------------------------------------------------------------------
# Primary loop
# ----------------------------------------------------------------------
def run_primary(image_dir, n_img, out_csv, models, refs, N0=16000, alpha=0.05,
                delta_global=0.1, seed=12345):
    scopes = {"per_input": delta_global, "simultaneous": delta_global / n_img}

    fieldnames = ["model", "ref", "scope", "img", "path", "tgt", "T_hat",
                  "h1_lo", "h1_hi", "verdict", "eta_mean", "pmin", "pmax",
                  "delta_per", "Q_pilot", "Q_audit"]
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                done.add((r["model"], r["ref"], r["scope"], r["img"]))
        fh = open(out_csv, "a", newline=""); writer = csv.DictWriter(fh, fieldnames)
    else:
        fh = open(out_csv, "w", newline=""); writer = csv.DictWriter(fh, fieldnames)
        writer.writeheader()

    # ---- select the SHARED image set ONCE (fixed seed) using the FIRST model's
    # predictions as the target; the same paths+masks are used for every model so
    # comparisons are on an identical input set.  target class is per (model,image)
    # since each model has its own prediction; we store the path set here and
    # recompute each model's own predicted target inside the loop.
    sel_model, sel_w = build_model(models[0])
    shared = _select_paths(image_dir, n_img, seed, exclude=5)
    print(f"\n### shared image set: {len(shared)} paths (fixed seed, smoke excluded)")
    del sel_model

    for mname in models:
        model, weights = build_model(mname)
        Adapter = adapter_for(mname)
        Adapter.model = model
        preprocess = weights.transforms()
        imgs = _load_with_targets(shared, model, preprocess)
        print(f"### {mname}: {len(imgs)} images, target = model's own prediction.")

        for ref in refs:
            for scope, delta_per in scopes.items():
                rows = []
                t0 = time.time()
                for idx, (path, x_cpu, tgt) in enumerate(imgs):
                    key = (mname, ref, scope, str(idx))
                    if key in done:
                        continue
                    ctr = QueryCounter()
                    ad = Adapter(x_cpu.to(DEVICE), tgt, reference=ref, counter=ctr)

                    before = ctr.pilot
                    dd = audit_determinism(ad, n=256, seed=idx)
                    ctr.audit += (ctr.pilot - before); ctr.pilot = before
                    if dd != 0.0:
                        print(f"  !! {mname}/{ref}/{scope} img{idx}: NON-deterministic "
                              f"(|y1-y2|={dd:.2e}); skipped.")
                        continue

                    before = ctr.pilot
                    zr = torch.randint(0, 2, (512, ad.d), device=DEVICE, dtype=PILOT_DTYPE)
                    pr = ad.eval(zr)
                    ctr.diag += (ctr.pilot - before); ctr.pilot = before
                    pmin, pmax = float(pr.min()), float(pr.max())

                    r = screen(ad, alpha, N0, delta_per, seed=70000 + idx)
                    row = dict(model=mname, ref=ref, scope=scope, img=idx,
                               path=os.path.basename(path), tgt=tgt,
                               T_hat=r["T_hat"], h1_lo=r["h1_lo"], h1_hi=r["h1_hi"],
                               verdict=r["verdict"], eta_mean=r["eta_mean"],
                               pmin=pmin, pmax=pmax, delta_per=delta_per,
                               Q_pilot=ctr.pilot, Q_audit=ctr.audit)
                    writer.writerow(row); fh.flush()
                    rows.append(row)
                _summary(mname, ref, scope, rows, time.time() - t0)
    fh.close()
    print(f"\nWrote {out_csv}.  Use it for Phase C independent validation.")


def _select_paths(image_dir, n_img, seed, exclude=5):
    paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    rng = np.random.default_rng(seed)
    rng.shuffle(paths)
    return paths[exclude:exclude + n_img]


def _load_with_targets(paths, model, preprocess):
    from torchvision.io import read_image
    out = []
    with torch.inference_mode():
        for p in paths:
            try:
                x = preprocess(read_image(p)).to(DEVICE)
            except Exception:
                continue
            pred = int(model(x.unsqueeze(0)).argmax(1).item())
            label = _label_from_name(p)
            if label is not None and label != pred:
                continue   # predicted-correct filter when label is known
            out.append((p, x.cpu(), pred))
    return out


def _summary(mname, ref, scope, rows, secs):
    if not rows:
        print(f"  {mname}/{ref}/{scope}: (all cached)"); return
    n = len(rows)
    frac = lambda v: sum(r["verdict"] == v for r in rows) / n
    T = np.array([r["T_hat"] for r in rows])
    lo = np.array([r["h1_lo"] for r in rows]); hi = np.array([r["h1_hi"] for r in rows])
    feas = np.mean([not np.isnan(r["h1_lo"]) for r in rows])
    Q = rows[0]["Q_pilot"] + rows[0]["Q_audit"]
    print(f"  {mname}/{ref}/{scope:>12}  n={n}  "
          f"insuf={frac(INSUFFICIENT):.2f} suf={frac(SUFFICIENT):.2f} "
          f"unres={frac(UNRESOLVED):.2f}  feas={feas:.2f}  "
          f"T_hat[med={np.median(T):.4f}, p10={np.percentile(T,10):.4f}, "
          f"p90={np.percentile(T,90):.4f}]  "
          f"h1[lo_med={np.median(lo):.4f}, hi_med={np.median(hi):.4f}]  "
          f"Q/img={Q:,}  {secs:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n-img", type=int, default=50)
    ap.add_argument("--out", type=str, default="primary.csv")
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit_b_16"])
    ap.add_argument("--refs", nargs="+", default=["white", "black", "blur"])
    ap.add_argument("--N0", type=int, default=16000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    print(f"device={DEVICE}  models={args.models}  refs={args.refs}  "
          f"n_img={args.n_img}  N0={args.N0}  alpha={args.alpha}  delta={args.delta}")
    print(f"per-input delta={args.delta}; simultaneous delta={args.delta/args.n_img:.4f} "
          f"(= delta/n over {args.n_img} images)")
    run_primary(args.images, args.n_img, args.out, args.models, args.refs,
                N0=args.N0, alpha=args.alpha, delta_global=args.delta, seed=args.seed)
