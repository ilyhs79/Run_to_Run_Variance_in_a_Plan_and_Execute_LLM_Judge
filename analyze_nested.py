# -*- coding: utf-8 -*-
"""Nested random-effects variance components per item, averaged over items:
  full pipeline : planner (P) > worker draw (W) > solver call (S)
  no Worker     : planner (P) > solver call (R), same plans
Amplification = sigma2_P(full) - sigma2_P(noWorker). Item-bootstrap CIs.
Also: residual variance and call cost of alternative averaging designs."""
import os
import json
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(11)
D = pd.DataFrame([json.loads(l) for l in open(BASE + r"\output_nested_variance.jsonl", encoding="utf-8") if l.strip()])
print("records:", len(D), D["status"].value_counts().to_dict())
D = D[D.status == "ok"]


def comps(g):
    g = g.sort_values("plan_idx")
    y = np.array(g["full"].tolist(), dtype=float)      # P x W x S
    z = np.array(g["noworker"].tolist(), dtype=float)  # P x R
    if y.ndim != 3 or z.ndim != 2 or np.isnan(y).any() or np.isnan(z).any() or y.shape[0] < 4:
        return None
    P, W, S = y.shape; R = z.shape[1]
    s2S = y.var(axis=2, ddof=1).mean()
    ab = y.mean(axis=2)
    vb = ab.var(axis=1, ddof=1).mean()
    s2W = vb - s2S / S
    s2P = ab.mean(axis=1).var(ddof=1) - vb / W
    s2Snw = z.var(axis=1, ddof=1).mean()
    s2Pnw = z.mean(axis=1).var(ddof=1) - s2Snw / R
    return dict(s2P=s2P, s2W=s2W, s2S=s2S, s2Pnw=s2Pnw, s2Snw=s2Snw,
                n_ev=g["n_evidence"].mean(), plans_distinct=g["plan_hash"].nunique(),
                nev_distinct=g["n_evidence"].nunique(), mean_full=y.mean(), mean_nw=z.mean())


rows = []
for (iid, st), g in D.groupby(["item_id", "set"]):
    c = comps(g)
    if c:
        c.update(item_id=iid, set=st); rows.append(c)
C = pd.DataFrame(rows)

for st in ["orig100", "new100"]:
    X = C[C.set == st].reset_index(drop=True); n = len(X)
    print("=" * 74); print(f"{st}: complete items = {n}"); print("=" * 74)
    tot = X[["s2P", "s2W", "s2S"]].sum(axis=1).mean()
    keys = [("Planner", "s2P"), ("Worker", "s2W"), ("Solver", "s2S")]
    B = {k: [] for _, k in keys}; B["amp"] = []; B["s2Pnw"] = []; B["ratio"] = []
    for _ in range(10000):
        ii = RNG.integers(0, n, n); s = X.iloc[ii]
        for _, k in keys:
            B[k].append(s[k].mean())
        B["s2Pnw"].append(s["s2Pnw"].mean()); B["amp"].append(s["s2P"].mean() - s["s2Pnw"].mean())
        B["ratio"].append(s["s2P"].mean() / max(s["s2Pnw"].mean(), 1e-9))
    ci = lambda k: f"[{np.percentile(B[k], 2.5):+.3f}, {np.percentile(B[k], 97.5):+.3f}]"
    print(f"full pipeline, single-run variance implied = {tot:.3f}")
    for name, k in keys:
        print(f"  {name:8s} {X[k].mean():+.3f}  ({X[k].mean() / tot:4.0%})  95% CI {ci(k)}")
    print(f"same plans, no Worker: Planner {X.s2Pnw.mean():+.3f} {ci('s2Pnw')}  Solver {X.s2Snw.mean():+.3f}")
    print(f"amplification sigma2_P(full) - sigma2_P(noWorker) = {X.s2P.mean() - X.s2Pnw.mean():+.3f}  95% CI {ci('amp')}")
    print(f"ratio sigma2_P(full)/sigma2_P(noWorker) = {X.s2P.mean() / X.s2Pnw.mean():.1f}x  95% CI [{np.percentile(B['ratio'], 2.5):.1f}, {np.percentile(B['ratio'], 97.5):.1f}]")
    print(f"plans: distinct texts per item (of 4) mean {X.plans_distinct.mean():.2f}; items whose 4 plans differ in #evidence: {(X.nev_distinct > 1).mean():.0%}")

    # averaging designs: residual variance of the item score and calls per item
    sP, sW, sS = X.s2P.mean(), X.s2W.mean(), X.s2S.mean(); ne = X.n_ev.mean()
    print(f"\naveraging designs (mean #evidence/plan = {ne:.1f}); calls = p*(1 + w*(n_ev + s))")
    print(f"  {'design':38s} {'calls':>6} {'resid var':>9}")
    for lab, p, w, s in [("1 full run", 1, 1, 1), ("5 full runs (current ensemble)", 5, 1, 1),
                         ("1 plan, 5 worker+solver reruns", 1, 5, 1), ("1 plan, 1 worker, 5 solver reruns", 1, 1, 5),
                         ("3 plans x 1 worker x 1 solver", 3, 1, 1), ("2 plans x 2 workers x 1 solver", 2, 2, 1)]:
        calls = p * (1 + w * (ne + s))
        rv = sP / p + sW / (p * w) + sS / (p * w * s)
        print(f"  {lab:38s} {calls:6.1f} {rv:9.3f}")
    print()
