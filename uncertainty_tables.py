# -*- coding: utf-8 -*-
"""Dialogue-bootstrap intervals for Table 1 (ICC(2,1) per judge design) and for
Table 3 (each nested variance component and the paired differences between
components). Uses existing run outputs only."""
import json, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(41)
B = 2000


def jl(fn):
    return pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, fn), encoding="utf-8") if l.strip()])


def icc(x):
    n, k = x.shape; gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - gm) ** 2).sum() - k * ((x.mean(1) - gm) ** 2).sum() - n * ((x.mean(0) - gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def mat(df, col="score"):
    df = df.copy(); df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.pivot_table(index="item_id", columns="run", values=col, aggfunc="first").dropna().to_numpy(float)


C = jl("output_conditions_variance.jsonl"); C = C[C.status == "ok"]
S = pd.concat([jl("output_failure_sample.jsonl"), jl("output_failure_sample_batch2.jsonl")])
rows = [("Direct", jl("output_direct_variance.jsonl")), ("G-Eval", jl("output_geval.jsonl")),
        ("Direct+rubric", C[C.cond == "rubric"]), ("Direct+CoT", C[C.cond == "cot"]), ("Direct+SC", C[C.cond == "sc"]),
        ("Direct+example", C[C.cond == "example"]), ("Direct+combined", jl("output_jointmatched_variance.jsonl")),
        ("Planner+Solver", C[C.cond == "noworker"]), ("PWS", jl("output_pipeline_variance.jsonl"))]
mats = {n: mat(d[d.status == "ok"] if "status" in d else d) for n, d in rows}
mats["Direct (set 2)"] = mat(S[S.direct_status == "ok"], "direct_score")
mats["PWS (set 2)"] = mat(S[S.pipeline_status == "ok"], "pipeline_score")
print("Table 1: ICC(2,1) with 95% dialogue-bootstrap CI")
boot = {}
for n, x in mats.items():
    bs = np.array([icc(x[RNG.integers(0, len(x), len(x))]) for _ in range(B)])
    boot[n] = bs
    print(f"  {n:18s} {icc(x):.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
# paired ICC difference PWS vs each design on the same dialogues (set 1)
print("Paired ICC difference, design minus PWS (same dialogues), 95% CI:")
ref = mats["PWS"]
for n in ["Direct", "Planner+Solver", "Direct+CoT", "Direct+combined"]:
    x = mats[n]; m = min(len(x), len(ref))
    d = []
    for _ in range(B):
        ii = RNG.integers(0, m, m); d.append(icc(x[ii]) - icc(ref[ii]))
    print(f"  {n:18s} {icc(x[:m]) - icc(ref[:m]):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

print("\nTable 3: nested components with 95% CI, and paired differences")
N = jl("output_nested_variance.jsonl"); N = N[N.status == "ok"]
for st in ["orig100", "new100"]:
    comps = []
    for _, g in N[N.set == st].groupby("item_id"):
        y = np.array(g.sort_values("plan_idx").full.tolist(), dtype=float)
        if y.ndim != 3 or np.isnan(y).any():
            continue
        P, W, Sn = y.shape
        s2S = y.var(axis=2, ddof=1).mean(); ab = y.mean(axis=2); vb = ab.var(axis=1, ddof=1).mean()
        comps.append((ab.mean(axis=1).var(ddof=1) - vb / W, vb - s2S / Sn, s2S))
    A = np.array(comps); n = len(A)
    bs = np.array([A[RNG.integers(0, n, n)].mean(0) for _ in range(B)])
    pt = A.mean(0)
    print(f"  {st} (n={n})")
    for j, lab in enumerate(["Planner", "Worker", "Solver"]):
        print(f"    {lab:8s} {pt[j]:.3f} [{np.percentile(bs[:, j], 2.5):.3f}, {np.percentile(bs[:, j], 97.5):.3f}]")
    for a, b, lab in [(0, 1, "Planner - Worker"), (0, 2, "Planner - Solver"), (1, 2, "Worker - Solver")]:
        d = bs[:, a] - bs[:, b]
        print(f"    {lab:17s} {pt[a] - pt[b]:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")
