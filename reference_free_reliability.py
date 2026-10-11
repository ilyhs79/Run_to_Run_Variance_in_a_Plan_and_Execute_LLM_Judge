# -*- coding: utf-8 -*-
"""Reference-free run-to-run reliability of each judge design (no human labels).
Items x runs matrix per condition: ICC(2,1)/ICC(2,k) across runs, mean per-item SD,
share of items with an identical score in every run, and pairwise flip rate of
which of two dialogues scores higher."""
import os
import json, itertools
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def jl(fn):
    return pd.DataFrame([json.loads(l) for l in open(BASE + "\\" + fn, encoding="utf-8") if l.strip()])


def icc(x):
    n, k = x.shape; gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - gm) ** 2).sum() - k * ((x.mean(1) - gm) ** 2).sum() - n * ((x.mean(0) - gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n), (msr - mse) / (msr + (msc - mse) / n)


def stats_for(df, col="score"):
    df = df.copy(); df[col] = pd.to_numeric(df[col], errors="coerce")
    x = df.pivot_table(index="item_id", columns="run", values=col, aggfunc="first").dropna().to_numpy(float)
    i1, ik = icc(x)
    sd = x.std(axis=1, ddof=1).mean()
    same = (x.max(1) == x.min(1)).mean()
    # pairwise ordering flips: for item pairs, does run a vs run b disagree on which is higher (ignoring ties in either)
    rng = np.random.default_rng(0); n = len(x); fl = tot = 0
    for _ in range(20000):
        i, j = rng.choice(n, 2, replace=False); a, b = rng.choice(x.shape[1], 2, replace=False)
        da, db = x[i, a] - x[j, a], x[i, b] - x[j, b]
        if da != 0 and db != 0:
            tot += 1; fl += np.sign(da) != np.sign(db)
    return len(x), x.shape[1], i1, ik, sd, same, fl / tot


C = jl("output_conditions_variance.jsonl"); C = C[C.status == "ok"]
rows = [("Direct", jl("output_direct_variance.jsonl")), ("G-Eval (logprob)", jl("output_geval.jsonl")),
        ("Direct+rubric", C[C.cond == "rubric"]), ("Direct+CoT", C[C.cond == "cot"]),
        ("Direct+SC", C[C.cond == "sc"]), ("Direct+example", C[C.cond == "example"]),
        ("Direct+combined", jl("output_jointmatched_variance.jsonl")),
        ("Planner+Solver", C[C.cond == "noworker"]), ("Ours (P-W-S)", jl("output_pipeline_variance.jsonl"))]

print("Original 100 items, gpt-4o-mini, 5 runs each")
print(f"{'condition':18s} {'N':>4} {'K':>2} {'ICC(2,1)':>9} {'ICC(2,k)':>9} {'item SD':>8} {'all-same':>9} {'pair flip':>9}")
for name, df in rows:
    df = df[df["status"] == "ok"] if "status" in df else df
    n, k, i1, ik, sd, same, fl = stats_for(df)
    print(f"{name:18s} {n:4d} {k:2d} {i1:9.3f} {ik:9.3f} {sd:8.3f} {same:9.0%} {fl:9.0%}")

S = pd.concat([jl("output_failure_sample.jsonl"), jl("output_failure_sample_batch2.jsonl")], ignore_index=True)
print("\n299 new items (13 categories), gpt-4o-mini, 5 runs each")
for name, col, st in [("Direct", "direct_score", "direct_status"), ("Ours (P-W-S)", "pipeline_score", "pipeline_status")]:
    n, k, i1, ik, sd, same, fl = stats_for(S[S[st] == "ok"], col)
    print(f"{name:18s} {n:4d} {k:2d} {i1:9.3f} {ik:9.3f} {sd:8.3f} {same:9.0%} {fl:9.0%}")
