# -*- coding: utf-8 -*-
"""Stage-wise variance decomposition. Within-item run-to-run variance for
S (Solver only), WS (Worker+Solver), PWS (full pipeline), Direct. Increments:
Worker = V_WS - V_S, Planner = V_PWS - V_WS, with item-bootstrap CIs.
S2 (stored evidence, current conversation framing) checks the framing confound,
since the existing PWS runs used the current framing."""
import os
import json, itertools
from collections import Counter
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(7)


def jl(fn):
    return pd.DataFrame([json.loads(l) for l in open(BASE + "\\" + fn, encoding="utf-8") if l.strip()])


def mat(df, col="score"):
    df = df.copy(); df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.pivot_table(index="item_id", columns="run", values=col, aggfunc="first")


def icc(x):
    n, k = x.shape; gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - gm) ** 2).sum() - k * ((x.mean(1) - gm) ** 2).sum() - n * ((x.mean(0) - gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def pair_flip(x):
    n = len(x); fl = tot = 0
    for _ in range(20000):
        i, j = RNG.choice(n, 2, replace=False); a, b = RNG.choice(x.shape[1], 2, replace=False)
        da, db = x[i, a] - x[j, a], x[i, b] - x[j, b]
        if da and db:
            tot += 1; fl += np.sign(da) != np.sign(db)
    return fl / tot


SF = jl("output_stage_fixed.jsonl"); SF = SF[SF.status == "ok"]
FS = pd.concat([jl("output_failure_sample.jsonl"), jl("output_failure_sample_batch2.jsonl")], ignore_index=True)
pv = jl("output_pipeline_variance.jsonl"); dv = jl("output_direct_variance.jsonl")

sets = {
    "orig100": {"S": mat(SF[(SF.set == "orig100") & (SF.cond == "S")]),
                "S2": mat(SF[(SF.set == "orig100") & (SF.cond == "S2")]),
                "WS": mat(SF[(SF.set == "orig100") & (SF.cond == "WS")]),
                "PWS": mat(pv[pv.status == "ok"]), "Direct": mat(dv[dv.status == "ok"])},
    "new299": {"S": mat(SF[(SF.set == "new299") & (SF.cond == "S")]),
               "S2": mat(SF[(SF.set == "new299") & (SF.cond == "S2")]),
               "WS": mat(SF[(SF.set == "new299") & (SF.cond == "WS")]),
               "PWS": mat(FS[FS.pipeline_status == "ok"], "pipeline_score"),
               "Direct": mat(FS[FS.direct_status == "ok"], "direct_score")},
}

for sname, M in sets.items():
    common = None
    for m in M.values():
        idx = m.dropna().index
        common = idx if common is None else common.intersection(idx)
    print("=" * 78); print(f"{sname}: items with all 5 runs in every condition = {len(common)}"); print("=" * 78)
    print(f"{'cond':7s} {'ICC(2,1)':>9} {'within-var':>11} {'item SD':>8} {'all-same':>9} {'pair flip':>10} {'mean':>6}")
    X = {c: m.loc[common].to_numpy(float) for c, m in M.items()}
    for c in ["S", "S2", "WS", "PWS", "Direct"]:
        x = X[c]; wv = x.var(axis=1, ddof=1)
        print(f"{c:7s} {icc(x):9.3f} {wv.mean():11.3f} {np.sqrt(wv).mean():8.3f} {(x.max(1) == x.min(1)).mean():9.0%} {pair_flip(x):10.0%} {x.mean():6.2f}")

    wv = {c: X[c].var(axis=1, ddof=1) for c in X}
    n = len(common); boots = {"Solver (V_S)": [], "Worker (V_WS-V_S)": [], "Planner (V_PWS-V_WS)": [], "framing (V_S2-V_S)": []}
    for _ in range(10000):
        ii = RNG.integers(0, n, n)
        boots["Solver (V_S)"].append(wv["S"][ii].mean())
        boots["Worker (V_WS-V_S)"].append(wv["WS"][ii].mean() - wv["S"][ii].mean())
        boots["Planner (V_PWS-V_WS)"].append(wv["PWS"][ii].mean() - wv["WS"][ii].mean())
        boots["framing (V_S2-V_S)"].append(wv["S2"][ii].mean() - wv["S"][ii].mean())
    tot = wv["PWS"].mean()
    print(f"\nvariance decomposition (total PWS within-item variance = {tot:.3f}; Direct = {wv['Direct'].mean():.3f})")
    for k, v in boots.items():
        pt = {"Solver (V_S)": wv["S"].mean(), "Worker (V_WS-V_S)": wv["WS"].mean() - wv["S"].mean(),
              "Planner (V_PWS-V_WS)": wv["PWS"].mean() - wv["WS"].mean(), "framing (V_S2-V_S)": wv["S2"].mean() - wv["S"].mean()}[k]
        share = f"{pt / tot:6.0%}" if "framing" not in k else "      "
        print(f"  {k:22s} {pt:+.3f} {share}  95% CI [{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")

    # evidence text identity across WS reruns
    ws = SF[(SF.set == sname) & (SF.cond == "WS")]
    same_all = tot_e = 0; modal = []
    for iid, g in ws.groupby("item_id"):
        hs = [r for r in g["ev_hash"] if isinstance(r, dict)]
        for e in set(itertools.chain.from_iterable(hs)):
            vals = [d.get(e) for d in hs]
            c = Counter(vals); tot_e += 1; same_all += len(c) == 1; modal.append(c.most_common(1)[0][1] / len(vals))
    print(f"\nWorker evidence strings identical across all 5 reruns: {same_all}/{tot_e} ({same_all / max(tot_e, 1):.0%}); mean modal share {np.mean(modal):.2f}")
    print()
