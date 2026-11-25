# -*- coding: utf-8 -*-
"""Stage-wise run-to-run variance of a multi-stage judge (nested design).

Works on any judge whose stages are sampled one after another, not only the
Planner-Worker-Solver judge of the paper. Collect repeated scores in a long
table with one row per final score:

    item     dialogue / example id
    plan     index of the first-stage draw (e.g. plan) within the item
    worker   index of the second-stage draw within the plan
    solver   index of the final scoring call within the worker draw
    score    numeric score

For a two-stage judge (e.g. Critique-Score) give a constant `worker` column.
Optionally add rows scored without the middle stage (evidence-free) with
`worker` set to "noW"; these give sigma2_P,noW and the amplification ratio.
The design must be balanced within each item.

Usage:
    python nested_components.py scores.csv            # or .jsonl
    python nested_components.py scores.csv --boot 10000 --seed 11

Programmatic use:
    from nested_components import item_components, summarize
"""
import argparse
import json
import numpy as np
import pandas as pd


def item_components(g):
    """Unbiased nested random-effects components for one item (Sec. 3.5)."""
    full = g[g["worker"].astype(str) != "noW"]
    plans = sorted(full["plan"].unique())
    workers = sorted(full["worker"].unique())
    solvers = sorted(full["solver"].unique())
    P, W, S = len(plans), len(workers), len(solvers)
    if P < 2 or len(full) != P * W * S:
        return None
    y = (full.set_index(["plan", "worker", "solver"])["score"]
         .reindex(pd.MultiIndex.from_product([plans, workers, solvers]))
         .to_numpy(dtype=float).reshape(P, W, S))
    if np.isnan(y).any():
        return None
    out = {}
    s2S = y.var(axis=2, ddof=1).mean() if S > 1 else np.nan
    ab = y.mean(axis=2)
    vb = ab.var(axis=1, ddof=1).mean() if W > 1 else np.nan
    if W > 1 and S > 1:
        out["s2S"] = s2S
        out["s2W"] = vb - s2S / S
        out["s2P"] = ab.mean(axis=1).var(ddof=1) - vb / W
    elif S > 1:  # two-stage judge: first stage and final call only
        out["s2S"] = s2S
        out["s2W"] = 0.0
        out["s2P"] = ab.mean(axis=1).var(ddof=1) - s2S / S
    else:
        return None
    nw = g[g["worker"].astype(str) == "noW"]
    if len(nw):
        R = nw.groupby("plan")["score"].size()
        if R.min() == R.max() and R.iloc[0] > 1 and set(R.index) == set(plans):
            z = nw.sort_values(["plan", "solver"]).groupby("plan")["score"].apply(list)
            z = np.array(z.tolist(), dtype=float)
            out["s2Pnw"] = z.mean(axis=1).var(ddof=1) - z.var(axis=1, ddof=1).mean() / z.shape[1]
    out["mean"] = y.mean()
    return out


def summarize(df, n_boot=10000, seed=11):
    """Average components over items, with item-bootstrap 95% intervals."""
    rows = []
    for item, g in df.groupby("item"):
        c = item_components(g)
        if c is not None:
            c["item"] = item
            rows.append(c)
    C = pd.DataFrame(rows)
    if C.empty:
        raise SystemExit("no complete, balanced items found")
    rng = np.random.default_rng(seed)
    keys = [k for k in ["s2P", "s2W", "s2S", "s2Pnw"] if k in C]
    n = len(C)
    boot = {k: [] for k in keys}
    boot["ratio"] = []
    for _ in range(n_boot):
        s = C.iloc[rng.integers(0, n, n)]
        for k in keys:
            boot[k].append(s[k].mean())
        if "s2Pnw" in C:
            den = s["s2Pnw"].mean()
            boot["ratio"].append(s["s2P"].mean() / den if den > 0 else np.nan)
    res = {"items": n}
    for k in keys:
        res[k] = (C[k].mean(), *np.percentile(boot[k], [2.5, 97.5]))
    res["single_run"] = C["s2P"].mean() + C["s2W"].mean() + C["s2S"].mean()
    if "s2Pnw" in C and C["s2Pnw"].mean() > 0:
        r = np.array(boot["ratio"], dtype=float)
        res["ratio"] = (C["s2P"].mean() / C["s2Pnw"].mean(),
                        *np.nanpercentile(r, [2.5, 97.5]), np.isnan(r).mean())
    return res, C


def residual_variance(res, p, w, s):
    """Residual variance of a score averaged over p plans, w draws, s calls (Eq. 7)."""
    return res["s2P"][0] / p + res["s2W"][0] / (p * w) + res["s2S"][0] / (p * w * s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    if a.path.endswith(".jsonl"):
        df = pd.DataFrame([json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()])
    else:
        df = pd.read_csv(a.path)
    res, _ = summarize(df, a.boot, a.seed)
    tot = res["single_run"]
    print(f"complete items: {res['items']}")
    print(f"single-run variance: {tot:.3f}")
    for name, k in [("first stage (P)", "s2P"), ("middle stage (W)", "s2W"), ("final call (S)", "s2S")]:
        m, lo, hi = res[k]
        print(f"  {name:17s} {m:.3f} ({m / tot:4.0%})  95% CI [{lo:.3f}, {hi:.3f}]")
    if "s2Pnw" in res:
        m, lo, hi = res["s2Pnw"]
        print(f"  P without middle  {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    if "ratio" in res:
        m, lo, hi, dropped = res["ratio"]
        print(f"  amplification     {m:.1f}x  95% CI [{lo:.1f}, {hi:.1f}]  ({dropped:.0%} of bootstrap samples undefined)")
    print("residual variance of averaged scores:")
    for p, w, s in [(1, 1, 1), (3, 1, 1), (2, 2, 1), (5, 1, 1), (1, 5, 1), (1, 1, 5)]:
        print(f"  {p} plans x {w} draws x {s} calls: {residual_variance(res, p, w, s):.3f}")


if __name__ == "__main__":
    main()
