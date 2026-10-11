# -*- coding: utf-8 -*-
"""Run-to-run reliability across benchmarks (MT-Bench-101, MT-Bench) and judge
models (gpt-4o-mini, claude-haiku-4.5, gemini-3.1-flash-lite). Per cell: Direct
ICC(2,1) and within-item variance; pipeline nested variance components
(Planner / Worker / Solver), implied single-run reliability, variance ratio
pipeline/Direct, and Planner variance with vs without the Worker on the same plans."""
import json, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(23)


def jl(fn):
    p = os.path.join(BASE, fn)
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]) if os.path.exists(p) else None


def icc(x):
    n, k = x.shape; gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((x - gm) ** 2).sum() - k * ((x.mean(1) - gm) ** 2).sum() - n * ((x.mean(0) - gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / den if den > 0 else 1.0


def comps(g):
    key = "plan_idx" if "plan_idx" in g and g["plan_idx"].notna().all() else "idx"
    g = g.sort_values(key)
    try:
        y = np.array(g["full"].tolist(), dtype=float); z = np.array(g["noworker"].tolist(), dtype=float)
    except (ValueError, TypeError):
        return None
    if y.ndim != 3 or z.ndim != 2 or np.isnan(y).any() or np.isnan(z).any() or y.shape[0] < 2:
        return None
    P, W, S = y.shape; R = z.shape[1]
    s2S = y.var(axis=2, ddof=1).mean() if S > 1 else np.nan
    ab = y.mean(axis=2); vb = ab.var(axis=1, ddof=1).mean()
    s2W = vb - s2S / S; s2P = ab.mean(axis=1).var(ddof=1) - vb / W
    if R > 1:
        s2Snw = z.var(axis=1, ddof=1).mean(); s2Pnw = z.mean(axis=1).var(ddof=1) - s2Snw / R
    else:
        s2Pnw = np.nan
    return dict(item_id=g["item_id"].iloc[0], s2P=s2P, s2W=s2W, s2S=s2S, s2Pnw=s2Pnw, imean=y.mean(),
                vom=s2P / P + s2W / (P * W) + s2S / (P * W * S), plans=g["plan_hash"].nunique(), P=P)


def cell(direct, plans):
    if direct is None or plans is None or not len(plans):
        return None
    x = direct.dropna().to_numpy(float)
    C = pd.DataFrame([c for _, g in plans.groupby("item_id") if (c := comps(g))])
    if not len(C):
        return None
    tot = C[["s2P", "s2W", "s2S"]].sum(axis=1)
    s2item = C.imean.var(ddof=1) - C.vom.mean()
    out = dict(n=len(C), d_icc=icc(x), d_var=x.var(axis=1, ddof=1).mean(),
               p_rel=s2item / (s2item + tot.mean()) if s2item + tot.mean() > 0 else 1.0, p_var=tot.mean(),
               shP=C.s2P.mean() / tot.mean() if tot.mean() > 0 else np.nan,
               shW=C.s2W.mean() / tot.mean() if tot.mean() > 0 else np.nan,
               shS=C.s2S.mean() / tot.mean() if tot.mean() > 0 else np.nan,
               distinct_plans=(C.plans > 1).mean())
    # variance multiplier on dialogues present in both designs, with a dialogue bootstrap CI
    dv = pd.Series(direct.dropna().var(axis=1, ddof=1), name="dvar")
    J = C.set_index("item_id").join(dv, how="inner")
    J["pvar"] = J[["s2P", "s2W", "s2S"]].sum(axis=1)
    if len(J) and J.dvar.mean() > 0:
        rs = []
        for _ in range(5000):
            s = J.iloc[RNG.integers(0, len(J), len(J))]
            rs.append(s.pvar.mean() / s.dvar.mean() if s.dvar.mean() > 0 else np.nan)
        out.update(mult=J.pvar.mean() / J.dvar.mean(), mult_lo=np.nanpercentile(rs, 2.5),
                   mult_hi=np.nanpercentile(rs, 97.5), n_joint=len(J))
    if C.s2Pnw.notna().all() and C.s2Pnw.mean() > 0:
        n = len(C); bs = []; br = []
        for _ in range(5000):
            s = C.iloc[RNG.integers(0, n, n)]
            bs.append(s.s2P.mean() - s.s2Pnw.mean())
            br.append(s.s2P.mean() / s.s2Pnw.mean() if s.s2Pnw.mean() > 0 else np.nan)
        out.update(amp=C.s2P.mean() / C.s2Pnw.mean(), amp_lo=np.percentile(bs, 2.5), amp_hi=np.percentile(bs, 97.5),
                   ampr_lo=np.nanpercentile(br, 2.5), ampr_hi=np.nanpercentile(br, 97.5), amp_nan=float(np.mean(np.isnan(br))))
    return out


def dmat(df, idxcol):
    df = df[df.status == "ok"].copy(); df["score"] = pd.to_numeric(df.score, errors="coerce")
    return df.pivot_table(index="item_id", columns=idxcol, values="score")


cells = {}
nv = jl("output_nested_variance.jsonl"); dv = jl("output_direct_variance.jsonl")
cells[("MT-Bench-101", "gpt-4o-mini")] = cell(dmat(dv, "run"), nv[(nv.status == "ok") & (nv.set == "orig100")])
for tag, name in [("haiku", "claude-haiku-4.5"), ("gemini", "gemini-3.1-flash-lite")]:
    D = jl(f"output_mm_nested_{tag}.jsonl")
    cells[("MT-Bench-101", name)] = cell(dmat(D[D.kind == "direct"], "idx"), D[(D.kind == "plan") & (D.status == "ok")])
for tag, name in [("4omini", "gpt-4o-mini"), ("haiku", "claude-haiku-4.5"), ("gemini", "gemini-3.1-flash-lite")]:
    D = jl(f"output_mtbench_nested_{tag}.jsonl")
    cells[("MT-Bench", name)] = None if D is None else cell(dmat(D[D.kind == "direct"], "idx"), D[(D.kind == "plan") & (D.status == "ok")])

print(f"{'benchmark':13s} {'judge':22s} {'N':>4} {'Dir ICC':>7} {'Dir var':>7} {'Pipe rel':>8} {'Pipe var':>8} {'ratio':>6} "
      f"{'P/W/S share':>15} {'plans differ':>14} {'amplif (CI of diff)':>24}")
for (b, mname), c in cells.items():
    if c is None:
        print(f"{b:13s} {mname:22s} (no data yet)"); continue
    ratio = c["p_var"] / c["d_var"] if c["d_var"] > 0 else (0.0 if c["p_var"] == 0 else np.inf)
    sh = "-" if np.isnan(c["shP"]) else f"{c['shP']:.0%}/{c['shW']:.0%}/{c['shS']:.0%}"
    amp = f"{c['amp']:.1f}x [{c['amp_lo']:+.3f},{c['amp_hi']:+.3f}]" if "amp" in c else "-"
    print(f"{b:13s} {mname:22s} {c['n']:4d} {c['d_icc']:7.3f} {c['d_var']:7.3f} {c['p_rel']:8.3f} {c['p_var']:8.3f} {ratio:6.1f} "
          f"{sh:>15} {c['distinct_plans']:14.2f} {amp:>24}")
    if "amp" in c:
        print(f"{'':37s} amplification ratio 95% CI [{c['ampr_lo']:.2f}, {c['ampr_hi']:.2f}]"
              f" (bootstrap draws with zero denominator: {c['amp_nan']:.1%})")
    if "mult" in c:
        print(f"{'':37s} multiplier on {c['n_joint']} shared dialogues: {c['mult']:.2f}x  95% CI [{c['mult_lo']:.2f}, {c['mult_hi']:.2f}]")
