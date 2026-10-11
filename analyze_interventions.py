# -*- coding: utf-8 -*-
"""Interventions on the PWS judge (first MT-Bench-101 set, GPT-4o-mini): nested
variance components and single-run variance with dialogue-bootstrap CIs, vs the
baseline PWS nested run and Direct; whether the intervention changes the
scores (correlation and mean shift vs the baseline PWS average); and tokens."""
import json, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(67)
B = 4000


def jl(fn):
    p = os.path.join(BASE, fn)
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]) if os.path.exists(p) else None


def comps(D, idx="plan_idx"):
    rows = []
    for iid, g in D.groupby("item_id"):
        try:
            y = np.array(g.sort_values(idx).full.tolist(), dtype=float)
        except (TypeError, ValueError):
            continue
        if y.ndim != 3 or np.isnan(y).any() or y.shape[0] < 2:
            continue
        P, W, S = y.shape
        s2S = y.var(axis=2, ddof=1).mean(); ab = y.mean(axis=2); vb = ab.var(axis=1, ddof=1).mean()
        rows.append(dict(item_id=iid, P=ab.mean(axis=1).var(ddof=1) - vb / W, W=vb - s2S / S, S=s2S, mean=y.mean()))
    C = pd.DataFrame(rows).set_index("item_id"); C["tot"] = C.P + C.W + C.S
    return C


DATASET = os.environ.get("DATASET", "mtb101")
if DATASET == "mtb101":
    base = jl("output_nested_variance.jsonl"); base = base[(base.status == "ok") & (base.set == "orig100")]
    Cb = comps(base)
    dv = jl("output_direct_variance.jsonl"); dv["score"] = pd.to_numeric(dv.score, errors="coerce")
    dvar = dv.pivot_table(index="item_id", columns="run", values="score").var(axis=1, ddof=1)
else:
    mb = jl("output_mtbench_nested_4omini.jsonl")
    Cb = comps(mb[(mb.kind == "plan") & (mb.status == "ok")], idx="idx")
    dd = mb[(mb.kind == "direct") & (mb.status == "ok")].copy(); dd["score"] = pd.to_numeric(dd.score, errors="coerce")
    dvar = dd.pivot_table(index="item_id", columns="idx", values="score").var(axis=1, ddof=1)
print(f"[{DATASET}] baseline PWS: n={len(Cb)} P {Cb.P.mean():.3f} W {Cb.W.mean():.3f} S {Cb.S.mean():.3f} total {Cb.tot.mean():.3f} | Direct {dvar.mean():.3f}")

print(f"  baseline between-dialogue SD of means {Cb['mean'].std():.3f}")
for v in ["short", "prob", "both", "ronly"]:
    D = jl(f"output_intervention_{v}_{DATASET}.jsonl")
    if D is None:
        print(f"\n{v}: no data"); continue
    D = D[D.status == "ok"]
    C = comps(D)
    J = C.join(Cb[["tot", "mean"]].rename(columns={"tot": "btot", "mean": "bmean"}), how="inner").join(dvar.rename("dvar"), how="inner")
    n = len(J)
    if n < 10:
        print(f"\n{v}: only {n} complete dialogues"); continue
    bs = []
    for _ in range(B):
        s = J.iloc[RNG.integers(0, n, n)]
        bs.append((s.tot.mean(), s.tot.mean() / s.btot.mean(), s.tot.mean() / s.dvar.mean(), s.P.mean(), s.W.mean(), s.S.mean()))
    bs = np.array(bs); lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    r = np.corrcoef(J["mean"], J["bmean"])[0, 1]
    print(f"\n=== {v} (n={n}) ===")
    print(f"  single-run variance {J.tot.mean():.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  vs baseline {J.btot.mean():.3f}: ratio {J.tot.mean() / J.btot.mean():.2f} [{lo[1]:.2f}, {hi[1]:.2f}]")
    print(f"  vs Direct {J.dvar.mean():.3f}: ratio {J.tot.mean() / J.dvar.mean():.2f} [{lo[2]:.2f}, {hi[2]:.2f}]")
    print(f"  components P {J.P.mean():.3f} [{lo[3]:.3f}, {hi[3]:.3f}]  W {J.W.mean():.3f} [{lo[4]:.3f}, {hi[4]:.3f}]  S {J.S.mean():.3f} [{lo[5]:.3f}, {hi[5]:.3f}]")
    print(f"  scores vs baseline PWS average: r={r:.3f}, mean shift {J['mean'].mean() - J.bmean.mean():+.3f}")
    vom = (J.P / 3 + J.W / 6 + J.S / 12).mean()  # sampling variance of a dialogue mean under the 3x2x2 design
    rel = (J["mean"].var() - vom) / ((J["mean"].var() - vom) + J.tot.mean())
    print(f"  between-dialogue SD of means {J['mean'].std():.3f} (baseline {J.bmean.std():.3f}); approx single-run reliability {rel:.3f}")
    print(f"  evidence words {D.ev_words.mean():.0f}; tokens per plan unit in {D.tok_in.mean():.0f} out {D.tok_out.mean():.0f}")
