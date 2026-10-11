# -*- coding: utf-8 -*-
"""Critique -> Score judge: nested variance (critique vs scorer), ratio to the
Direct judge's run-to-run variance on the same dialogues, implied single-run
reliability, and how often critiques differ, with dialogue-bootstrap CIs."""
import json, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(59)
B = 4000


def jl(fn):
    return pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, fn), encoding="utf-8") if l.strip()])


CS = jl("output_cs_nested.jsonl"); CS = CS[CS.status == "ok"]
dv = jl("output_direct_variance.jsonl"); dv["score"] = pd.to_numeric(dv.score, errors="coerce")
direct = {"mtb101": dv.pivot_table(index="item_id", columns="run", values="score").dropna()}
m4 = jl("output_mtbench_nested_4omini.jsonl"); m4 = m4[m4.kind == "direct"].copy(); m4["score"] = pd.to_numeric(m4.score, errors="coerce")
direct["mtbench"] = m4.pivot_table(index="item_id", columns="idx", values="score").dropna()

for ds, label in [("mtb101", "MT-Bench-101 (first set)"), ("mtbench", "MT-Bench")]:
    rows = []
    for iid, g in CS[CS.dataset == ds].groupby("item_id"):
        y = np.array(g.sort_values("crit_idx").scores.tolist(), dtype=float)  # P x S
        if y.ndim != 2 or np.isnan(y).any() or y.shape[0] < 2:
            continue
        P, S = y.shape
        s2S = y.var(axis=1, ddof=1).mean()
        s2C = y.mean(axis=1).var(ddof=1) - s2S / S
        rows.append(dict(item_id=iid, s2C=s2C, s2S=s2S, imean=y.mean(), vom=s2C / P + s2S / (P * S),
                         crit_differ=g.crit_hash.nunique() > 1, crit_len=g.crit_len.mean()))
    C = pd.DataFrame(rows).set_index("item_id")
    d = direct[ds]
    C = C.join(d.var(axis=1, ddof=1).rename("dvar"), how="inner")
    C["tot"] = C.s2C + C.s2S
    n = len(C)
    s2item = C.imean.var(ddof=1) - C.vom.mean()
    rel = s2item / (s2item + C.tot.mean())
    bs = []
    for _ in range(B):
        s = C.iloc[RNG.integers(0, n, n)]
        bs.append((s.s2C.mean(), s.s2S.mean(), s.tot.mean() / s.dvar.mean(), s.s2C.mean() / s.tot.mean(), s.s2C.mean() - s.s2S.mean()))
    bs = np.array(bs); lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
    print(f"=== {label}: n={n} dialogues (P=critiques, S=scorer calls) ===")
    print(f"  CS single-run variance {C.tot.mean():.3f}  | Direct variance {C.dvar.mean():.3f}  | implied CS reliability {rel:.3f}")
    print(f"  critique component {C.s2C.mean():.3f} [{lo[0]:.3f}, {hi[0]:.3f}]  share {C.s2C.mean() / C.tot.mean():.0%} [{lo[3]:.0%}, {hi[3]:.0%}]")
    print(f"  scorer component   {C.s2S.mean():.3f} [{lo[1]:.3f}, {hi[1]:.3f}]")
    print(f"  critique minus scorer {C.s2C.mean() - C.s2S.mean():+.3f} [{lo[4]:+.3f}, {hi[4]:+.3f}]")
    print(f"  ratio CS / Direct variance {C.tot.mean() / C.dvar.mean():.2f} [{lo[2]:.2f}, {hi[2]:.2f}]")
    print(f"  critiques differ in text for {C.crit_differ.mean():.0%} of dialogues; mean critique length {C.crit_len.mean():.0f} words")
