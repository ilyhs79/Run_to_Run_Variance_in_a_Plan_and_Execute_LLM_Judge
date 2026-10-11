# -*- coding: utf-8 -*-
"""Did Direct's MT-Bench system difference change between the afternoon session
(5 Direct runs inside the nested experiment) and the evening session (10
independent runs)? Paired by question."""
import json, os
import numpy as np, pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))


def load(fn, kind_col, kind_val):
    D = pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, fn), encoding="utf-8") if l.strip()])
    D = D[(D[kind_col] == kind_val) & (D.status == "ok")].copy()
    D["score"] = pd.to_numeric(D.score, errors="coerce")
    return D


def qdiff(D):
    m = D.groupby("item_id").score.mean()
    t = pd.DataFrame({"q": m.index.str.extract(r"M(\d+)_")[0].values,
                      "s": np.where(m.index.str.endswith("_gpt4o"), "g", "n"), "m": m.values})
    t = t.pivot(index="q", columns="s", values="m")
    return (t["g"] - t["n"]).dropna()


A = qdiff(load("output_mtbench_nested_4omini.jsonl", "kind", "direct"))
B = qdiff(load("output_independent_runs.jsonl", "judge", "direct"))
for lab, v in [("afternoon, 5 runs", A), ("evening, 10 runs", B)]:
    print(f"{lab:18s} system diff {v.mean():+.3f}  paired p={stats.ttest_1samp(v, 0).pvalue:.4f}  n={len(v)}")
ch = (B - A).dropna()
print(f"change evening - afternoon {ch.mean():+.3f}  p={stats.ttest_1samp(ch, 0).pvalue:.4f}")
