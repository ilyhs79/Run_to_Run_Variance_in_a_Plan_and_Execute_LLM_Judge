# -*- coding: utf-8 -*-
"""Paired bootstrap of the PWS-minus-Direct difference in leaderboard stability
(Kendall tau between single runs, pair flip rate, top-system changes). The same
resampled questions and runs are used for both judges; run k of the two judges
was executed in the same batch, so runs are paired."""
import json, os, itertools
import numpy as np, pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
K = int(os.environ.get("K", "10"))
D = pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, "output_independent_runs.jsonl"), encoding="utf-8") if l.strip()])
D = D[(D.status == "ok") & (D.run <= K)].copy()
D["score"] = pd.to_numeric(D.score, errors="coerce")
D["system"] = D.item_id.str.split("_").str[1]
D["q"] = D.item_id.str.extract(r"M(\d+)_")[0].astype(int)
A = {j: D[D.judge == j].pivot_table(index="q", columns=["run", "system"], values="score") for j in ["pws", "direct"]}
runs = list(range(1, K + 1))
qidx = A["pws"].index.intersection(A["direct"].index).to_numpy()


def stats_of(Aq, rs):
    Mb = pd.DataFrame([Aq[r].mean() for r in rs]).dropna(axis=1)
    Mb.index = range(len(rs))
    ab = Mb.mean()
    tau = np.mean([stats.kendalltau(Mb.loc[a], Mb.loc[b]).statistic
                   for a, b in itertools.combinations(Mb.index, 2) if rs[a] != rs[b]])
    fl = np.mean([np.mean(np.sign(Mb[a] - Mb[b]) != np.sign(ab[a] - ab[b]))
                  for a, b in itertools.combinations(Mb.columns, 2) if ab[a] != ab[b]])
    top = np.mean(Mb.idxmax(axis=1) != ab.idxmax())
    return np.array([tau, fl, top])


point = stats_of(A["pws"].loc[qidx], runs) - stats_of(A["direct"].loc[qidx], runs)
rng = np.random.default_rng(73)
bs = []
while len(bs) < 2000:
    q = rng.choice(qidx, len(qidx), replace=True)
    rs = list(rng.choice(runs, K, replace=True))
    if len(set(rs)) < 2:
        continue
    bs.append(stats_of(A["pws"].loc[q], rs) - stats_of(A["direct"].loc[q], rs))
bs = np.array(bs)
for j, name in enumerate(["Kendall tau", "pair flip rate", "top changes"]):
    lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
    side = np.mean(bs[:, j] < 0) if j == 0 else np.mean(bs[:, j] > 0)
    print(f"PWS - Direct {name:15s} point {point[j]:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"share of bootstrap samples in the observed direction {side:.0%}")
