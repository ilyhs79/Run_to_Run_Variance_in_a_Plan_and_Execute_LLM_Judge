# -*- coding: utf-8 -*-
"""Six-system MT-Bench leaderboard under repeated complete evaluations
(output_independent_runs.jsonl, runs 1..K where all systems were scored).
Per judge: rank agreement between single runs (Kendall tau), how often the
top system changes, and for each of the 15 system pairs the share of single
runs that order the pair against the K-run average, by the size of the gap."""
import json, os, itertools
import numpy as np, pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
K = int(os.environ.get("K", "5"))
D = pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, "output_independent_runs.jsonl"), encoding="utf-8") if l.strip()])
D = D[(D.status == "ok") & (D.run <= K)].copy()
D["score"] = pd.to_numeric(D.score, errors="coerce")
D["system"] = D.item_id.str.split("_").str[1]
systems = sorted(D.system.unique())
print("systems:", systems, "| runs:", K)

for judge in ["pws", "direct"]:
    J = D[D.judge == judge]
    M = J.groupby(["run", "system"]).score.mean().unstack()  # runs x systems
    M = M.dropna(axis=1)
    avg = M.mean()
    print(f"\n=== {judge.upper()} ===  K-run average by system: " + ", ".join(f"{s} {v:.2f}" for s, v in avg.sort_values(ascending=False).items()))
    taus = [stats.kendalltau(M.loc[a], M.loc[b]).statistic for a, b in itertools.combinations(M.index, 2)]
    top = M.idxmax(axis=1)
    print(f"  Kendall tau between single runs: mean {np.mean(taus):.2f}, min {np.min(taus):.2f} ({len(taus)} run pairs)")
    print(f"  top system per run: {top.tolist()}  | runs whose top differs from the average's top: {np.mean(top != avg.idxmax()):.0%}")
    rows = []
    for a, b in itertools.combinations(M.columns, 2):
        gap = avg[a] - avg[b]
        flips = np.mean(np.sign(M[a] - M[b]) != np.sign(gap)) if gap != 0 else np.nan
        rows.append((f"{a} vs {b}", abs(gap), flips))
    R = pd.DataFrame(rows, columns=["pair", "abs_gap", "flip"]).sort_values("abs_gap")
    print(R.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 10)]:
        s = R[(R.abs_gap >= lo) & (R.abs_gap < hi)]
        if len(s):
            print(f"  pairs with gap in [{lo}, {hi}): n={len(s)}, mean flip rate {s.flip.mean():.0%}")

    # two-level bootstrap: resample questions (all systems together) and runs, recompute everything
    J2 = J.copy(); J2["q"] = J2.item_id.str.extract(r"M(\d+)_")[0].astype(int)
    A = J2.pivot_table(index="q", columns=["run", "system"], values="score")  # questions x (run, system)
    runs = sorted(J2.run.unique())

    def stats_of(Aq, rs):
        Mb = pd.DataFrame([Aq[r].mean() for r in rs]).dropna(axis=1)  # runs x systems (duplicates kept)
        Mb.index = range(len(rs))
        ab = Mb.mean()
        tau = np.mean([stats.kendalltau(Mb.loc[a], Mb.loc[b]).statistic
                       for a, b in itertools.combinations(Mb.index, 2) if rs[a] != rs[b]])
        fl = np.mean([np.mean(np.sign(Mb[a] - Mb[b]) != np.sign(ab[a] - ab[b]))
                      for a, b in itertools.combinations(Mb.columns, 2) if ab[a] != ab[b]])
        top = np.mean(Mb.idxmax(axis=1) != ab.idxmax())
        return np.array([tau, fl, top])

    point = stats_of(A, runs)
    rng = np.random.default_rng(71)
    qidx = A.index.to_numpy()
    for label, resample_runs in [("questions", False), ("questions+runs", True)]:
        bs = []
        for _ in range(2000):
            Aq = A.loc[rng.choice(qidx, len(qidx), replace=True)]
            rs = list(rng.choice(runs, len(runs), replace=True)) if resample_runs else runs
            if resample_runs and len(set(rs)) < 2:
                continue
            bs.append(stats_of(Aq, rs))
        bs = np.array(bs); lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        for j, name in enumerate(["mean tau", "pair flip rate", "top changes"]):
            basic = (2 * point[j] - hi[j], 2 * point[j] - lo[j])
            print(f"  [{label:14s}] {name:15s} point {point[j]:.2f}  boot mean {bs[:, j].mean():.2f}  "
                  f"percentile [{lo[j]:.2f}, {hi[j]:.2f}]  basic [{basic[0]:.2f}, {basic[1]:.2f}]")
