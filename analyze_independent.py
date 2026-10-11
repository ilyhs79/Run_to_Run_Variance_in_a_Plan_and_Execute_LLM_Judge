# -*- coding: utf-8 -*-
"""Analysis of independent complete MT-Bench evaluations (output_independent_runs.jsonl).
(a) empirical single-run within-dialogue variance vs the nested-model prediction;
(b) a run effect shared by all dialogues in a run (two-way random-effects);
(c) the two-system comparison in each real run vs the all-runs average;
(d) observed spread of the system difference across runs vs the spread that
independent per-dialogue noise alone would produce."""
import json, os
import numpy as np, pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(61)
PRED = {"pws": 0.663, "direct": 0.214}  # nested-model single-run variance and Direct 5-run variance, MT-Bench
SIM_RUN_SD = {"pws": 0.118, "direct": 0.067}  # simulated single-run SD of the system difference (Table 5)

D = pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, "output_independent_runs.jsonl"), encoding="utf-8") if l.strip()])
D = D[D.status == "ok"].copy(); D["score"] = pd.to_numeric(D.score, errors="coerce")
D["system"] = np.where(D.item_id.str.endswith("_gpt4o"), "gpt4o", "nano")
D["q"] = D.item_id.str.extract(r"M(\d+)_")[0].astype(int)
print("runs:", sorted(D.run.unique()), "| failures/unparsed:", int(D.score.isna().sum()))

for judge in ["pws", "direct"]:
    J = D[D.judge == judge]
    X = J.pivot_table(index="item_id", columns="run", values="score").dropna()
    n, k = X.shape
    x = X.to_numpy(float)
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = (((x - gm) ** 2).sum() - k * ((x.mean(1) - gm) ** 2).sum() - n * ((x.mean(0) - gm) ** 2).sum()) / ((n - 1) * (k - 1))
    F = msc / mse; pF = 1 - stats.f.cdf(F, k - 1, (n - 1) * (k - 1))
    s2_run = max((msc - mse) / n, 0.0)
    wv = x.var(axis=1, ddof=1).mean()
    bs = [x[RNG.integers(0, n, n)].var(axis=1, ddof=1).mean() for _ in range(4000)]
    print(f"\n=== {judge.upper()}: {n} dialogues x {k} runs ===")
    print(f"  (a) empirical within-dialogue variance {wv:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]  vs prediction {PRED[judge]:.3f}")
    print(f"  (b) run effect: F={F:.2f}, p={pF:.3g}; run variance component {s2_run:.4f} (residual {mse:.3f}); run means {np.round(x.mean(0), 3).tolist()}")

    # (c) system comparison per real run, paired by question
    W = J.pivot_table(index=["q", "task"], columns=["run", "system"], values="score")
    per_run = []
    for r in sorted(J.run.unique()):
        a, b = W[(r, "gpt4o")], W[(r, "nano")]
        ok = a.notna() & b.notna()
        per_run.append(((a - b)[ok].mean(), stats.ttest_rel(a[ok], b[ok]).pvalue, (a - b)[ok].groupby(level="task").mean()))
    Ds = np.array([p[0] for p in per_run]); ps = np.array([p[1] for p in per_run])
    avg = W.T.groupby(level="system").mean().T
    d_all = (avg["gpt4o"] - avg["nano"]).dropna()
    cat_all = d_all.groupby(level="task").mean()
    p_all = stats.ttest_rel(avg["gpt4o"].loc[d_all.index], avg["nano"].loc[d_all.index]).pvalue
    clear = cat_all[cat_all.abs() >= 0.25]
    flips = np.mean([np.mean(np.sign(p[2][clear.index]) != np.sign(clear)) for p in per_run]) if len(clear) else float("nan")
    print(f"  (c) system diff per run: mean {Ds.mean():+.3f}, SD {Ds.std(ddof=1):.3f} (simulated {SIM_RUN_SD[judge]:.3f}), range [{Ds.min():+.3f}, {Ds.max():+.3f}]")
    print(f"      all-runs average diff {d_all.mean():+.3f}, p={p_all:.3g}; runs with p<.05: {np.mean(ps < .05):.0%}; "
          f"category flips vs average ({len(clear)} clear categories): {flips:.0%}")

    # (d) spread expected from independent per-dialogue noise alone
    dev = {}
    for (q, t), row in W.iterrows():
        a = np.array([row[(r, "gpt4o")] for r in sorted(J.run.unique())]); b = np.array([row[(r, "nano")] for r in sorted(J.run.unique())])
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        dev[q] = (a - b)
    M = np.array(list(dev.values()))  # questions x runs
    var_indep = M.var(axis=1, ddof=1).sum() / len(M) ** 2  # variance of the mean difference if dialogues varied independently
    print(f"  (d) observed SD of system diff across runs {Ds.std(ddof=1):.3f} vs independent-noise expectation {np.sqrt(var_indep):.3f}")
