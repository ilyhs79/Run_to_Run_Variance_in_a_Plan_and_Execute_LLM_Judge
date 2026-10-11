# -*- coding: utf-8 -*-
"""System-level consequences of run-to-run variance on MT-Bench. Two systems
(GPT-4o and GPT-5.4-nano answers) are compared on the same questions. A
"single run" of a judge draws one realization per dialogue from the repeated
runs (Direct: one of its runs; PWS: one plan, one Worker draw, one Solver
call). Against the all-runs average we report, per judge and backend:
the spread of the single-run system difference, how often a single run picks
the other winner, how often its paired test verdict (p<.05) differs, and how
often per-category winners flip."""
import json, os, sys
import numpy as np, pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(31)
NSIM = int(os.environ.get("NSIM", "4000"))
MIN_CAT_DIFF = 0.25


def jl(fn):
    p = os.path.join(BASE, fn)
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]) if os.path.exists(p) else None


def load(tag):
    D = jl(f"output_mtbench_nested_{tag}.jsonl")
    if D is None:
        return None
    D = D[D.status == "ok"]
    meta = pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, "data", "mtbench_dialogues.jsonl"), encoding="utf-8")])
    meta["system"] = np.where(meta.item_id.str.endswith("_gpt4o"), "gpt4o", "nano")
    meta = meta.set_index("item_id")[["question_id", "task", "system"]]
    direct, pws = {}, {}
    for iid, g in D[D.kind == "direct"].groupby("item_id"):
        v = pd.to_numeric(g.sort_values("idx").score, errors="coerce").to_numpy(float)
        if len(v) and not np.isnan(v).any():
            direct[iid] = v
    for iid, g in D[D.kind == "plan"].groupby("item_id"):
        try:
            y = np.array(g.sort_values("idx").full.tolist(), dtype=float)
        except (ValueError, TypeError):
            continue
        if y.ndim == 3 and not np.isnan(y).any():
            pws[iid] = y.reshape(-1)  # every (plan, worker, solver) path is one single-run realization
    return meta, direct, pws


def analyze(name, meta, draws, paired):
    items = [i for i in draws if i in meta.index]
    m = meta.loc[items].copy()
    if paired:
        q2 = m.groupby("question_id").system.nunique()
        keep = set(q2[q2 == 2].index)
        m = m[m.question_id.isin(keep)]
    items = list(m.index)
    if len(items) < 10:
        print(f"  {name}: too few complete items ({len(items)})"); return None
    A = [draws[i] for i in items]
    ref = np.array([a.mean() for a in A])
    sysv = m.system.to_numpy(); qid = m.question_id.to_numpy(); task = m.task.to_numpy()

    def diff_and_p(score):
        s = pd.DataFrame({"q": qid, "sys": sysv, "task": task, "x": score})
        if paired:
            w = s.pivot_table(index=["q", "task"], columns="sys", values="x").dropna()
            d = w["gpt4o"] - w["nano"]
            p = stats.ttest_rel(w["gpt4o"], w["nano"]).pvalue if d.std() > 0 else 1.0
            cat = d.groupby(level="task").mean()
        else:
            a = s[s.sys == "gpt4o"].x; b = s[s.sys == "nano"].x
            d = pd.Series([a.mean() - b.mean()])
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            cat = s.groupby(["task", "sys"]).x.mean().unstack().pipe(lambda t: t["gpt4o"] - t["nano"])
        return d.mean(), p, cat

    D_ref, p_ref, cat_ref = diff_and_p(ref)
    sims = []
    for _ in range(NSIM):
        one = np.array([a[RNG.integers(len(a))] for a in A])
        sims.append(diff_and_p(one))
    Ds = np.array([x[0] for x in sims]); ps = np.array([x[1] for x in sims])
    cats = pd.DataFrame([x[2] for x in sims])
    # a "winner" is only meaningful where the all-runs-average difference is clear
    cat_flip = pd.Series({c: float(np.mean(np.sign(cats[c]) != np.sign(cat_ref[c])))
                          for c in cats.columns if c in cat_ref.index and abs(cat_ref[c]) >= MIN_CAT_DIFF})
    # sampling spread of the all-runs-average difference over questions, for scale
    if paired:
        s = pd.DataFrame({"q": qid, "sys": sysv, "x": ref}).pivot_table(index="q", columns="sys", values="x").dropna()
        dq = (s["gpt4o"] - s["nano"]).to_numpy()
        boot = [dq[RNG.integers(0, len(dq), len(dq))].mean() for _ in range(4000)]
    else:
        a = ref[sysv == "gpt4o"]; b = ref[sysv == "nano"]
        boot = [a[RNG.integers(0, len(a), len(a))].mean() - b[RNG.integers(0, len(b), len(b))].mean() for _ in range(4000)]
    print(f"  {name:28s} n_items={len(items):3d}  diff(all-runs avg)={D_ref:+.3f} p={p_ref:.3g}")
    print(f"  {'':28s} single run: diff mean {Ds.mean():+.3f}, SD {Ds.std():.3f}, range [{Ds.min():+.3f}, {Ds.max():+.3f}]"
          f"  | question-sampling SD {np.nanstd(boot):.3f}  | run SD / question SD = {Ds.std() / np.nanstd(boot):.2f}")
    print(f"  {'':28s} P(single run picks other winner) = {np.mean(np.sign(Ds) != np.sign(D_ref)):.1%}"
          f"  | P(p<.05) = {np.mean(ps < .05):.1%} (all-runs avg: {'sig' if p_ref < .05 else 'ns'})"
          f"  | P(verdict differs) = {np.mean((ps < .05) != (p_ref < .05)):.1%}")
    if len(cat_flip):
        print(f"  {'':28s} per-category winner flips ({len(cat_flip)} categories with |avg diff| >= {MIN_CAT_DIFF}): "
              f"mean {cat_flip.mean():.1%}; " + ", ".join(f"{k} {v:.0%}" for k, v in cat_flip.sort_values(ascending=False).items()))
    else:
        print(f"  {'':28s} per-category winner flips: no category with |avg diff| >= {MIN_CAT_DIFF}")
    return dict(Ds=Ds, D_ref=D_ref)


for tag, label, paired in [("4omini", "GPT-4o-mini", True), ("haiku", "Claude Haiku 4.5 (unpaired)", False),
                           ("gemini", "Gemini 3.1 Flash-Lite", True)]:
    L = load(tag)
    print(f"=== {label} ===")
    if L is None:
        print("  no data"); continue
    meta, direct, pws = L
    analyze("Direct", meta, direct, paired)
    analyze("PWS", meta, pws, paired)
