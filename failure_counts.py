# -*- coding: utf-8 -*-
"""Count failed or unparsed judge calls in every run used by the paper."""
import json, os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def jl(fn):
    return pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, fn), encoding="utf-8") if l.strip()])


def flat(x):
    out = []
    for v in x if isinstance(x, list) else [x]:
        out.extend(flat(v) if isinstance(v, list) else [v])
    return out


rows = []
for fn, label in [("output_pipeline_variance.jsonl", "MTB101 set1 PWS x5"), ("output_direct_variance.jsonl", "MTB101 set1 Direct x5"),
                  ("output_conditions_variance.jsonl", "MTB101 set1 single-call variants x5"), ("output_geval.jsonl", "MTB101 set1 G-Eval x5"),
                  ("output_jointmatched_variance.jsonl", "MTB101 set1 Direct+combined x5"), ("output_stage_fixed.jsonl", "MTB101 stage-fixed")]:
    D = jl(fn)
    bad = (D.status != "ok") | pd.to_numeric(D.get("score"), errors="coerce").isna()
    rows.append((label, len(D), int(bad.sum())))
S = pd.concat([jl("output_failure_sample.jsonl"), jl("output_failure_sample_batch2.jsonl")])
for col, st, lab in [("direct_score", "direct_status", "MTB101 set2 Direct x5"), ("pipeline_score", "pipeline_status", "MTB101 set2 PWS x5")]:
    bad = (S[st] != "ok") | pd.to_numeric(S[col], errors="coerce").isna()
    rows.append((lab, len(S), int(bad.sum())))
for fn, label in [("output_nested_variance.jsonl", "nested gpt-4o-mini MTB101"), ("output_mm_nested_haiku.jsonl", "nested+Direct Haiku MTB101"),
                  ("output_mm_nested_gemini.jsonl", "nested+Direct Gemini MTB101"), ("output_mtbench_nested_4omini.jsonl", "nested+Direct gpt-4o-mini MTB"),
                  ("output_mtbench_nested_haiku.jsonl", "nested+Direct Haiku MTB"), ("output_mtbench_nested_gemini.jsonl", "nested+Direct Gemini MTB")]:
    D = jl(fn)
    kind = D["kind"] if "kind" in D else pd.Series("plan", index=D.index)
    for k in ["direct", "plan"]:
        sub = D[kind == k]
        if not len(sub):
            continue
        if k == "direct":
            n = len(sub); bad = int(((sub.status != "ok") | pd.to_numeric(sub.score, errors="coerce").isna()).sum())
        else:
            calls = [flat(r.full) + flat(r.noworker) if r.status == "ok" else [None] for r in sub.itertuples()]
            n = sum(len(c) for c in calls); bad = sum(sum(v is None for v in c) for c in calls)
        rows.append((f"{label} [{k} {'calls' if k == 'plan' else 'runs'}]", n, bad))
T = pd.DataFrame(rows, columns=["run set", "scored calls", "failed/unparsed"])
T["rate"] = (T["failed/unparsed"] / T["scored calls"]).map("{:.2%}".format)
print(T.to_string(index=False))
print("\nTotal:", int(T["failed/unparsed"].sum()), "of", int(T["scored calls"].sum()))
