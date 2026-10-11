# -*- coding: utf-8 -*-
"""Dollar cost of the averaging designs in Table 6, from measured gpt-4o-mini
token usage per call type on MT-Bench-101 (output_token_accounting.json) and
list prices of USD 0.15 / 0.60 per million input / output tokens."""
import json, os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
U = pd.DataFrame(json.load(open(os.path.join(BASE, "output_token_accounting.json"))))
U = U[U.dataset == "mtb101"]


def per1000(kind):
    u = U[U.kind == kind]
    return (u["in"].mean() * 0.15e-6 + u["out"].mean() * 0.60e-6) * 1000


pl, wk, so, di = per1000("planner"), per1000("worker"), per1000("solver"), per1000("direct")
NE = 3.4  # mean evidence requests per plan in the nested runs
print(f"per call, USD per 1,000 dialogues: planner {pl:.3f}, worker {wk:.4f}, solver {so:.3f}, direct {di:.3f}")
for lab, p, w, s in [("1 full run", 1, 1, 1), ("1 plan, 1 Worker, 5 Solver", 1, 1, 5), ("1 plan, 5 Worker+Solver", 1, 5, 1),
                     ("3 plans", 3, 1, 1), ("2 plans x 2 Workers", 2, 2, 1), ("5 full runs", 5, 1, 1)]:
    print(f"  {lab:28s} USD {p * (pl + w * (NE * wk + s * so)):.2f} per 1,000 dialogues")
print(f"  {'Direct, 1 call':28s} USD {di:.2f} per 1,000 dialogues")
