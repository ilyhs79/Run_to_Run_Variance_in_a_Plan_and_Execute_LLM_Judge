# -*- coding: utf-8 -*-
"""Share of the 100 dialogues whose Solver score, rerun on the exact stored
Solver prompt (plan + evidence) of the original run, matches the stored score."""
import json, os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]
src = pd.concat([pd.read_csv(os.path.join(BASE, f)) for f in SOURCE_FILES], ignore_index=True)
src = src.drop_duplicates(subset=["task", "id"], keep="first")[["task", "id", "Our_Approach_Score"]]
R = pd.DataFrame([json.loads(l) for l in open(os.path.join(BASE, "output_diag_storedprocessed_solver.jsonl"), encoding="utf-8") if l.strip()])
R = R[R.status == "ok"].merge(src, on=["task", "id"])
R["Our_Approach_Score"] = pd.to_numeric(R["Our_Approach_Score"], errors="coerce")
m = R.dropna(subset=["score", "Our_Approach_Score"])
print(f"dialogues={len(m)}  rerun score == stored score: {(m.score == m.Our_Approach_Score).mean():.1%}")
