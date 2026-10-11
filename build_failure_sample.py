# -*- coding: utf-8 -*-
"""Gate 1 prep: draw a FAILURE-INCLUSIVE human-evaluation sample.

Unlike the original 100 (drawn from the four-way-complete pool, i.e. conditioned
on every judge succeeding), this sample is drawn BEFORE any execution filtering:
a stratified random draw across all 13 MT-Bench-101 categories (Mathematical
Reasoning included), with the already-annotated 100 items excluded. The pipeline
is run on it separately (run_failure_sample.py) and failures are kept, not
dropped, so the study can report coverage and an end-to-end fallback result.

Outputs:
  failure_sample_manifest.csv    - item_id, task, id, number_of_turns (for the runs)
  failure_sample_annotation.csv  - blinded sheet for annotators (no scores)
"""
import csv, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
ANSWER_KEY = os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")
PER_CAT = 10
SEED = 20261012  # ARR Oct 2026 deadline, for reproducibility

src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
already = set((r["task"], int(r["id"])) for _, r in pd.read_csv(ANSWER_KEY).iterrows())
src = src[~src.apply(lambda r: (r["task"], int(r["id"])) in already, axis=1)]

rng = np.random.default_rng(SEED)
picks = []
for task in sorted(src["task"].unique()):
    pool = src[src["task"] == task]
    n = min(PER_CAT, len(pool))
    idx = rng.choice(pool.index.to_numpy(), size=n, replace=False)
    picks.extend(idx.tolist())
sample = src.loc[picks].reset_index(drop=True)
sample = sample.sort_values(["task", "id"]).reset_index(drop=True)
sample.insert(0, "item_id", [f"F{i+1:03d}" for i in range(len(sample))])

# blind the annotator order (shuffle so category is not obvious in sequence)
order = rng.permutation(len(sample))
sample_shuf = sample.iloc[order].reset_index(drop=True)


def dialogue_text(row):
    n = int(row["number_of_turns"])
    parts = []
    for t in range(1, n + 1):
        parts.append(f"[Turn {t}]\nUser: {row[f'question_{t}']}\nAssistant: {row[f'answer_{t}']}")
    return "\n\n".join(parts)


# manifest for the pipeline runs
sample[["item_id", "task", "id", "number_of_turns"]].to_csv(
    os.path.join(BASE, "failure_sample_manifest.csv"), index=False)

# blinded annotation sheet (no scores, no category shown, shuffled)
with open(os.path.join(BASE, "failure_sample_annotation.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "dialogue_text", "dialogue_text_ko", "your_score_1_to_9"])
    for _, r in sample_shuf.iterrows():
        w.writerow([r["item_id"], dialogue_text(r), "", ""])

print(f"sampled {len(sample)} dialogues across {sample['task'].nunique()} categories (excluded the original 100)")
print(sample.groupby("task").size().to_dict())
print("wrote failure_sample_manifest.csv and failure_sample_annotation.csv")
