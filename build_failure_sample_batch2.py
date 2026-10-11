# -*- coding: utf-8 -*-
"""Gate 1 expansion (batch 2): draw an additional stratified sample to raise the
failure-inclusive human study to ~300 dialogues for statistical power. Excludes
the original 100 AND batch-1 (130). 13 per category x 13 categories = 169. Item
ids continue from F131."""
import csv, os
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
ANSWER_KEY = os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")
BATCH1 = os.path.join(BASE, "failure_sample_manifest.csv")
PER_CAT = 13
SEED = 20261012 + 2

src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
used = set((r["task"], int(r["id"])) for _, r in pd.read_csv(ANSWER_KEY).iterrows())
used |= set((r["task"], int(r["id"])) for _, r in pd.read_csv(BATCH1).iterrows())
src = src[~src.apply(lambda r: (r["task"], int(r["id"])) in used, axis=1)]

rng = np.random.default_rng(SEED)
picks = []
for task in sorted(src["task"].unique()):
    pool = src[src["task"] == task]
    n = min(PER_CAT, len(pool))
    picks.extend(rng.choice(pool.index.to_numpy(), size=n, replace=False).tolist())
sample = src.loc[picks].sort_values(["task", "id"]).reset_index(drop=True)
sample.insert(0, "item_id", [f"F{131+i:03d}" for i in range(len(sample))])

order = rng.permutation(len(sample))
shuf = sample.iloc[order].reset_index(drop=True)


def dialogue_text(row):
    n = int(row["number_of_turns"])
    return "\n\n".join(f"[Turn {t}]\nUser: {row[f'question_{t}']}\nAssistant: {row[f'answer_{t}']}" for t in range(1, n + 1))


sample[["item_id", "task", "id", "number_of_turns"]].to_csv(os.path.join(BASE, "failure_sample_manifest_batch2.csv"), index=False)
with open(os.path.join(BASE, "failure_sample_annotation_batch2.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "dialogue_text", "dialogue_text_ko", "your_score_1_to_9"])
    for _, r in shuf.iterrows():
        w.writerow([r["item_id"], dialogue_text(r), "", ""])

print(f"batch2: {len(sample)} dialogues; ids {sample['item_id'].iloc[0]}..{sample['item_id'].iloc[-1]}")
print(sample.groupby("task").size().to_dict())
