# -*- coding: utf-8 -*-
"""Isolate the SOLVER. The stored Our_Approach_Processed_Plan is the exact user
prompt the notebook fed to the solver (solver_common + conversation + ORIGINAL
worker-resolved evidence). Re-run only the solver on it. If this reproduces
~0.887, the solver is faithful and the live gap is entirely in planner+worker
(drift). If it is lower, the solver itself drifts."""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
ANSWER_KEY = os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")
PROMPTS = os.path.join(BASE, "pipeline_prompts.json")
OUT = os.path.join(BASE, "output_diag_storedprocessed_solver.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"  # pin exact snapshot the paper names
write_lock = threading.Lock()
openai.api_key = os.environ.get("OPENAI_API_KEY")
SYS = json.load(open(PROMPTS, encoding="utf-8"))["System_prompt"]
RATING_RE = re.compile(r"\[\[\s*([1-9])\s*\]\]")
FB = re.compile(r"[Rr]ating[:\s\*]*\[?\[?\s*([1-9])")


def call(system, user, retries=5):
    last = None
    for a in range(retries):
        try:
            r = openai.chat.completions.create(model=MODEL, temperature=0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content or ""
        except Exception as e:
            last = e; time.sleep(min(2 ** a * 3, 30))
    raise last


def parse(t):
    m = RATING_RE.findall(t or "")
    if m: return int(m[-1])
    m = FB.findall(t or "")
    return int(m[-1]) if m else None


def main():
    key = pd.read_csv(ANSWER_KEY)
    src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
    si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
    done = set(json.loads(l)["item_id"] for l in open(OUT, encoding="utf-8")) if os.path.exists(OUT) else set()
    todo = [k for _, k in key.iterrows() if k["item_id"] not in done]

    def work(kr):
        rec = {"item_id": kr["item_id"], "task": kr["task"], "id": int(kr["id"])}
        try:
            row = si[(kr["task"], int(kr["id"]))]
            processed = str(row["Our_Approach_Processed_Plan"])
            out = call(SYS, processed)
            rec["score"] = parse(out); rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:150]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(work, k) for k in todo]):
            f.result()
    print("done")


if __name__ == "__main__":
    main()
