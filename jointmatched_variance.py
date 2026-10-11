# -*- coding: utf-8 -*-
"""K-run version of the jointly-matched single-call control (reviewer's 'joint
contribution' concern): one undecomposed call given category guidance + worked
example + turn-by-turn CoT + comparable budget. 5 runs on the 100 human items."""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("jm", os.path.join(BASE, "direct_jointmatched_baseline.py"))
# We reuse build_prompt/conv_block/parse_rating/ChatGPT from the base script by import,
# but that script runs main() on import guard, so replicate the pieces we need instead.
import openai, re
openai.api_key = os.environ.get("OPENAI_API_KEY")
P = json.load(open(os.path.join(BASE, "pipeline_prompts.json"), encoding="utf-8"))
MODEL = "gpt-4o-mini-2024-07-18"
N_RUNS = int(os.environ.get("N_RUNS", "5"))
OUT = os.path.join(BASE, "output_jointmatched_variance.jsonl")
ANSWER_KEY = os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
write_lock = threading.Lock()

# pull the exact builders from the base module's source without running main
src = open(os.path.join(BASE, "direct_jointmatched_baseline.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(BASE, "direct_jointmatched_baseline.py"), "__name__": "jm_base"}
exec(src.split("def main(")[0], ns)  # defines CATEGORY_RUBRIC, DEMO, COT, build_prompt, conv_block, parse_rating, ChatGPT
build_prompt = ns["build_prompt"]; conv_block = ns["conv_block"]; parse_rating = ns["parse_rating"]; ChatGPT = ns["ChatGPT"]


def load_done():
    d = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); d.add((r["run"], r["item_id"]))
    return d


def main():
    key = pd.read_csv(ANSWER_KEY)
    src2 = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
    si = {(r["task"], int(r["id"])): r for _, r in src2.iterrows()}
    done = load_done()
    jobs = [(run, k) for run in range(1, N_RUNS + 1) for _, k in key.iterrows() if (run, k["item_id"]) not in done]
    print(f"jobs={len(jobs)}")

    def work(job):
        run, kr = job
        rec = {"run": run, "item_id": kr["item_id"], "task": kr["task"], "id": int(kr["id"])}
        try:
            row = si[(kr["task"], int(kr["id"]))]
            rec["score"] = parse_rating(ChatGPT(build_prompt(kr["task"], conv_block(row)))); rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:150]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result()
    print("done")


if __name__ == "__main__":
    main()
