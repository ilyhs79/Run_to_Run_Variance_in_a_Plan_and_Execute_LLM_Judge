# -*- coding: utf-8 -*-
"""Execution-variance characterization. Run the FULL pipeline (all gpt-4o-mini,
exact notebook prompts, temperature 0, matching Our_Approach_Model in the
released data) end to end K times on the 100 human-rated items. Report per-run
r and MAE vs the human consensus, then mean +/- std. This quantifies the
gpt-4o-mini nondeterminism that makes the single-run 0.887 non-reproducible and
answers the reviewers' "run once / no execution variance" critique.

K from N_RUNS env (default 5). Key from OPENAI_API_KEY.
"""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
ANSWER_KEY = os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")
PROMPTS = os.path.join(BASE, "pipeline_prompts.json")
OUT = os.path.join(BASE, "output_pipeline_variance.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"
N_RUNS = int(os.environ.get("N_RUNS", "5"))
MAX_WORKERS = 6
write_lock = threading.Lock()
openai.api_key = os.environ.get("OPENAI_API_KEY")

P = json.load(open(PROMPTS, encoding="utf-8"))
SYS, SYS_EXEC = P["System_prompt"], P["System_prompt_plan_execution"]
SOLVER_COMMON, PLANNER_COMMON, CATS = P["User_prompt_solver_common"], P["User_prompt_planner_common"], P["planner_categories"]
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


def conv_block(row):
    n = int(row["number_of_turns"])
    parts = ["<|The Start of Assistant A's Conversation with User|>"]
    for t in range(1, n + 1):
        parts.append(f"### User:\n{row[f'question_{t}']}\n### Assistant A:\n{row[f'answer_{t}']}\n")
    parts.append('<|The End of Assistant A\'s Conversation with User|>"')
    return "\n".join(parts)


def parse(t):
    m = RATING_RE.findall(t or "")
    if m: return int(m[-1])
    m = FB.findall(t or "")
    return int(m[-1]) if m else None


def run_pipeline(task, conv):
    plan = call(SYS, PLANNER_COMMON + CATS[task] + "\n" + conv)
    calls = re.findall(r"#(E\d+)\s*=\s*LLM\[(.*?)\]", plan, re.DOTALL)
    processed = plan
    for var, instr in calls:
        res = call(SYS_EXEC, instr)
        processed = re.sub(re.escape(f"#{var} = LLM[{instr}]"), f'#{var} = """{res}"""', processed, count=1)
    return parse(call(SYS, SOLVER_COMMON + conv + "\n\n" + processed))


def load_done():
    d = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); d.add((r["run"], r["item_id"]))
    return d


def main():
    key = pd.read_csv(ANSWER_KEY)
    src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
    si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
    done = load_done()
    jobs = [(run, k) for run in range(1, N_RUNS + 1) for _, k in key.iterrows() if (run, k["item_id"]) not in done]
    print(f"runs={N_RUNS} jobs={len(jobs)}")

    def work(job):
        run, kr = job
        rec = {"run": run, "item_id": kr["item_id"], "task": kr["task"], "id": int(kr["id"])}
        try:
            row = si[(kr["task"], int(kr["id"]))]
            rec["score"] = run_pipeline(kr["task"], conv_block(row)); rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:150]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result(); n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}")
    print("done")


if __name__ == "__main__":
    main()
