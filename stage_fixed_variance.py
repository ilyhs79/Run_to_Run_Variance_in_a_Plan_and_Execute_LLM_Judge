# -*- coding: utf-8 -*-
"""Stage-fixed variance. For each item, freeze stored intermediates from the
original run and re-execute only the downstream stages K times:
  S  : stored solver prompt (plan + stored evidence) fixed -> rerun Solver
  WS : stored plan fixed -> rerun Worker (every #E) + Solver
Full-pipeline (PWS) and Direct K-run data already exist. Items: the original 100
and the 299 new items. gpt-4o-mini snapshot, temperature 0, as in the paper."""
import json, os, re, time, threading, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
OUT = os.path.join(BASE, "output_stage_fixed.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"
N_RUNS = int(os.environ.get("N_RUNS", "5"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
write_lock = threading.Lock()
openai.api_key = os.environ.get("OPENAI_API_KEY")

P = json.load(open(os.path.join(BASE, "pipeline_prompts.json"), encoding="utf-8"))
SYS, SYS_EXEC, SOLVER_COMMON = P["System_prompt"], P["System_prompt_plan_execution"], P["User_prompt_solver_common"]
RR = re.compile(r"\[\[\s*([1-9])\s*\]\]"); FB = re.compile(r"[Rr]ating[:\s\*]*\[?\[?\s*([1-9])")


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
    m = RR.findall(t or "")
    if m: return int(m[-1])
    m = FB.findall(t or "")
    return int(m[-1]) if m else None


def conv_block(row):
    n = int(row["number_of_turns"])
    parts = ["<|The Start of Assistant A's Conversation with User|>"]
    for t in range(1, n + 1):
        parts.append(f"### User:\n{row[f'question_{t}']}\n### Assistant A:\n{row[f'answer_{t}']}\n")
    parts.append('<|The End of Assistant A\'s Conversation with User|>"')
    return "\n".join(parts)


def h(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


PLAN_TAG = "<|The Start of Evaluation Plan|>"


def split_stored(row):
    pp = str(row["Our_Approach_Processed_Plan"])
    i = pp.find(PLAN_TAG)
    return pp[:i], pp[i:]


def run_S(row):
    return parse(call(SYS, str(row["Our_Approach_Processed_Plan"]))), {}


def run_S2(row):
    _, stored_plan = split_stored(row)
    return parse(call(SYS, SOLVER_COMMON + conv_block(row) + "\n\n" + stored_plan)), {}


def run_WS(row):
    prefix, _ = split_stored(row)
    plan = str(row["Our_Approach_Initial_Plan"])
    calls = re.findall(r"#(E\d+)\s*=\s*LLM\[(.*?)\]", plan, re.DOTALL)
    processed, ev = plan, {}
    for var, instr in calls:
        res = call(SYS_EXEC, instr)
        ev[var] = h(res)
        processed = re.sub(re.escape(f"#{var} = LLM[{instr}]"), lambda m, r=f'#{var} = """{res}"""': r, processed, count=1)
    return parse(call(SYS, prefix + processed)), {"ev_hash": ev, "n_evidence": len(calls)}


RUNNERS = {"S": run_S, "S2": run_S2, "WS": run_WS}


def main():
    src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
    si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
    items = pd.concat([
        pd.read_csv(os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv"))[["item_id", "task", "id"]].assign(set="orig100"),
        pd.read_csv(os.path.join(BASE, "failure_sample_manifest.csv"))[["item_id", "task", "id"]].assign(set="new299"),
        pd.read_csv(os.path.join(BASE, "failure_sample_manifest_batch2.csv"))[["item_id", "task", "id"]].assign(set="new299")], ignore_index=True)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); done.add((d["cond"], d["run"], d["item_id"]))
    jobs = []
    for _, it in items.iterrows():
        row = si[(it["task"], int(it["id"]))]
        stored = pd.notna(row["Our_Approach_Processed_Plan"]) and PLAN_TAG in str(row["Our_Approach_Processed_Plan"])
        if not stored:
            continue
        for run in range(1, N_RUNS + 1):
            for cond in RUNNERS:
                if (cond, run, it["item_id"]) not in done:
                    jobs.append((cond, run, it, row))
    print(f"model={MODEL} runs={N_RUNS} jobs={len(jobs)}")

    def work(job):
        cond, run, it, row = job
        rec = {"cond": cond, "run": run, "item_id": it["item_id"], "set": it["set"], "task": it["task"]}
        try:
            s, extra = RUNNERS[cond](row)
            rec.update(extra); rec["score"] = s; rec["status"] = "ok" if s is not None else "parse_fail"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:150]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result(); n += 1
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
