# -*- coding: utf-8 -*-
"""Run Ours (pipeline) and Direct on the failure-inclusive sample, 5 runs each,
KEEPING failures (parse failures, API errors, empty output) rather than dropping
them. Records a per-item per-run status so the study can report coverage and an
end-to-end Direct-fallback result. GPT-4o-mini, key from OPENAI_API_KEY."""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
MANIFEST = os.path.join(BASE, os.environ.get("MANIFEST_FILE", "failure_sample_manifest.csv"))
PROMPTS = os.path.join(BASE, "pipeline_prompts.json")
OUT = os.path.join(BASE, os.environ.get("OUT_FILE", "output_failure_sample.jsonl"))
MODEL = "gpt-4o-mini-2024-07-18"
N_RUNS = int(os.environ.get("N_RUNS", "5"))
write_lock = threading.Lock()
openai.api_key = os.environ.get("OPENAI_API_KEY")

P = json.load(open(PROMPTS, encoding="utf-8"))
SYS, SYS_EXEC = P["System_prompt"], P["System_prompt_plan_execution"]
SOLVER_COMMON, PLANNER_COMMON, CATS = P["User_prompt_solver_common"], P["User_prompt_planner_common"], P["planner_categories"]
DIRECT_INSTR = P["User_prompt_multiturn"].split("<|The Start of Assistant A's Conversation")[0].strip()
RR = re.compile(r"\[\[\s*([1-9])\s*\]\]"); FB = re.compile(r"[Rr]ating[:\s\*]*\[?\[?\s*([1-9])")


def call(system, user, retries=5):
    last = None
    for a in range(retries):
        try:
            r = openai.chat.completions.create(model=MODEL, temperature=0, max_tokens=4096,
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
    m = RR.findall(t or "")
    if m: return int(m[-1])
    m = FB.findall(t or "")
    return int(m[-1]) if m else None


def run_direct(conv):
    try:
        s = parse(call(SYS, DIRECT_INSTR + "\n\n" + conv))
        return (s, "ok") if s is not None else (None, "parse_fail")
    except Exception as e:
        return None, "error:" + str(e)[:60]


def run_pipeline(task, conv):
    try:
        plan = call(SYS, PLANNER_COMMON + CATS[task] + "\n" + conv)
        calls = re.findall(r"#(E\d+)\s*=\s*LLM\[(.*?)\]", plan, re.DOTALL)
        if not calls:
            # planner produced no parseable evidence plan -> pipeline cannot run
            processed = plan
        else:
            processed = plan
            for var, instr in calls:
                res = call(SYS_EXEC, instr)
                repl = f'#{var} = """{res}"""'
                processed = re.sub(re.escape(f"#{var} = LLM[{instr}]"), lambda m, r=repl: r, processed, count=1)
        s = parse(call(SYS, SOLVER_COMMON + conv + "\n\n" + processed))
        if s is None:
            return None, "parse_fail", len(calls)
        return s, "ok", len(calls)
    except Exception as e:
        return None, "error:" + str(e)[:60], 0


def load_done():
    d = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); d.add((r["run"], r["item_id"]))
    return d


def main():
    man = pd.read_csv(MANIFEST)
    src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
    si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
    done = load_done()
    jobs = [(run, r) for run in range(1, N_RUNS + 1) for _, r in man.iterrows() if (run, r["item_id"]) not in done]
    print(f"jobs={len(jobs)}")

    def work(job):
        run, m = job
        rec = {"run": run, "item_id": m["item_id"], "task": m["task"], "id": int(m["id"])}
        try:
            row = si[(m["task"], int(m["id"]))]
            conv = conv_block(row)
            rec["direct_score"], rec["direct_status"] = run_direct(conv)
            rec["pipeline_score"], rec["pipeline_status"], rec["n_evidence"] = run_pipeline(m["task"], conv)
        except Exception as e:
            rec["error"] = str(e)[:150]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result()
    print("done")


if __name__ == "__main__":
    main()
