# -*- coding: utf-8 -*-
"""Independent complete evaluations on MT-Bench (160 dialogues). Each run is
one batch that scores every dialogue with the PWS judge and the Direct judge;
runs are executed one after another with start and end times recorded. This
measures run-to-run variation of whole evaluations, including any effect
shared by all dialogues in a run, and validates the nested-model predictions.
gpt-4o-mini snapshot, temperature 0, same prompts as the other runs."""
import json, os, re, time, threading, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output_independent_runs.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"
K = int(os.environ.get("K_RUNS", "10"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "6"))
lock = threading.Lock()
client = OpenAI()

P = json.load(open(os.path.join(BASE, "pipeline_prompts.json"), encoding="utf-8"))
SYS, SYS_EXEC = P["System_prompt"], P["System_prompt_plan_execution"]
SOLVER_COMMON, PLANNER_COMMON = P["User_prompt_solver_common"], P["User_prompt_planner_common"]
DIRECT_INSTR = P["User_prompt_multiturn"].split("<|The Start of Assistant A's Conversation")[0].strip()
CATS = json.load(open(os.path.join(BASE, "mtbench_planner_categories.json"), encoding="utf-8"))
RR = re.compile(r"\[\[\s*([1-9])\s*\]\]"); FB = re.compile(r"[Rr]ating[:\s\*]*\[?\[?\s*([1-9])")


def call(system, user, retries=5):
    last = None
    for a in range(retries):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0,
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


def pws(task, conv):
    plan = call(SYS, PLANNER_COMMON + CATS[task] + "\n" + conv)
    processed = plan
    for var, instr in re.findall(r"#(E\d+)\s*=\s*LLM\[(.*?)\]", plan, re.DOTALL):
        res = call(SYS_EXEC, instr)
        processed = re.sub(re.escape(f"#{var} = LLM[{instr}]"), lambda m, r=f'#{var} = """{res}"""': r, processed, count=1)
    return parse(call(SYS, SOLVER_COMMON + conv + "\n\n" + processed))


def main():
    dialogues = [json.loads(l) for l in open(os.path.join(BASE, "data", "mtbench_dialogues.jsonl"), encoding="utf-8")]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d["status"] == "ok":
                    done.add((d["run"], d["judge"], d["item_id"]))
    for run in range(1, K + 1):
        jobs = [(r, j) for r in dialogues for j in ("pws", "direct") if (run, j, r["item_id"]) not in done]
        if not jobs:
            continue
        start = datetime.datetime.now().isoformat(timespec="seconds")
        print(f"run {run}: {len(jobs)} jobs, start {start}", flush=True)

        def work(job):
            r, j = job
            conv = conv_block(r)
            rec = {"run": run, "judge": j, "item_id": r["item_id"], "task": r["task"],
                   "t_start": datetime.datetime.now().isoformat(timespec="seconds")}
            try:
                rec["score"] = pws(r["task"], conv) if j == "pws" else parse(call(SYS, DIRECT_INSTR + "\n\n" + conv))
                rec["status"] = "ok"
            except Exception as e:
                rec["status"] = "error"; rec["error"] = str(e)[:150]
            rec["t_end"] = datetime.datetime.now().isoformat(timespec="seconds")
            with lock:
                open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for f in as_completed([ex.submit(work, j) for j in jobs]):
                f.result()
        print(f"run {run} done {datetime.datetime.now().isoformat(timespec='seconds')}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
