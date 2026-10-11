# -*- coding: utf-8 -*-
"""Nested variance design. Per item: P planner draws; per plan, W worker draws
with S solver calls each (full pipeline), plus R solver calls on the same plan
left unresolved (no Worker). Lets nested ANOVA separate planner / worker / solver
variance and test whether planner variance is amplified by the Worker, holding
the plans fixed. gpt-4o-mini snapshot, temperature 0, paper prompts."""
import json, os, re, time, threading, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
OUT = os.path.join(BASE, "output_nested_variance.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"
NP, NW, NS, NR = 4, 3, 2, 3
N_NEW = int(os.environ.get("N_NEW", "100"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
write_lock = threading.Lock()
openai.api_key = os.environ.get("OPENAI_API_KEY")

P = json.load(open(os.path.join(BASE, "pipeline_prompts.json"), encoding="utf-8"))
SYS, SYS_EXEC = P["System_prompt"], P["System_prompt_plan_execution"]
SOLVER_COMMON, PLANNER_COMMON, CATS = P["User_prompt_solver_common"], P["User_prompt_planner_common"], P["planner_categories"]
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


def one_plan(task, conv):
    plan = call(SYS, PLANNER_COMMON + CATS[task] + "\n" + conv)
    calls = re.findall(r"#(E\d+)\s*=\s*LLM\[(.*?)\]", plan, re.DOTALL)
    full, evh = [], []
    for _ in range(NW):
        processed, ev = plan, {}
        for var, instr in calls:
            res = call(SYS_EXEC, instr); ev[var] = h(res)
            processed = re.sub(re.escape(f"#{var} = LLM[{instr}]"), lambda m, r=f'#{var} = """{res}"""': r, processed, count=1)
        prompt = SOLVER_COMMON + conv + "\n\n" + processed
        full.append([parse(call(SYS, prompt)) for _ in range(NS)]); evh.append(ev)
    nw = [parse(call(SYS, SOLVER_COMMON + conv + "\n\n" + plan)) for _ in range(NR)]
    return {"plan_hash": h(plan), "n_evidence": len(calls), "full": full, "noworker": nw, "ev_hash": evh}


def main():
    src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
    si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
    orig = pd.read_csv(os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv"))[["item_id", "task", "id"]].assign(set="orig100")
    new = pd.concat([pd.read_csv(os.path.join(BASE, f)) for f in ["failure_sample_manifest.csv", "failure_sample_manifest_batch2.csv"]])[["item_id", "task", "id"]]
    new = new[new["task"] != "MR"].sample(n=N_NEW, random_state=20261011).assign(set="new100")
    items = pd.concat([orig, new], ignore_index=True)

    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                d = json.loads(l); done.add((d["item_id"], d["plan_idx"]))
    jobs = [(it, p) for _, it in items.iterrows() for p in range(NP) if (it["item_id"], p) not in done]
    print(f"model={MODEL} items={len(items)} jobs={len(jobs)} (P={NP} W={NW} S={NS} R={NR})", flush=True)

    def work(job):
        it, p = job
        rec = {"item_id": it["item_id"], "set": it["set"], "task": it["task"], "plan_idx": p}
        try:
            row = si[(it["task"], int(it["id"]))]
            rec.update(one_plan(it["task"], conv_block(row))); rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:150]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result(); n += 1
            if n % 40 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
