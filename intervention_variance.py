# -*- coding: utf-8 -*-
"""Interventions on the PWS judge's intermediate outputs, measured with the
nested design (P plans x W Worker draws x S Solver calls) on the first
MT-Bench-101 set. Variants (VARIANT env):
  short : the Worker answers each evidence request in at most two sentences,
          conclusion first
  prob  : the Solver gives only the rating, and the score is the
          probability-weighted mean over the digits 1-9
  both  : short + prob
Records token usage per dialogue so cost can be compared."""
import json, math, os, re, time, threading, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
VARIANT = os.environ.get("VARIANT", "short")
DATASET = os.environ.get("DATASET", "mtb101")
OUT = os.path.join(BASE, f"output_intervention_{VARIANT}_{DATASET}.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"
NP, NW, NS = 3, 2, 2
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "6"))
lock = threading.Lock()
client = OpenAI()

P = json.load(open(os.path.join(BASE, "pipeline_prompts.json"), encoding="utf-8"))
SYS, SYS_EXEC = P["System_prompt"], P["System_prompt_plan_execution"]
SOLVER_COMMON, PLANNER_COMMON = P["User_prompt_solver_common"], P["User_prompt_planner_common"]
CATS = P["planner_categories"] if DATASET == "mtb101" else json.load(open(os.path.join(BASE, "mtbench_planner_categories.json"), encoding="utf-8"))
SHORT = "\n\nAnswer in at most two sentences. State your conclusion first."
PROB_TAIL = "\n\nRespond with the rating only, as a single integer from 1 to 9, and nothing else.\n\nRating:"
RR = re.compile(r"\[\[\s*([1-9])\s*\]\]"); FB = re.compile(r"[Rr]ating[:\s\*]*\[?\[?\s*([1-9])")


def call(system, user, usage, prob=False, retries=5):
    last = None
    for a in range(retries):
        try:
            kw = dict(model=MODEL, temperature=0, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if prob:
                kw.update(max_tokens=12, logprobs=True, top_logprobs=20)
            r = client.chat.completions.create(**kw)
            usage[0] += r.usage.prompt_tokens; usage[1] += r.usage.completion_tokens
            if not prob:
                return r.choices[0].message.content or ""
            if VARIANT == "ronly":  # rating-only prompt, greedy digit (no probability weighting)
                m = re.search(r"[1-9]", r.choices[0].message.content or "")
                return int(m.group(0)) if m else None
            # weight over the position where the rating digit was actually generated
            for pos in r.choices[0].logprobs.content:
                if not (len(pos.token.strip()) == 1 and pos.token.strip() in "123456789"):
                    continue
                w = {}
                for alt in pos.top_logprobs:
                    s = alt.token.strip()
                    if len(s) == 1 and s in "123456789":
                        w[int(s)] = w.get(int(s), 0.0) + math.exp(alt.logprob)
                if w:
                    z = sum(w.values())
                    return sum(k * v for k, v in w.items()) / z
            return None
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


def one_plan(task, conv):
    usage = [0, 0]
    short = VARIANT in ("short", "both"); prob = VARIANT in ("prob", "both", "ronly")
    plan = call(SYS, PLANNER_COMMON + CATS[task] + "\n" + conv, usage)
    calls = re.findall(r"#(E\d+)\s*=\s*LLM\[(.*?)\]", plan, re.DOTALL)
    full, ev_words = [], []
    for _ in range(NW):
        processed = plan
        for var, instr in calls:
            res = call(SYS_EXEC, instr + (SHORT if short else ""), usage)
            ev_words.append(len(res.split()))
            processed = re.sub(re.escape(f"#{var} = LLM[{instr}]"), lambda m, r=f'#{var} = """{res}"""': r, processed, count=1)
        prompt = SOLVER_COMMON + conv + "\n\n" + processed + (PROB_TAIL if prob else "")
        full.append([call(SYS, prompt, usage, prob=True) if prob else parse(call(SYS, prompt, usage)) for _ in range(NS)])
    return {"plan_hash": hashlib.md5(plan.encode("utf-8")).hexdigest()[:12], "n_evidence": len(calls), "full": full,
            "ev_words": sum(ev_words) / max(len(ev_words), 1), "tok_in": usage[0], "tok_out": usage[1]}


def items():
    if DATASET == "mtb101":
        src = pd.concat([pd.read_csv(os.path.join(BASE, f)) for f in ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                         "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]).drop_duplicates(subset=["task", "id"])
        si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
        return [(k["item_id"], k["task"], conv_block(si[(k["task"], int(k["id"]))]))
                for _, k in pd.read_csv(os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")).iterrows()]
    rows = [json.loads(l) for l in open(os.path.join(BASE, "data", "mtbench_dialogues.jsonl"), encoding="utf-8")]
    return [(r["item_id"], r["task"], conv_block(r)) for r in rows if r["item_id"].endswith(("_gpt4o", "_nano"))]


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d["status"] == "ok":
                    done.add((d["item_id"], d["plan_idx"]))
    jobs = [(iid, task, conv, p) for iid, task, conv in items() for p in range(NP) if (iid, p) not in done]
    print(f"variant={VARIANT} dataset={DATASET} jobs={len(jobs)}", flush=True)

    def work(job):
        iid, task, conv, p = job
        rec = {"item_id": iid, "task": task, "plan_idx": p, "variant": VARIANT}
        try:
            rec.update(one_plan(task, conv)); rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:150]
        with lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result(); n += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
