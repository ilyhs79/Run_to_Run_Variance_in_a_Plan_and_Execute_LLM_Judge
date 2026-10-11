# -*- coding: utf-8 -*-
"""Cross-model replication of the reliability findings. For one judge model
(PROVIDER = anthropic | gemini | together): Direct x5 per item, and a reduced
nested pipeline design per item (P plans; per plan W worker draws x S solver
calls; plus R no-Worker solver calls on the same plan). Same paper prompts and
temperature 0 as the gpt-4o-mini runs. Items: the original 100."""
import json, os, re, time, threading, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
PROVIDER = os.environ.get("PROVIDER", "anthropic")
DATASET = os.environ.get("DATASET", "mtb101")
NP, NW, NS, NR, ND = (int(os.environ.get(k, d)) for k, d in [("NP", 3), ("NW", 2), ("NS", 2), ("NR", 2), ("ND", 5)])
N_ITEMS = int(os.environ.get("N_ITEMS", "100"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
MAX_TOKENS = 4096
write_lock = threading.Lock()

if PROVIDER == "anthropic":
    import anthropic
    MODEL = os.environ.get("MM_MODEL", "claude-haiku-4-5-20251001")
    _c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _raw(system, user):
        r = _c.messages.create(model=MODEL, system=system, max_tokens=MAX_TOKENS,
                               messages=[{"role": "user", "content": user}], extra_body={"temperature": 0})
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
elif PROVIDER == "gemini":
    from google import genai
    from google.genai import types
    MODEL = os.environ.get("MM_MODEL", "gemini-3.1-flash-lite")
    _c = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

    def _raw(system, user):
        cfg = types.GenerateContentConfig(system_instruction=system, temperature=0, max_output_tokens=MAX_TOKENS,
                                          thinking_config=types.ThinkingConfig(thinking_budget=0))
        return _c.models.generate_content(model=MODEL, contents=user, config=cfg).text or ""
elif PROVIDER == "together":
    from openai import OpenAI
    MODEL = os.environ.get("MM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    _c = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")

    def _raw(system, user):
        r = _c.chat.completions.create(model=MODEL, temperature=0, max_tokens=MAX_TOKENS,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or ""
elif PROVIDER == "openai":
    from openai import OpenAI
    MODEL = os.environ.get("MM_MODEL", "gpt-4o-mini-2024-07-18")
    _c = OpenAI()

    def _raw(system, user):
        r = _c.chat.completions.create(model=MODEL, temperature=0,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or ""
else:
    raise SystemExit("PROVIDER must be anthropic | gemini | together | openai")

TAG = {"anthropic": "haiku", "gemini": "gemini", "together": "llama8b", "openai": "4omini"}[PROVIDER]
OUT = os.path.join(BASE, f"output_mm_nested_{TAG}.jsonl" if DATASET == "mtb101" else f"output_{DATASET}_nested_{TAG}.jsonl")

P = json.load(open(os.path.join(BASE, "pipeline_prompts.json"), encoding="utf-8"))
SYS, SYS_EXEC = P["System_prompt"], P["System_prompt_plan_execution"]
SOLVER_COMMON, PLANNER_COMMON, CATS = P["User_prompt_solver_common"], P["User_prompt_planner_common"], P["planner_categories"]
if DATASET == "mtbench":
    CATS = json.load(open(os.path.join(BASE, "mtbench_planner_categories.json"), encoding="utf-8"))
DIRECT_INSTR = P["User_prompt_multiturn"].split("<|The Start of Assistant A's Conversation")[0].strip()
RR = re.compile(r"\[\[\s*([1-9])\s*\]\]"); FB = re.compile(r"[Rr]ating[:\s\*]*\[?\[?\s*([1-9])")


def call(system, user, retries=6):
    last = None
    for a in range(retries):
        try:
            return _raw(system, user)
        except (TypeError, ValueError) as e:
            raise e
        except Exception as e:
            last = e; time.sleep(min(2 ** a * 3, 45))
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
        full.append([parse(call(SYS, SOLVER_COMMON + conv + "\n\n" + processed)) for _ in range(NS)]); evh.append(ev)
    nw = [parse(call(SYS, SOLVER_COMMON + conv + "\n\n" + plan)) for _ in range(NR)]
    return {"plan_hash": h(plan), "n_evidence": len(calls), "full": full, "noworker": nw, "ev_hash": evh}


def main():
    if DATASET == "mtbench":
        rows = [json.loads(l) for l in open(os.path.join(BASE, "data", "mtbench_dialogues.jsonl"), encoding="utf-8") if l.strip()]
        items = pd.DataFrame(rows)
        if os.environ.get("MTB_SUBSET") == "balanced80":
            # every question once; answer model alternates by question rank within category
            items["rank"] = items.groupby("task")["question_id"].rank(method="dense").astype(int)
            items = items[((items["rank"] % 2 == 1) & items["item_id"].str.endswith("_gpt4o")) |
                          ((items["rank"] % 2 == 0) & items["item_id"].str.endswith("_nano"))]
        items = items.sort_values("item_id").head(N_ITEMS)
        items["id"] = range(len(items))
        si = {(r["task"], int(r["id"])): r for _, r in items.iterrows()}
    else:
        src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
        si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
        items = pd.read_csv(os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv"))[["item_id", "task", "id"]].head(N_ITEMS)
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d["status"] == "ok":
                    done.add((d["kind"], d["item_id"], d["idx"]))
    jobs = [("direct", it, k) for _, it in items.iterrows() for k in range(ND) if ("direct", it["item_id"], k) not in done]
    jobs += [("plan", it, p) for _, it in items.iterrows() for p in range(NP) if ("plan", it["item_id"], p) not in done]
    print(f"dataset={DATASET} provider={PROVIDER} model={MODEL} items={len(items)} jobs={len(jobs)} (P={NP} W={NW} S={NS} R={NR} D={ND})", flush=True)

    def work(job):
        kind, it, k = job
        rec = {"kind": kind, "item_id": it["item_id"], "task": it["task"], "idx": k, "model": MODEL}
        try:
            conv = conv_block(si[(it["task"], int(it["id"]))])
            if kind == "direct":
                rec["score"] = parse(call(SYS, DIRECT_INSTR + "\n\n" + conv))
            else:
                rec.update(one_plan(it["task"], conv))
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:200]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result(); n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
