# -*- coding: utf-8 -*-
"""Second sequential judge design: Critique -> Score (CS). Stage 1 writes a
free-text critique of the assistant's turns without scoring. Stage 2 rates the
dialogue from the transcript and that critique. Nested design per dialogue:
P critiques, each scored S times, so the single-run variance splits into a
critique component and a scorer component. Same system prompt, rating format,
temperature 0, and gpt-4o-mini snapshot as the other judges.
Datasets: first MT-Bench-101 set (100) and MT-Bench (160)."""
import json, os, re, time, threading, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
OUT = os.path.join(BASE, "output_cs_nested.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"
NP, NS = int(os.environ.get("NP", "4")), int(os.environ.get("NS", "3"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "6"))
lock = threading.Lock()
client = OpenAI()

SYS = json.load(open(os.path.join(BASE, "pipeline_prompts.json"), encoding="utf-8"))["System_prompt"]

CRITIC = (
    "[Instruction]\nPlease act as an impartial judge and review the AI assistant's responses in the multi-turn "
    "conversation below. Do not give a score. Write a critique that goes through the conversation turn by turn. "
    "For each assistant turn, state what it does well and what is wrong, missing, or unhelpful, considering "
    "accuracy, relevance, following the user's instructions, and use of earlier turns. Quote the specific parts of "
    "the responses you refer to. End with a short summary of the main strengths and weaknesses.\n\n{conv}")

SCORER = (
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the AI assistant's responses in the "
    "multi-turn conversation below. Your evaluation should consider factors such as the helpfulness, relevance, "
    "accuracy, depth, creativity, and level of detail of the responses. A critique of the conversation written by a "
    "reviewer is provided after the conversation. Use it as evidence, but check it against the conversation. "
    "Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your "
    "explanation, please rate the response on a scale of 1 to 9 by strictly following this format: \"[[rating]]\", "
    "for example: \"Rating: [[5]].\"\n\n{conv}\n\n<|The Start of Critique|>\n{critique}\n<|The End of Critique|>")

RR = re.compile(r"\[\[\s*([1-9])\s*\]\]"); FB = re.compile(r"[Rr]ating[:\s\*]*\[?\[?\s*([1-9])")


def call(user, retries=5):
    last = None
    for a in range(retries):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0,
                                               messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}])
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


def one_critique(conv):
    crit = call(CRITIC.format(conv=conv))
    scores = [parse(call(SCORER.format(conv=conv, critique=crit))) for _ in range(NS)]
    return {"crit_hash": hashlib.md5(crit.encode("utf-8")).hexdigest()[:12], "crit_len": len(crit.split()), "scores": scores}


def items():
    src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
    si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
    out = []
    for _, k in pd.read_csv(os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")).iterrows():
        out.append(("mtb101", k["item_id"], conv_block(si[(k["task"], int(k["id"]))])))
    for l in open(os.path.join(BASE, "data", "mtbench_dialogues.jsonl"), encoding="utf-8"):
        r = json.loads(l); out.append(("mtbench", r["item_id"], conv_block(r)))
    return out


def main():
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                if d["status"] == "ok":
                    done.add((d["item_id"], d["crit_idx"]))
    jobs = [(ds, iid, conv, p) for ds, iid, conv in items() for p in range(NP) if (iid, p) not in done]
    print(f"model={MODEL} jobs={len(jobs)} (P={NP} S={NS})", flush=True)

    def work(job):
        ds, iid, conv, p = job
        rec = {"dataset": ds, "item_id": iid, "crit_idx": p}
        try:
            rec.update(one_critique(conv)); rec["status"] = "ok"
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
