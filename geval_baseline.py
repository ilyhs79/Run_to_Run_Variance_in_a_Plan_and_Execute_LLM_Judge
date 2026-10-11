# -*- coding: utf-8 -*-
"""Established evaluator baseline: G-Eval (Liu et al., 2023), the standard
reasoning-based LLM judge. Task intro + criteria + explicit evaluation steps,
then a probability-weighted score read from the token logprobs over digits 1-9
(the G-Eval scoring scheme). Same GPT-4o-mini, same 100 human-rated items, run
five times for execution variance, compared to Ours-mini and Direct-mini.

Key from OPENAI_API_KEY.
"""
import json, os, math, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
ANSWER_KEY = os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")
OUT = os.path.join(BASE, "output_geval.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"
N_RUNS = int(os.environ.get("N_RUNS", "5"))
write_lock = threading.Lock()
openai.api_key = os.environ.get("OPENAI_API_KEY")

PROMPT = """You will be given a multi-turn conversation between a user and an AI assistant (Assistant A). You will rate the assistant's overall response quality on one metric.

Please make sure you read and understand these instructions carefully. Keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

Quality (1-9) - the overall quality of Assistant A's responses across the whole conversation. Consider helpfulness, relevance, accuracy, depth, creativity, and level of detail, and also whether later turns stay logically consistent with earlier turns and resolve references (pronouns, entities, prior constraints) correctly. A high score requires that the assistant handle the cross-turn structure of the dialogue well, not only produce fluent individual replies.

Evaluation Steps:

1. Read the entire conversation carefully and understand what the user asks at each turn.
2. For each assistant turn, check whether the response is helpful and relevant, whether it is factually and logically accurate, and whether it stays consistent with everything said earlier and resolves references to earlier turns correctly.
3. Weigh the strengths and weaknesses you found across all turns, giving weight to cross-turn consistency and reference errors.
4. Assign a single overall Quality score from 1 (poor) to 9 (excellent).

Conversation:

{conv}

Now assign the overall Quality score. Respond with a single integer from 1 to 9 and nothing else.

Quality:"""


def conv_block(row):
    n = int(row["number_of_turns"])
    parts = []
    for t in range(1, n + 1):
        parts.append(f"### User:\n{row[f'question_{t}']}\n### Assistant A:\n{row[f'answer_{t}']}\n")
    return "\n".join(parts)


def geval_score(conv, retries=5):
    last = None
    for a in range(retries):
        try:
            r = openai.chat.completions.create(
                model=MODEL, temperature=1, max_tokens=6,
                logprobs=True, top_logprobs=20,
                messages=[{"role": "user", "content": PROMPT.format(conv=conv)}])
            # find the first token position whose top_logprobs contain a 1-9 digit
            content = r.choices[0].logprobs.content
            for pos in content:
                weights = {}
                for alt in pos.top_logprobs:
                    s = alt.token.strip()
                    if s in "123456789" and len(s) == 1:
                        weights[int(s)] = weights.get(int(s), 0.0) + math.exp(alt.logprob)
                if weights:
                    z = sum(weights.values())
                    return sum(k * v for k, v in weights.items()) / z, r.choices[0].message.content.strip()
            return None, r.choices[0].message.content.strip()
        except Exception as e:
            last = e; time.sleep(min(2 ** a * 3, 30))
    raise last


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
    print(f"jobs={len(jobs)}")

    def work(job):
        run, kr = job
        rec = {"run": run, "item_id": kr["item_id"], "task": kr["task"], "id": int(kr["id"])}
        try:
            row = si[(kr["task"], int(kr["id"]))]
            sc, raw = geval_score(conv_block(row))
            rec["score"] = sc; rec["raw"] = raw; rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:150]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result(); n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}")
    print("done")


if __name__ == "__main__":
    main()
