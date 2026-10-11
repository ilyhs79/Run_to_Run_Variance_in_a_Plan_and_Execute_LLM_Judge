# -*- coding: utf-8 -*-
"""Generate two-turn assistant answers for the 80 MT-Bench questions with two
answer models (not used as judges), giving 160 dialogues for the cross-benchmark
reliability study. gpt-4o uses MT-Bench's per-category sampling temperatures
(FastChat config); gpt-5.4-nano is a reasoning model and keeps its default."""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai

BASE = os.path.dirname(os.path.abspath(__file__))
QFILE = os.path.join(BASE, "data", "mtbench_question.jsonl")
OUT = os.path.join(BASE, "data", "mtbench_dialogues.jsonl")
MODELS = {"gpt4o": "gpt-4o-2024-08-06", "nano": "gpt-5.4-nano-2026-03-17"}
TEMP = {"writing": 0.7, "roleplay": 0.7, "extraction": 0.0, "math": 0.0, "coding": 0.0,
        "reasoning": 0.0, "stem": 0.1, "humanities": 0.1}
client = openai.OpenAI()
lock = threading.Lock()


def chat(model_key, category, messages, retries=5):
    kw = dict(model=MODELS[model_key], messages=messages)
    if model_key == "gpt4o":
        kw.update(temperature=TEMP[category], max_tokens=1024)
    else:
        kw.update(max_completion_tokens=4096)
    last = None
    for a in range(retries):
        try:
            return client.chat.completions.create(**kw).choices[0].message.content or ""
        except Exception as e:
            last = e; time.sleep(min(2 ** a * 3, 30))
    raise last


def main():
    qs = [json.loads(l) for l in open(QFILE, encoding="utf-8") if l.strip()]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["item_id"] for l in open(OUT, encoding="utf-8") if l.strip()}
    jobs = [(q, mk) for q in qs for mk in MODELS if f"M{q['question_id']}_{mk}" not in done]
    print(f"jobs={len(jobs)}", flush=True)

    def work(job):
        q, mk = job
        msgs, answers = [], []
        for turn in q["turns"]:
            msgs.append({"role": "user", "content": turn})
            ans = chat(mk, q["category"], msgs)
            answers.append(ans); msgs.append({"role": "assistant", "content": ans})
        rec = {"item_id": f"M{q['question_id']}_{mk}", "question_id": q["question_id"], "task": q["category"],
               "answer_model": MODELS[mk], "number_of_turns": len(q["turns"])}
        for i, (u, a) in enumerate(zip(q["turns"], answers), 1):
            rec[f"question_{i}"] = u; rec[f"answer_{i}"] = a
        with lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result()
    print("done")


if __name__ == "__main__":
    main()
