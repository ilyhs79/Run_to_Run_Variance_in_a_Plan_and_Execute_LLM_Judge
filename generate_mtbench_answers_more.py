# -*- coding: utf-8 -*-
"""Two-turn MT-Bench answers from four more answer systems, so the system
comparison covers six systems (with gpt-4o and gpt-5.4-nano from
generate_mtbench_answers.py). Non-reasoning models use MT-Bench's per-category
temperatures; reasoning models keep their defaults. Appends to the same file."""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
QFILE = os.path.join(BASE, "data", "mtbench_question.jsonl")
OUT = os.path.join(BASE, "data", "mtbench_dialogues.jsonl")
TEMP = {"writing": 0.7, "roleplay": 0.7, "extraction": 0.0, "math": 0.0, "coding": 0.0,
        "reasoning": 0.0, "stem": 0.1, "humanities": 0.1}
MODELS = {"gpt4": ("openai", "gpt-4-0613", False), "mini54": ("openai", "gpt-5.4-mini-2026-03-17", True),
          "o4mini": ("openai", "o4-mini-2025-04-16", True), "gemfl": ("gemini", "gemini-3.1-flash-lite", False)}
oa = OpenAI()
lock = threading.Lock()
_gem = None


def gemini_chat(model, msgs, temp):
    global _gem
    from google import genai
    from google.genai import types
    if _gem is None:
        _gem = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    contents = [types.Content(role="user" if m["role"] == "user" else "model", parts=[types.Part(text=m["content"])]) for m in msgs]
    cfg = types.GenerateContentConfig(temperature=temp, max_output_tokens=1024, thinking_config=types.ThinkingConfig(thinking_budget=0))
    return _gem.models.generate_content(model=model, contents=contents, config=cfg).text or ""


def chat(key, category, msgs, retries=5):
    prov, model, reasoning = MODELS[key]
    last = None
    for a in range(retries):
        try:
            if prov == "gemini":
                return gemini_chat(model, msgs, TEMP[category])
            kw = dict(model=model, messages=msgs)
            if reasoning:
                kw.update(max_completion_tokens=4096)
            else:
                kw.update(temperature=TEMP[category], max_tokens=1024)
            return oa.chat.completions.create(**kw).choices[0].message.content or ""
        except Exception as e:
            last = e; time.sleep(min(2 ** a * 3, 30))
    raise last


def main():
    qs = [json.loads(l) for l in open(QFILE, encoding="utf-8") if l.strip()]
    done = {json.loads(l)["item_id"] for l in open(OUT, encoding="utf-8") if l.strip()}
    jobs = [(q, k) for q in qs for k in MODELS if f"M{q['question_id']}_{k}" not in done]
    print(f"jobs={len(jobs)}", flush=True)

    def work(job):
        q, k = job
        msgs, answers = [], []
        for turn in q["turns"]:
            msgs.append({"role": "user", "content": turn})
            ans = chat(k, q["category"], msgs)
            answers.append(ans); msgs.append({"role": "assistant", "content": ans})
        rec = {"item_id": f"M{q['question_id']}_{k}", "question_id": q["question_id"], "task": q["category"],
               "answer_model": MODELS[k][1], "number_of_turns": len(q["turns"])}
        for i, (u, a) in enumerate(zip(q["turns"], answers), 1):
            rec[f"question_{i}"] = u; rec[f"answer_{i}"] = a
        with lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result()
    print("done")


if __name__ == "__main__":
    main()
