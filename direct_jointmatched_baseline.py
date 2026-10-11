# -*- coding: utf-8 -*-
"""
Reviewer (calibrated, iter5/6): a JOINTLY-matched single-call control.

The individual ablations each add one ingredient to Direct: category rubric
(direct_rubric_ablation), turn-by-turn CoT (direct_cot_ablation), a worked
example (direct_workedexample_baseline). The reviewer asks for one undecomposed
call that receives ALL of them at once -- category guidance + the worked
demonstration + explicit evidence-checking instructions + comparable budget --
to test whether the pipeline's gain survives against a single call that is given
every ingredient except the Planner/Worker/Solver decomposition.

Single call, GPT-4o-mini, temperature 0, run on the 100 human-rated items so it
slots into the human-alignment table next to Direct-mini and Ours-mini.

Key from OPENAI_API_KEY (never hardcoded).
"""
import json, csv, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
ANSWER_KEY = os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")
PROMPTS = os.path.join(BASE, "pipeline_prompts.json")
OUT_JSONL = os.path.join(BASE, "output_direct_jointmatched.jsonl")
OUT_CSV = os.path.join(BASE, "output_direct_jointmatched.csv")
MODEL = "gpt-4o-mini"
MAX_WORKERS = 6
write_lock = threading.Lock()

openai.api_key = os.environ.get("OPENAI_API_KEY")
if not openai.api_key:
    raise SystemExit("set OPENAI_API_KEY")

# category definition sentence, identical to direct_cot_ablation.py
CATEGORY_RUBRIC = {
    "GR": "General Reasoning, which is to collaboratively solve complex general reasoning problems with users across dialogue turns.",
    "IC": "Instruction Clarification, which is to seek clarification by asking further questions on ambiguous user queries.",
    "CM": "Context Memory, which is to recall early dialogue details to address the user's current question.",
    "AR": "Anaphora Resolution, which is to identify pronoun referents throughout a multi-turn dialogue.",
    "SI": "Separate Input, where the first turn outlines the task requirements and the following turns specify the task input.",
    "TS": "Topic Shift, which is to recognize and focus on the new topic when users unpredictably switch topics.",
    "CC": "Content Confusion, which is to avoid interference from similar-looking queries with distinct meanings in the dialogue's history.",
    "CR": "Content Rephrasing, which is to rephrase the content of the last response according to the user's newest requirement.",
    "FR": "Format Rephrasing, which is to rephrase the format of the last response according to the user's newest requirement.",
    "SC": "Self-Correction, which is to recorrect the last response according to the user feedback.",
    "SA": "Self-Affirmation, which is to preserve the last response against inaccurate user feedback.",
    "PI": "Proactive Interaction, which is to propose questions in reaction to user statements to spark their interest to continue the dialogue.",
}

# worked example demo, extracted the same way as direct_workedexample_baseline.py
P = json.load(open(PROMPTS, encoding="utf-8"))
END = "<|The End of Evaluation Plan|>"
DEMO = {}
for cat, txt in P["planner_categories"].items():
    i = txt.find(END)
    DEMO[cat] = txt[: i + len(END)].strip() if i != -1 else txt.strip()

# turn-by-turn CoT instruction, identical to direct_cot_ablation.py
COT = ('Before giving your final score, think step by step and be critical: for EACH turn in the '
       'conversation, write a detailed analysis (3-5 sentences) that separately examines (a) whether the '
       "assistant's response is genuinely helpful and relevant, explicitly noting any missed opportunities, "
       'vagueness, or unhelpful parts if present, (b) whether it is factually or logically accurate, explicitly '
       'flagging any errors, unsupported claims, or imprecision if present, (c) whether it stays consistent with '
       'everything said in earlier turns, explicitly flagging any contradiction or forgotten context if present, '
       'and (d) whether any pronouns, references, or topic shifts in that turn are correctly resolved against the '
       'right earlier turn, explicitly flagging any ambiguous or incorrect resolution if present. For each of '
       '(a)-(d), state a specific strength or a specific weakness. Do not default to generic praise, and actively '
       'look for problems. Do this turn-by-turn analysis for every single turn before moving on. Only after '
       'analyzing every turn individually in this detail, write an overall explanation that synthesizes these '
       'per-turn observations into a holistic judgment, weighing strengths and weaknesses fairly.')


def build_prompt(task, conv):
    rubric = CATEGORY_RUBRIC[task]
    return f"""[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.

This conversation is about {rubric}
When scoring, pay particular attention to this specific competency in addition to overall response quality.

Below is an example of a conversation and a worked evaluation plan for this category, to guide your reasoning:

{DEMO[task]}

{COT}

After your turn-by-turn analysis and overall explanation, please rate the response on a scale of 1 to 9 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]."

{conv}
"""


def ChatGPT(prompt, retries=5):
    last = None
    for a in range(retries):
        try:
            r = openai.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": "You are a helpful assistant"},
                          {"role": "user", "content": prompt}],
                temperature=0)
            return r.choices[0].message.content
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


def parse_rating(out):
    idx = out.find("Rating: [[")
    if idx != -1 and out[idx + 10].isdigit():
        return int(out[idx + 10])
    import re
    m = re.findall(r"\[\[\s*([1-9])\s*\]\]", out or "")
    return int(m[-1]) if m else None


def load_done():
    d = set()
    if os.path.exists(OUT_JSONL):
        for line in open(OUT_JSONL, encoding="utf-8"):
            line = line.strip()
            if line:
                d.add(json.loads(line)["item_id"])
    return d


def main():
    key = pd.read_csv(ANSWER_KEY)
    src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True)
    src = src.drop_duplicates(subset=["task", "id"], keep="first")
    src_idx = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
    done = load_done()
    todo = [r for _, r in key.iterrows() if r["item_id"] not in done]
    print(f"items={len(key)} done={len(done)} todo={len(todo)}")

    def work(kr):
        item_id, task, did = kr["item_id"], kr["task"], int(kr["id"])
        rec = {"item_id": item_id, "task": task, "id": did}
        try:
            row = src_idx.get((task, did))
            if row is None:
                rec["status"] = "no_source"; return rec
            out = ChatGPT(build_prompt(task, conv_block(row)))
            rec["score"] = parse_rating(out)
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:200]
        with write_lock:
            open(OUT_JSONL, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, f in enumerate(as_completed([ex.submit(work, r) for r in todo]), 1):
            f.result()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")

    rows = [json.loads(l) for l in open(OUT_JSONL, encoding="utf-8") if l.strip()]
    fn = sorted({k for r in rows for k in r})
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r.get("status") == "ok" and r.get("score")]
    print(f"done. ok={len(ok)}/{len(rows)}")


if __name__ == "__main__":
    main()
