# -*- coding: utf-8 -*-
"""Execution-variance for the ablation conditions, to put them on the same
footing as the K-run main pipeline result. Conditions (all gpt-4o-mini, exact
paper prompts): rubric, cot, example, sc (3x temp0.7 mean), noworker
(planner plan left unresolved -> solver). K runs each on the 100 human items.
Key from OPENAI_API_KEY."""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILES = [os.path.join(BASE, f) for f in
                ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                 "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]
ANSWER_KEY = os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv")
PROMPTS = os.path.join(BASE, "pipeline_prompts.json")
OUT = os.path.join(BASE, "output_conditions_variance.jsonl")
MODEL = "gpt-4o-mini-2024-07-18"
N_RUNS = int(os.environ.get("N_RUNS", "5"))
MAX_WORKERS = 6
write_lock = threading.Lock()
openai.api_key = os.environ.get("OPENAI_API_KEY")

P = json.load(open(PROMPTS, encoding="utf-8"))
SYS, SYS_EXEC = P["System_prompt"], P["System_prompt_plan_execution"]
SOLVER_COMMON, PLANNER_COMMON, CATS = P["User_prompt_solver_common"], P["User_prompt_planner_common"], P["planner_categories"]
RATING_RE = re.compile(r"\[\[\s*([1-9])\s*\]\]")
FB = re.compile(r"[Rr]ating[:\s\*]*\[?\[?\s*([1-9])")

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
END = "<|The End of Evaluation Plan|>"
DEMO = {c: (t[: t.find(END) + len(END)].strip() if t.find(END) != -1 else t.strip()) for c, t in CATS.items()}
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
RATE = 'please rate the response on a scale of 1 to 9 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]."'
HEAD = ("[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI "
        "assistant to the user question displayed below.\nYour evaluation should consider factors such as the "
        "helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.\n\n"
        "This conversation is about {rubric}\nWhen scoring, pay particular attention to this specific competency in "
        "addition to overall response quality.\n\n")


def call(system, user, temp=0, retries=5):
    last = None
    for a in range(retries):
        try:
            r = openai.chat.completions.create(model=MODEL, temperature=temp,
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
    m = RATING_RE.findall(t or "")
    if m: return int(m[-1])
    m = FB.findall(t or "")
    return int(m[-1]) if m else None


def cond_rubric(task, conv):
    return parse(call(SYS, HEAD.format(rubric=CATEGORY_RUBRIC[task]) + "Begin your evaluation by providing a short explanation. Be as objective as possible.\nAfter providing your explanation, " + RATE + "\n\n" + conv))


def cond_cot(task, conv):
    return parse(call(SYS, HEAD.format(rubric=CATEGORY_RUBRIC[task]) + COT + "\n\nAfter your turn-by-turn analysis and overall explanation, " + RATE + "\n\n" + conv))


def cond_example(task, conv):
    demo = DEMO[task]
    p = ("[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI "
         "assistant to the user question displayed below.\nYour evaluation should consider factors such as the "
         "helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response.\n\n"
         "Below is an example of a conversation and a worked evaluation plan for this category, to guide your reasoning:\n\n"
         f"{demo}\n\nNow evaluate the following conversation in a single response. Reason through the same kinds of checks "
         "the example plan illustrates, then give one holistic score.\nBegin your evaluation by providing a short "
         "explanation. Be as objective as possible.\nAfter providing your explanation, " + RATE + f"\n\n{conv}")
    return parse(call(SYS, p))


def cond_sc(task, conv):
    base = HEAD.format(rubric=CATEGORY_RUBRIC[task]) + "Begin your evaluation by providing a short explanation. Be as objective as possible.\nAfter providing your explanation, " + RATE + "\n\n" + conv
    vals = [parse(call(SYS, base, temp=0.7)) for _ in range(3)]
    vals = [v for v in vals if v]
    return round(sum(vals) / len(vals), 4) if vals else None


def cond_noworker(task, conv):
    plan = call(SYS, PLANNER_COMMON + CATS[task] + "\n" + conv)  # unresolved plan
    return parse(call(SYS, SOLVER_COMMON + conv + "\n\n" + plan))


CONDS = {"rubric": cond_rubric, "cot": cond_cot, "example": cond_example, "sc": cond_sc, "noworker": cond_noworker}


def load_done():
    d = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            if l.strip():
                r = json.loads(l); d.add((r["run"], r["cond"], r["item_id"]))
    return d


def main():
    key = pd.read_csv(ANSWER_KEY)
    src = pd.concat([pd.read_csv(f) for f in SOURCE_FILES], ignore_index=True).drop_duplicates(subset=["task", "id"], keep="first")
    si = {(r["task"], int(r["id"])): r for _, r in src.iterrows()}
    done = load_done()
    jobs = [(run, cond, k) for run in range(1, N_RUNS + 1) for cond in CONDS for _, k in key.iterrows()
            if (run, cond, k["item_id"]) not in done]
    print(f"jobs={len(jobs)}")

    def work(job):
        run, cond, kr = job
        rec = {"run": run, "cond": cond, "item_id": kr["item_id"], "task": kr["task"], "id": int(kr["id"])}
        try:
            row = si[(kr["task"], int(kr["id"]))]
            rec["score"] = CONDS[cond](kr["task"], conv_block(row)); rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"; rec["error"] = str(e)[:150]
        with write_lock:
            open(OUT, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in as_completed([ex.submit(work, j) for j in jobs]):
            f.result(); n += 1
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}")
    print("done")


if __name__ == "__main__":
    main()
