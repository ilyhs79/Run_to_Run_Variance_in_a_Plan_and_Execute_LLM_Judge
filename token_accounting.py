# -*- coding: utf-8 -*-
"""Measure input/output tokens per call type for gpt-4o-mini on a sample of
dialogues from each benchmark (Direct, Planner, Worker, Solver, Critic, Scorer),
so averaging designs can be costed in tokens and dollars."""
import json, os, re, random
import pandas as pd
from openai import OpenAI

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = "gpt-4o-mini-2024-07-18"
N = int(os.environ.get("N_SAMPLE", "20"))
client = OpenAI()
P = json.load(open(os.path.join(BASE, "pipeline_prompts.json"), encoding="utf-8"))
SYS, SYS_EXEC = P["System_prompt"], P["System_prompt_plan_execution"]
SOLVER_COMMON, PLANNER_COMMON = P["User_prompt_solver_common"], P["User_prompt_planner_common"]
DIRECT_INSTR = P["User_prompt_multiturn"].split("<|The Start of Assistant A's Conversation")[0].strip()
CATS = {"mtb101": P["planner_categories"], "mtbench": json.load(open(os.path.join(BASE, "mtbench_planner_categories.json"), encoding="utf-8"))}
import cs_judge_nested as cs

usage = []


def call(kind, ds, system, user):
    r = client.chat.completions.create(model=MODEL, temperature=0,
                                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    usage.append({"dataset": ds, "kind": kind, "in": r.usage.prompt_tokens, "out": r.usage.completion_tokens})
    return r.choices[0].message.content or ""


def main():
    random.seed(7)
    src = pd.concat([pd.read_csv(os.path.join(BASE, f)) for f in ["output_gpt-4o-mini.csv", "output_gpt-4o-mini_1_450.csv",
                     "output_gpt-4o-mini_450_900.csv", "output_gpt-4o-mini_900_end.csv"]]).drop_duplicates(subset=["task", "id"])
    key = pd.read_csv(os.path.join(BASE, "human_eval_answer_key_100_DO_NOT_OPEN_YET.csv"))
    rows = [("mtb101", r.task, cs.conv_block(src[(src.task == r.task) & (src.id == r.id)].iloc[0])) for r in key.sample(N, random_state=7).itertuples()]
    mtb = [json.loads(l) for l in open(os.path.join(BASE, "data", "mtbench_dialogues.jsonl"), encoding="utf-8")]
    rows += [("mtbench", r["task"], cs.conv_block(r)) for r in random.sample(mtb, N)]
    for ds, task, conv in rows:
        call("direct", ds, SYS, DIRECT_INSTR + "\n\n" + conv)
        plan = call("planner", ds, SYS, PLANNER_COMMON + CATS[ds][task] + "\n" + conv)
        processed = plan
        for var, instr in re.findall(r"#(E\d+)\s*=\s*LLM\[(.*?)\]", plan, re.DOTALL):
            res = call("worker", ds, SYS_EXEC, instr)
            processed = re.sub(re.escape(f"#{var} = LLM[{instr}]"), lambda m, r=f'#{var} = """{res}"""': r, processed, count=1)
        call("solver", ds, SYS, SOLVER_COMMON + conv + "\n\n" + processed)
        crit = call("critic", ds, SYS, cs.CRITIC.format(conv=conv))
        call("scorer", ds, SYS, cs.SCORER.format(conv=conv, critique=crit))
    U = pd.DataFrame(usage)
    json.dump(usage, open(os.path.join(BASE, "output_token_accounting.json"), "w"))
    t = U.groupby(["dataset", "kind"]).agg(calls=("in", "size"), mean_in=("in", "mean"), mean_out=("out", "mean")).round(0)
    print(t.to_string())
    print("\nper-dialogue calls and tokens for one full run:")
    for ds in ["mtb101", "mtbench"]:
        u = U[U.dataset == ds]; per = u.groupby("kind").agg(tin=("in", "sum"), tout=("out", "sum")) / N
        pws = per.loc[["planner", "worker", "solver"]].sum(); d = per.loc["direct"]; c = per.loc[["critic", "scorer"]].sum()
        for lab, v in [("Direct", d), ("PWS", pws), ("CS", c)]:
            cost = v.tin * 0.15e-6 + v.tout * 0.60e-6
            print(f"  {ds:8s} {lab:6s} in={v.tin:7.0f} out={v.tout:6.0f}  cost=${cost * 1000:.3f} per 1,000 dialogues -> ${cost:.5f}/dialogue")


if __name__ == "__main__":
    main()
