# Run-to-Run Variance in a Plan-and-Execute LLM Judge (code and run outputs)

Code and data for the paper "Run-to-Run Variance in a Plan-and-Execute LLM
Judge: A Stage-Wise Study on Multi-Turn Dialogue" by Heounmo Go and Sanghyun
Park (Yonsei University). The release contains every script
used to produce the paper's results, the prompts, the item lists, and the raw
outputs of every run, so all tables can be recomputed without calling any API.

## Requirements

Python 3.11 and the packages in `requirements.txt`. The analysis scripts need
only pandas, NumPy, and SciPy. Rerunning experiments also needs API access:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `GEMINI_API_KEY` (read from the
environment, never stored in code).

## Reproducing the tables from the saved outputs (no API calls)

Run each command from this folder.

| Result in the paper | Command |
|---|---|
| Reliability of nine judge designs and the second MT-Bench-101 set (Sec. 4) | `python reference_free_reliability.py` |
| Stage-fixed design (appendix table) and its framing control | `python analyze_stage_fixed.py` |
| Nested variance decomposition (Sec. 5) and averaging designs (Sec. 8) | `python analyze_nested.py` |
| Backends x benchmarks (Sec. 6) | `python analyze_crossbench.py` |
| Solver rerun on stored inputs, 80% (Sec. 4) | `python analyze_solver_repro.py` |
| Critique-Score judge (Sec. 6) | `python analyze_cs.py` |
| Two-system comparison, simulated (appendix) | `python analyze_system_ranking.py` |
| Ten repeated complete evaluations (appendix) | `python analyze_independent.py`, `python direct_session_check.py` |
| ICC and variance-component intervals | `python uncertainty_tables.py` |
| Failed-call counts (appendix) | `python failure_counts.py` |
| Six-system leaderboard, ten runs (Sec. 7) | `python analyze_leaderboard.py` |
| Paired PWS-Direct leaderboard differences | `python analyze_leaderboard_diff.py` |
| Interventions, MT-Bench-101 / MT-Bench (Sec. 8) | `python analyze_interventions.py`, then with `DATASET=mtbench` |
| Dollar cost of averaging designs | `python averaging_costs.py` (uses `output_token_accounting.json`) |

The human ratings discussed in the paper's appendix on human ratings are not included in this release,
so that appendix's numbers cannot be recomputed from these files. The paper
makes no claim that depends on them.

## Applying the protocol to your own judge

`nested_components.py` computes the stage-wise variance components of Sec. 3.5
for any judge whose stages are sampled one after another. It does not depend on
our prompts or backends.

1. For each item, draw P fresh first-stage outputs (e.g. plans or critiques),
   W second-stage draws on each, and S final scoring calls on each draw. For a
   two-stage judge such as Critique-Score, use W = 1.
2. Save one row per final score with the columns `item`, `plan`, `worker`,
   `solver`, `score` (CSV or JSONL). Optionally add rows scored without the
   middle stage, with `worker` set to `noW`, to estimate amplification.
3. Run `python nested_components.py scores.csv`.

It prints the Planner, Worker, and Solver components with item-bootstrap 95%
intervals, the single-run variance, the amplification ratio when `noW` rows are
present, and the residual variance of common averaging designs (Eq. 7). On our
released outputs it reproduces the nested decomposition table and the
Critique-Score components exactly.

## Rerunning the experiments (calls the APIs)

| Output file | Script | What it runs |
|---|---|---|
| `output_pipeline_variance.jsonl` | `repeat_pipeline_variance.py` | PWS judge, 5 runs, first MT-Bench-101 set |
| `output_direct_variance.jsonl` | `direct_variance.py` | Direct judge, 5 runs |
| `output_conditions_variance.jsonl` | `conditions_variance.py` | rubric, CoT, self-consistency, example, Planner+Solver, 5 runs each |
| `output_geval.jsonl` | `geval_baseline.py` | G-Eval with probability-weighted scoring, 5 runs |
| `output_jointmatched_variance.jsonl` | `jointmatched_variance.py` (uses `direct_jointmatched_baseline.py`) | Direct + rubric + example + CoT, 5 runs |
| `output_failure_sample*.jsonl` | `build_failure_sample*.py`, `run_failure_sample.py` | second MT-Bench-101 set (299 dialogues), Direct and PWS, 5 runs |
| `output_stage_fixed.jsonl` | `stage_fixed_variance.py` | S / S2 / WS conditions from stored intermediates |
| `output_nested_variance.jsonl` | `nested_variance.py` | nested design, GPT-4o-mini |
| `output_mm_nested_{haiku,gemini}.jsonl` | `multimodel_nested.py` (`PROVIDER=anthropic` or `gemini`) | nested design + Direct, other backends, MT-Bench-101 |
| `output_mtbench_nested_*.jsonl` | `multimodel_nested.py` (`DATASET=mtbench`, `PROVIDER=openai|anthropic|gemini`; Claude also `MTB_SUBSET=balanced80`) | MT-Bench runs |
| `data/mtbench_dialogues.jsonl` | `generate_mtbench_answers.py` | two-turn MT-Bench answers from GPT-4o and GPT-5.4-nano |
| `mtbench_planner_categories.json` | `build_mtbench_prompts.py` | the eight MT-Bench Planner instructions |
| `output_diag_storedprocessed_solver.jsonl` | `diag_solver_on_stored_processed.py` | Solver rerun on the stored Solver prompts |
| `output_cs_nested.jsonl` | `cs_judge_nested.py` | Critique-Score judge, 4 critiques x 3 scorer calls |
| `output_independent_runs.jsonl` | `independent_runs.py` | ten complete MT-Bench evaluations, PWS and Direct |
| `output_intervention_{short,prob,both,ronly}_mtb101.jsonl`, `output_intervention_prob_mtbench.jsonl` | `intervention_variance.py` (`VARIANT`, `DATASET`) | interventions, nested 3x2x2 design |
| additional systems in `data/mtbench_dialogues.jsonl` | `generate_mtbench_answers_more.py` | answers from four more systems for the leaderboard |
| `output_token_accounting.json` | `token_accounting.py` | token counts per call type on 20 dialogues per benchmark |

Design sizes for `multimodel_nested.py` are set with the environment variables
`NP`, `NW`, `NS`, `NR`, and `ND` (plans, Worker draws per plan, Solver calls per
Worker draw, evidence-free Solver calls per plan, Direct runs). The values
used for each run are given in the caption of the backends table (Sec. 6).

All judges ran at temperature 0 (self-consistency samples at 0.7) in October 2026 with these snapshots:
`gpt-4o-mini-2024-07-18`, `claude-haiku-4-5-20251001`, and
`gemini-3.1-flash-lite` (thinking disabled). Hosted models are not
deterministic, so reruns will not reproduce the saved outputs exactly. That
is the subject of the paper.

## Files

- `pipeline_prompts.json`: system prompts, the Direct prompt, and the Planner,
  Worker, and Solver prompts for the 13 MT-Bench-101 categories.
- `mtbench_planner_categories.json`: the eight MT-Bench Planner instructions.
- `output_gpt-4o-mini*.csv`: the MT-Bench-101 dialogues with the stored outputs
  of the original single run (including the plans and evidence reused by the
  stage-fixed design). `master_dialogue_level_scores.csv` lists which earlier
  configurations produced a score for each dialogue.
- `human_eval_answer_key_100_DO_NOT_OPEN_YET.csv`: the IDs of the first set of
  100 dialogues (the file name dates from the blind annotation study).
- `failure_sample_manifest*.csv`: the IDs of the second set (299 dialogues).
- `data/mtbench_question.jsonl`: MT-Bench questions. `data/mtbench_dialogues.jsonl`:
  the generated dialogues (two systems for the nested runs, six for the leaderboard).

## Licenses and attribution

- MT-Bench-101 (Bai et al., ACL 2024) is distributed under the Apache License 2.0
  at github.com/mtbench101/mt-bench-101. The dialogues in `output_gpt-4o-mini*.csv`
  come from its `mtbench101.jsonl`.
- MT-Bench questions (Zheng et al., 2023) come from FastChat, Apache License 2.0.
- Our code and outputs are released under the Apache License 2.0.
