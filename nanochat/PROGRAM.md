# nanochat

This workstream lets the LLM do its own research by running experiments. It only requires one GPU per experiment.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
   - `eval_suite.py` — benchmark harness adapter used for periodic direction checks.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize the run results folder**: Create `experiments/<tag>/` and `experiments/<tag>/logs/`. Create `experiments/<tag>/results.tsv` with just the header row. The baseline will be recorded after the first run. Store all run outputs for this experiment under this folder.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Run focused hyperparameter trials when useful, but hyperparameter sweeps are not needed. Hyperparameter-only trials do not need new commits; record them in the run results folder instead.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the benchmark score definition in `eval_suite.py` unless the user explicitly asks to change the evaluation suite.

**The goal is simple: get the lowest `val_bpb` during run-to-run iteration.** Training still uses the fixed 5-minute wall-clock budget, then `train.py` evaluates validation bits-per-byte as the cheap comparison metric. Everything is fair game inside `train.py`: change the architecture, the optimizer, the hyperparameters, the batch size, the model size, etc. The constraints are that the code runs without crashing and the final validation evaluation completes.

Run the heavier benchmark suite only when changing experiment direction or validating a promising direction-level result, by setting `NANOCHAT_RUN_BENCHMARK=1`. Do not run the full benchmark after every small hyperparameter or implementation trial.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful `val_bpb` or benchmark gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 `val_bpb` improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 `val_bpb` improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Primary run-to-run evaluation

The fixed 5-minute training loop is governed by validation bits-per-byte (`val_bpb`) for normal run-to-run comparison. Lower is better. This keeps the autonomous loop fast enough to try many small changes.

`train.py` computes `val_bpb` after every run. When `NANOCHAT_RUN_BENCHMARK=1` is set, it also calls `eval_suite.py` and prints a higher-is-better `benchmark_score`. The benchmark score is the macro-average of normalized task scores from the default benchmark suite. Accuracy, exact-match, instruction-following, and pass@1 metrics are used directly. WikiText-2 bits-per-byte is converted to `1 / (1 + bits_per_byte)` so it can participate in the same higher-is-better score.

Recommended default benchmark suite:

- `MMLU`
- `MMLU-stem`
- `MMLU-exclude-stem`
- `GSM8k`
- `MATH_6k`
- `MATH_6k_zeroshot`
- `BBH`
- `IFEval-nile`
- `HumanEval`
- `hellaswag`
- `arc-challenge`
- `winogrande`
- `TruthfulQA`
- `lambada`
- `wikitext2`

Benchmark coverage intent:

- Knowledge and domain split: `MMLU`, `MMLU-stem`, `MMLU-exclude stem`
- Math and symbolic reasoning: `GSM8k`, `MATH_6k`, `MATH_6k_zeroshot`, `BBH`
- Instruction following and code: `IFEval-nile`, `HumanEval`
- Commonsense and robust multiple choice: `hellaswag`, `arc-challenge`, `winogrande`
- Truthfulness and language modeling: `TruthfulQA`, `lambada`, `wikitext2`

Harness mapping:

- `MMLU` -> `mmlu`
- `MMLU-stem` -> `mmlu_stem`
- `MMLU-exclude-stem` -> `mmlu_humanities`, `mmlu_social_sciences`, `mmlu_other`
- `GSM8k` -> `gsm8k`
- `MATH_6k` -> `minerva_math`
- `MATH_6k_zeroshot` -> `hendrycks_math`
- `BBH` -> `bbh_zeroshot`
- `IFEval-nile` -> `ifeval`
- `HumanEval` -> `humaneval`
- `hellaswag` -> `hellaswag`
- `arc-challenge` -> `arc_challenge`
- `winogrande` -> `winogrande`
- `TruthfulQA` -> `truthfulqa_mc1`
- `lambada` -> `lambada_openai`
- `wikitext2` -> `wikitext`

Environment controls:

- `NANOCHAT_RUN_BENCHMARK=1`: also run the benchmark suite after computing `val_bpb`. Use this when changing experiment direction or validating a promising direction, not for every small trial.
- `NANOCHAT_BENCHMARKS`: comma-separated labels or lm-eval task names. Defaults to the full suite above.
- `NANOCHAT_BENCHMARK_LIMIT`: optional per-task sample limit for smoke runs. Omit it for full evaluation.
- `NANOCHAT_BENCHMARK_BATCH_SIZE`: adapter batch size for loglikelihood tasks. Defaults to `8`.
- `NANOCHAT_BENCHMARK_MAX_GEN_TOKS`: optional cap for generation-heavy tasks.
- `NANOCHAT_BENCHMARK_PROGRESS_INTERVAL`: progress callback interval for long eval phases. Defaults to `100` requests/examples.

Defer the shopping-specific benchmarks (`Shopping-Reasoning`, `Shop MMLU v3`, `Shop M-MMLU v3`, `ShopMMLU`, `M-MMLU`, `Shop M-MMLU`) unless this workstream explicitly becomes shopping-domain focused. Defer `qasper`, `race`, `boolq`, `openbookqa`, and `piqa` unless the user asks for broader classic NLP coverage.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          2.973421
benchmark_score:  N/A
training_seconds: 300.1
total_seconds:    318.4
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            9
```

When `NANOCHAT_RUN_BENCHMARK=1` is set, `benchmark_score` is numeric and the summary also includes `benchmark/<task>` lines. When the benchmark is skipped, `benchmark_score` is `N/A`. Training is configured to stop after 5 minutes, but benchmark evaluation can take substantially longer because it happens after training. During benchmark eval, `train.py` prints `benchmark_progress/<phase>` lines. You can extract the key metrics from the log file:

```
grep "^val_bpb:\|^benchmark_score:\|^peak_vram_mb:" experiments/<tag>/run.log
```

## Logging results

When an experiment is done, log it to `experiments/<tag>/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions). Commit `results.tsv` as the durable experiment record. Keep `run.log`, `logs/`, and local tracking/cache files under `experiments/<tag>/` but untracked.

The TSV has a header row and 6 columns:

```
commit	val_bpb	benchmark_score	memory_gb	status	description
```

1. git commit hash (short, 7 chars) for the code state under test. Hyperparameter-only trials do not need a new commit; record the current base commit and include the changed hyperparameters in the description.
2. `val_bpb` achieved (e.g. 2.973421) — lower is better; use 999.000000 for crashes
3. `benchmark_score` achieved when the benchmark ran; use `N/A` when skipped or unavailable
4. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	benchmark_score	memory_gb	status	description
a1b2c3d	2.973421	N/A	44.0	keep	baseline
b2c3d4e	2.965300	N/A	44.2	keep	increase LR to 0.04
c3d4e5f	2.962100	0.263100	44.2	keep	direction check after LR increase
d4e5f6g	2.981000	N/A	44.0	discard	switch to GeLU activation
e5f6g7h	999.000000	N/A	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Set `RUN_DIR=experiments/<tag>` and make sure `$RUN_DIR/logs` exists.
3. Tune `train.py` with an experimental idea by directly hacking the code.
4. Run the experiment: `uv run train.py > "$RUN_DIR/run.log" 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^benchmark_score:\|^peak_vram_mb:" "$RUN_DIR/run.log"`
6. If the grep output is empty, the run crashed. Run `tail -n 50 "$RUN_DIR/run.log"` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Save the run log under `$RUN_DIR/logs/` with a short trial name so every result has an audit trail.
8. Record the results in the tsv and commit `results.tsv` as the durable experiment record. Do not commit run logs or local tracking/cache files.
9. Commit only meaningful code states that are worth preserving. Hyperparameter changes do not need a new commit per trial; if a final hyperparameter set is worth keeping, commit the chosen state once.
10. If `val_bpb` improved (lower), keep the change or continue from it.
11. If `val_bpb` is equal or worse, revert the trial change and continue from the prior best state.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Benchmark cadence**: Run `NANOCHAT_RUN_BENCHMARK=1 uv run train.py > "$RUN_DIR/run.log" 2>&1` when changing experiment direction or validating a promising direction. For fast smoke benchmark checks, set `NANOCHAT_BENCHMARK_LIMIT` to a small value and record that limit in the result description. Do not compare limited and full-suite scores as if they were the same metric.

**Final full eval**: At the very end of the experiment loop, before handing results back to the human, run one final full benchmark evaluation on the current best code state. Use `NANOCHAT_RUN_BENCHMARK=1 uv run train.py > "$RUN_DIR/run.log" 2>&1` with no `NANOCHAT_BENCHMARK_LIMIT`, save the log under `$RUN_DIR/logs/`, record both `val_bpb` and the numeric `benchmark_score` in `results.tsv`, and commit the updated `results.tsv`.

**Timeout**: Training plus `val_bpb` should be relatively quick after the fixed ~5-minute training window. Full benchmark evaluation can take much longer because it includes generation-heavy tasks such as IFEval and HumanEval.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
