# async-rl

This workstream lets the LLM do its own autonomous fully async RL research by running experiments with slime's fully asynchronous rollout path.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist - this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The workstream is small. Read these files for full context:
   - `README.md` - workstream context and commands.
   - `prepare.py` - fixed setup, model/data download, and checkpoint conversion. Do not modify.
   - `train.py` - the main file you modify. Ray/slime launch arguments, RL hyperparameters, GPU split, and runtime configuration.
   - `fully_async_rollout.py` - persistent async rollout worker. Modify only when the experiment is explicitly about async rollout behavior.
4. **Verify runtime prerequisites**: Confirm the active environment can see 8 GPUs and has the required slime checkout, prompt data, model files, and converted checkpoints.
5. **Verify setup**: Run `python train.py --dry-run`. If required inputs are missing, run `python prepare.py`.
6. **Initialize the run results folder**: Create `experiments/<tag>/` and `experiments/<tag>/logs/`. Create `experiments/<tag>/results.tsv` with just the header row. The baseline will be recorded after the first run. Store all run outputs for this experiment under this folder.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on 8 GPUs. The baseline launch is:

```bash
python train.py
```

**What you CAN do:**
- Modify `train.py` - this is the primary file you edit. RL hyperparameters, rollout/train balance, GPU partitioning, SGLang args, checkpoint cadence, logging, and command ergonomics are fair game.
- Modify `fully_async_rollout.py` when testing an async rollout idea such as queue sizing, retry policy, collection order, worker concurrency, or staleness behavior.
- Run focused hyperparameter trials when useful, but broad sweeps are not required. Hyperparameter-only trials do not need new commits; record them in the run results folder instead.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only after setup is working.
- Change the configured model architecture, tokenizer, checkpoint target, or model config unless the human explicitly asks.
- Change the total 8-GPU budget unless the human explicitly asks.
- Change the default prompt/reward setup unless the human explicitly asks.

**The goal is simple: get the highest reward.** The primary comparison metric is final reward mean over the last completed rollout window. Throughput, queue wait time, GPU memory, crash/OOM rate, and checkpoint health are diagnostics, not the objective.

**VRAM and stability** are hard practical constraints. A change that improves reward but frequently OOMs, stalls rollout generation, or corrupts checkpoints is not a keeper.

**Simplicity criterion**: All else being equal, simpler is better. A small reward improvement that adds fragile complexity is not worth it. Conversely, removing complexity and getting equal or better reward is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the reward improvement.

**The first run**: Your very first run should always establish the baseline, so run `train.py` as is.

## Output format

Once the script finishes it should print a summary like this:

```text
---
reward:            0.412000
throughput:        128.0
rollout_seconds:   482.3
total_seconds:     517.8
num_rollout:       3000
actor_gpus:        4
rollout_gpus:      4
status:            ok
```

Extract the key local signals from a run log with:

```bash
grep -Ei "reward|score|samples/sec|tokens/sec|rollout|oom|error|exception" experiments/<tag>/run.log | tail -n 80
```

Use the final local summary as the primary source of truth. If the summary is missing, use the final matching reward/score and throughput lines from `run.log` and note that the result was reconstructed from logs.

## Logging results

When an experiment is done, log it to `experiments/<tag>/results.tsv` (tab-separated, NOT comma-separated - commas break in descriptions). Commit `results.tsv` as the durable experiment record. Keep `run.log`, `logs/`, Ray state, checkpoints, and local tracking/cache files under `experiments/<tag>/` but untracked.

The TSV has a header row and 6 columns:

```text
commit	reward	throughput	status	run_type	description
```

1. git commit hash (short, 7 chars) for the code state under test. Hyperparameter-only trials do not need a new commit; record the current base commit and include changed hyperparameters in the description.
2. final reward mean. Use `0.000000` for crashes.
3. best available throughput signal, such as samples/sec or tokens/sec. Use `0.0` for crashes or unknown values.
4. status: `keep`, `discard`, or `crash`.
5. run type: `full` or `short`.
6. short text description of what this experiment tried.

Example:

```text
commit	reward	throughput	status	run_type	description
a1b2c3d	0.412000	128.0	keep	full	baseline
b2c3d4e	0.425000	132.5	keep	full	increase rollout batch to 40
c3d4e5f	0.410000	149.0	discard	short	aggressive queue drain interval
d4e5f6g	0.000000	0.0	crash	short	raise max tokens per gpu causing OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Set `RUN_DIR=experiments/<tag>` and make sure `$RUN_DIR/logs` exists.
3. Tune `train.py`, or `fully_async_rollout.py` for async rollout experiments, with one experimental idea.
4. Run the experiment: `python train.py > "$RUN_DIR/run.log" 2>&1` (redirect everything - do NOT use tee or let output flood your context).
5. Read out the results: inspect the final summary, then grep the local log for reward, throughput, and errors.
6. If the run crashed, run `tail -n 80 "$RUN_DIR/run.log"` to read the stack trace and attempt a fix. If you cannot get the idea to work after a few attempts, give up on that idea.
7. Save the run log under `$RUN_DIR/logs/` with a short trial name so every result has an audit trail.
8. Record the results in the TSV and commit `results.tsv` as the durable experiment record. Do not commit run logs, Ray state, checkpoints, or local tracking/cache files.
9. Commit only meaningful code states that are worth preserving. Hyperparameter changes do not need a new commit per trial; if a final hyperparameter set is worth keeping, commit the chosen state once.
10. If reward improved, keep the change or continue from it.
11. If reward did not improve, or the run is unstable, or the complexity is too high for the gain, revert the trial change and continue from the prior best state.

The idea is that you are a completely autonomous fully async RL researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel stuck, re-read the in-scope files, inspect logs, try smaller controlled changes, and combine previous near-misses.

**Timeout**: Full 8-GPU runs can be long. Use the current run budget agreed at setup for comparable results. If a run stalls with no rollout progress for 30 minutes, kill it and treat it as a failure unless there is clear evidence that it is still making progress.

**Crashes**: If a run crashes (OOM, Ray failure, SGLang failure, checkpoint issue, or a bug), use your judgment. If it is easy to fix, fix it and re-run. If the idea itself is fundamentally broken, log `crash` and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer and expects you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder.
