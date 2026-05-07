# nanochat experiments

Use this directory for experiment outputs inside the nanochat workstream.

Recommended layout:

```text
experiments/<run-tag>/results.tsv
experiments/<run-tag>/run.log
experiments/<run-tag>/logs/<trial-name>.log
experiments/<run-tag>/wandb/
```

Commit `results.tsv`; it is the durable experiment record. Keep `run.log`, `logs/`, and `wandb/` untracked. Those files are local execution details, verbose logs, or W&B cache/sync state.

Hyperparameter sweeps are not required for nanochat. If you run hyperparameter-only trials, record them here rather than creating one commit per trial.
