# async-rl experiments

Use one folder per run tag:

```text
experiments/<tag>/
  results.tsv
  run.log
  logs/
  wandb/
  ray/
```

Commit `results.tsv` as the compact experiment record. Keep logs, local W&B files, Ray state, and checkpoints untracked.
