---
date: 2026-01-12
title: Logger Protocol With Context-Managed Runs
tags:
  - logging
  - mlflow
  - wandb
  - sklearn
  - experiment-runner
---

Here’s what I found and how to bake context‑manager behavior into your logger Protocol.

How existing libs behave
- Transformers uses a callback system (TrainerCallback) with event hooks (e.g., `on_train_begin`, `on_log`, `on_evaluate`, `on_train_end`) and loads integrations based on installed packages and a `report_to` list. This gives a “plugin‑style” logger surface that reacts to lifecycle events rather than owning the training loop.
- MLflow’s `mlflow.start_run()` returns an `ActiveRun` that is a context manager; it auto‑closes on `with` exit, even on exceptions.
- W&B recommends `with wandb.init() as run:`; the `Run` is the logging object and the context manager automatically finishes the run.

Design recommendation: split “Logger” vs “Run”
To preserve context‑manager semantics (mlflow/wandb), make your Protocol return a run handle that itself is a context manager.

Suggested shape (conceptual):
- `LoggerProtocol.start_run(...) -> RunProtocol`
- `RunProtocol` implements `__enter__/__exit__` and logging methods (`log_params`, `log_metrics`, `log_artifact`, `set_tags`, `log_model`, `finish(status=...)`)
- `Experiment` does:
  `with logger.start_run(...) as run: ... run.log_metrics(...)`

Why this helps:
- W&B: `start_run` can wrap `wandb.init()` and return the `wandb.Run` object (which is already a context manager).
- MLflow: `start_run` can call `mlflow.start_run()` and return the `ActiveRun` context manager; `__exit__` triggers `end_run` semantics.
- For simple logging (stdout / JSONL), your `RunProtocol` can implement no‑op `__enter__/__exit__` and just write locally.

Optional: add callback‑style hooks for fine‑grained events
Transformers uses callbacks with `on_log` / `on_train_begin` / `on_train_end` patterns. You can include optional hooks in `RunProtocol` or a separate `CallbackProtocol`, but keep the core run interface small.

API ideas (concise)
- `LoggerProtocol.start_run(name, config, tags, nested=False) -> RunProtocol`
- `RunProtocol`:
  - `__enter__`, `__exit__` (convert exceptions to “failed” status)
  - `log_params(dict)`, `log_metrics(dict, step=None)`
  - `log_artifact(path, name=None)`, `log_model(obj, name=None)`
  - `set_tags(dict)`, `finish(status="success"|"failed")`
- `CompositeLogger.start_run(...) -> CompositeRun` that fan‑outs to multiple underlying runs.

If you want, I can propose a concrete Protocol signature and a minimal adapter sketch for `wandb` and `mlflow` next.
