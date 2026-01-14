"""MLflow logger."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from sklab._lazy import LazyModule

mlflow = LazyModule("mlflow", install_hint="Install mlflow to use MLflowLogger.")


@dataclass
class MLflowLogger:
    """Logger that tracks experiments with MLflow.

    MLflow uses module-level functions that operate on the active run,
    so we don't need to store run state.
    """

    experiment_name: str | None = None

    @contextmanager
    def start_run(self, name=None, config=None, tags=None, nested=False):
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=name, nested=nested):
            if config:
                self.log_params(config)
            if tags:
                self.set_tags(tags)
            yield self

    def log_params(self, params) -> None:
        mlflow.log_params(dict(params))

    def log_metrics(self, metrics, step: int | None = None) -> None:
        mlflow.log_metrics(dict(metrics), step=step)

    def set_tags(self, tags) -> None:
        mlflow.set_tags(dict(tags))

    def log_artifact(self, path: str, name: str | None = None) -> None:
        if name is None:
            mlflow.log_artifact(path)
        else:
            mlflow.log_artifact(path, name=name)

    def log_model(self, model: Any, name: str | None = None) -> None:
        mlflow.sklearn.log_model(model, name=name or "model")
