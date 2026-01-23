"""Result dataclasses returned by Experiment methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from sklab._lazy import LazyModule

if TYPE_CHECKING:
    pass

shap = LazyModule("shap", install_hint="Install with: pip install sklab[shap]")

RawT = TypeVar("RawT")


@dataclass(slots=True)
class FitResult:
    """Result of a single fit run.

    Attributes:
        estimator: The fitted pipeline/estimator.
        metrics: Empty dict (fit doesn't compute metrics).
        params: Merged parameters used for fitting.
        raw: The fitted estimator (same as estimator, for API consistency).
    """

    estimator: Any
    metrics: Mapping[str, float]
    params: Mapping[str, Any]
    raw: Any


@dataclass(slots=True)
class EvalResult:
    """Result of evaluating a fitted estimator on a dataset.

    Attributes:
        metrics: Computed metric scores.
        raw: The metrics dict (same as metrics, for API consistency).
    """

    metrics: Mapping[str, float]
    raw: Mapping[str, float]


@dataclass(slots=True)
class CVResult:
    """Result of a cross-validation run.

    Attributes:
        metrics: Aggregated metrics (mean/std across folds).
        fold_metrics: Per-fold metric values.
        estimator: Final refitted estimator (if refit=True), else None.
        raw: Full sklearn cross_validate() dict, including fit_time,
            score_time, and test scores for each fold.
    """

    metrics: Mapping[str, float]
    fold_metrics: Mapping[str, list[float]]
    estimator: Any | None
    raw: Mapping[str, Any]


@dataclass(slots=True)
class SearchResult(Generic[RawT]):
    """Result of a hyperparameter search run.

    Attributes:
        best_params: Best hyperparameters found.
        best_score: Best cross-validation score achieved.
        estimator: Best estimator refitted on full data (if refit=True).
        raw: The underlying search object. For OptunaConfig, this is the
            Optuna Study with full trial history. For sklearn searchers
            (GridSearchCV, RandomizedSearchCV), this is the fitted searcher
            with cv_results_ and other attributes.
    """

    best_params: Mapping[str, Any]
    best_score: float | None
    estimator: Any | None
    raw: RawT


class PlotKind(StrEnum):
    """Available SHAP plot types."""

    SUMMARY = auto()
    BAR = auto()
    BEESWARM = auto()
    WATERFALL = auto()
    FORCE = auto()
    DEPENDENCE = auto()


@dataclass(slots=True)
class ExplainResult:
    """Result of explaining model predictions with SHAP.

    Attributes:
        values: SHAP values array. Shape: (n_samples, n_features, n_outputs).
            Always 3D for consistency across binary/multiclass/regression.
        base_values: Expected value(s) of the model output.
        data: The input data that was explained.
        feature_names: Feature names if available, else None.
        raw: The underlying shap.Explanation object for advanced use.
    """

    values: np.ndarray
    base_values: np.ndarray
    data: np.ndarray
    feature_names: list[str] | None
    raw: Any  # shap.Explanation

    def plot(self, kind: PlotKind | str = PlotKind.BEESWARM, **kwargs: Any) -> None:
        """Thin passthrough to shap.plots.

        Parameters
        ----------
        kind : PlotKind or str, default=PlotKind.BEESWARM
            Plot type: "summary", "bar", "beeswarm", "waterfall", "force", "dependence".
        **kwargs
            Passed to the underlying shap plot function.
        """
        if isinstance(kind, str):
            try:
                kind = PlotKind(kind)
            except ValueError as exc:
                valid = [p.value for p in PlotKind]
                raise ValueError(
                    f"Unknown plot kind {kind!r}. Valid options: {valid}"
                ) from exc
        try:
            plot_fn = getattr(shap.plots, kind.value)
        except AttributeError as exc:
            valid = [p.value for p in PlotKind]
            raise ValueError(
                f"Unknown plot kind {kind!r}. Valid options: {valid}"
            ) from exc
        plot_fn(self.raw, **kwargs)
