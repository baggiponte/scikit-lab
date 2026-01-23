"""SHAP explanation support for Experiment."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from sklab._lazy import LazyModule

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

shap = LazyModule("shap", install_hint="Install with: pip install sklab[shap]")


class ExplainerMethod(StrEnum):
    """SHAP explainer selection strategy."""

    AUTO = auto()
    TREE = auto()
    LINEAR = auto()
    KERNEL = auto()
    DEEP = auto()


class ModelOutput(StrEnum):
    """What model output to explain."""

    AUTO = auto()
    RAW = auto()
    PROBABILITY = auto()
    LOG_ODDS = auto()


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

    def plot(self, kind: str = "beeswarm", **kwargs: Any) -> None:
        """Thin passthrough to shap.plots.

        Parameters
        ----------
        kind : str, default="beeswarm"
            Plot type: "summary", "bar", "beeswarm", "waterfall", "force", "dependence".
        **kwargs
            Passed to the underlying shap plot function.
        """
        try:
            plot_fn = getattr(shap.plots, kind)
        except AttributeError as exc:
            valid = ["summary", "bar", "beeswarm", "waterfall", "force", "dependence"]
            raise ValueError(
                f"Unknown plot kind {kind!r}. Valid options: {valid}"
            ) from exc
        plot_fn(self.raw, **kwargs)


def _select_explainer_method(estimator: Any) -> ExplainerMethod:
    """Select SHAP explainer based on estimator structure."""
    # 1. Check for tree structure (sklearn trees and ensembles)
    if hasattr(estimator, "tree_"):
        return ExplainerMethod.TREE
    if hasattr(estimator, "estimators_"):
        first = estimator.estimators_[0]
        check = first.flat[0] if isinstance(first, np.ndarray) else first
        if hasattr(check, "tree_"):
            return ExplainerMethod.TREE

    # 2. Check for linear structure
    if hasattr(estimator, "coef_"):
        return ExplainerMethod.LINEAR

    # 3. Check external libraries by module (not class name)
    module = type(estimator).__module__
    if any(lib in module for lib in ("xgboost", "lightgbm", "catboost")):
        return ExplainerMethod.TREE
    if any(lib in module for lib in ("keras", "tensorflow", "torch")):
        return ExplainerMethod.DEEP

    # 4. Fallback to kernel (model-agnostic)
    return ExplainerMethod.KERNEL


def _default_model_output(estimator: Any) -> ModelOutput:
    """Select model output based on estimator type."""
    if is_classifier(estimator) and hasattr(estimator, "predict_proba"):
        return ModelOutput.PROBABILITY
    return ModelOutput.RAW


def _validate_model_output(estimator: Any, model_output: ModelOutput) -> None:
    """Validate model_output is compatible with estimator type."""
    if model_output in (ModelOutput.PROBABILITY, ModelOutput.LOG_ODDS):
        if not is_classifier(estimator):
            raise ValueError(
                f"model_output={model_output!r} requires a classifier, "
                f"but got {type(estimator).__name__} (a regressor)."
            )
        if model_output == ModelOutput.PROBABILITY and not hasattr(
            estimator, "predict_proba"
        ):
            raise ValueError(
                f"model_output='probability' requires predict_proba, "
                f"but {type(estimator).__name__} does not have it."
            )


def _get_feature_names(
    estimator: Any,
    X: Any,
    user_provided: Sequence[str] | None,
) -> list[str]:
    """Attempt to recover feature names after preprocessing."""
    # 1. User override takes precedence
    if user_provided is not None:
        return list(user_provided)

    # 2. Try to get from DataFrame columns
    if hasattr(X, "columns"):
        return list(cast(Any, X).columns)

    # 3. Try sklearn's get_feature_names_out on pipeline
    if isinstance(estimator, Pipeline) and len(estimator.steps) > 1:
        # Get preprocessor (all but last step)
        preprocessor = Pipeline(estimator.steps[:-1])
        if hasattr(preprocessor, "get_feature_names_out"):
            try:
                return list(preprocessor.get_feature_names_out())
            except (ValueError, AttributeError):
                pass

    # 4. Fall back to generic names
    X_arr = np.asarray(X)
    n_features: int = X_arr.shape[1]
    return [f"x{i}" for i in range(n_features)]


def _compute_mean_abs_shap(shap_values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Compute mean |SHAP| per feature, handling multiclass."""
    if isinstance(shap_values, list):
        # Multiclass: stack to (n_classes, n_samples, n_features)
        stacked = cast(np.ndarray, np.stack(cast(Any, shap_values)))
        result: np.ndarray = np.abs(stacked).mean(axis=(0, 1))
        return result
    return np.abs(shap_values).mean(axis=0)


def _normalize_shap_values(shap_values: np.ndarray | list[np.ndarray]) -> np.ndarray:
    """Normalize SHAP values to consistent 3D shape (n_samples, n_features, n_outputs)."""
    if isinstance(shap_values, list):
        # Multiclass: list of (n_samples, n_features) -> (n_samples, n_features, n_classes)
        stacked = cast(np.ndarray, np.stack(cast(Any, shap_values), axis=-1))
        return stacked
    if shap_values.ndim == 2:
        # Binary/regression: (n_samples, n_features) -> (n_samples, n_features, 1)
        return shap_values[:, :, np.newaxis]
    return shap_values


def _extract_final_estimator(estimator: Any) -> tuple[Any, Any | None]:
    """Extract final estimator and preprocessor from pipeline.

    Returns:
        (final_estimator, preprocessor) where preprocessor is None if not a pipeline.
    """
    if isinstance(estimator, Pipeline) and len(estimator.steps) > 1:
        preprocessor = Pipeline(estimator.steps[:-1])
        final_estimator = estimator.steps[-1][1]
        return final_estimator, preprocessor
    return estimator, None


def _get_predict_fn(
    estimator: Any, model_output: ModelOutput
) -> tuple[Any, str | None]:
    """Get the prediction function based on model_output.

    Returns:
        (predict_fn, link) where link is for KernelExplainer.
    """
    if model_output == ModelOutput.PROBABILITY:
        return estimator.predict_proba, "identity"
    if model_output == ModelOutput.LOG_ODDS:
        # Use predict_proba but with logit link
        return estimator.predict_proba, "logit"
    # RAW
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function, "identity"
    return estimator.predict, "identity"


def _create_explainer(
    estimator: Any,
    X_background: ArrayLike,
    method: ExplainerMethod,
    model_output: ModelOutput,
) -> Any:
    """Create the appropriate SHAP explainer."""
    if method == ExplainerMethod.TREE:
        # TreeExplainer handles model_output internally
        return shap.TreeExplainer(estimator)

    if method == ExplainerMethod.LINEAR:
        return shap.LinearExplainer(estimator, X_background)

    if method == ExplainerMethod.DEEP:
        return shap.DeepExplainer(estimator, X_background)

    # KERNEL - model agnostic
    predict_fn, link = _get_predict_fn(estimator, model_output)
    warnings.warn(
        f"Using KernelExplainer for {type(estimator).__name__}. "
        "This may be slow for large datasets.",
        UserWarning,
        stacklevel=4,
    )
    return shap.KernelExplainer(predict_fn, X_background, link=link)


def _sample_background(X: ArrayLike, n_samples: int) -> np.ndarray:
    """Sample background data from X."""
    X_arr = np.asarray(X)
    n_available = X_arr.shape[0]
    if n_samples > n_available:
        raise ValueError(
            f"background={n_samples} is larger than available samples ({n_available})."
        )
    indices = np.random.choice(n_available, size=n_samples, replace=False)
    return X_arr[indices]


def compute_shap_explanation(
    estimator: Any,
    X: ArrayLike,
    *,
    method: ExplainerMethod | str = ExplainerMethod.AUTO,
    model_output: ModelOutput | str = ModelOutput.AUTO,
    background: ArrayLike | int | None = None,
    feature_names: Sequence[str] | None = None,
) -> ExplainResult:
    """Compute SHAP values for an estimator.

    This is the core computation function used by Experiment.explain().
    """
    # Convert string to enum if needed
    if isinstance(method, str):
        method = ExplainerMethod(method)
    if isinstance(model_output, str):
        model_output = ModelOutput(model_output)

    # Validate inputs
    X_arr = np.asarray(X)
    if X_arr.shape[0] == 0:
        raise ValueError("X is empty (0 samples).")

    # Extract final estimator from pipeline
    final_estimator, preprocessor = _extract_final_estimator(estimator)

    # Transform X if we have a preprocessor
    if preprocessor is not None:
        X_transformed = preprocessor.transform(X)
    else:
        X_transformed = X_arr

    # Select explainer method
    if method == ExplainerMethod.AUTO:
        method = _select_explainer_method(final_estimator)

    # Select model output
    if model_output == ModelOutput.AUTO:
        model_output = _default_model_output(final_estimator)

    # Validate model output is compatible
    _validate_model_output(final_estimator, model_output)

    # Prepare background data
    if background is None:
        X_background = X_transformed
    elif isinstance(background, int):
        X_background = _sample_background(X_transformed, background)
    else:
        X_background = np.asarray(background)
        if preprocessor is not None:
            X_background = preprocessor.transform(X_background)

    # Create explainer
    explainer = _create_explainer(final_estimator, X_background, method, model_output)

    # Compute SHAP values
    explanation = explainer(X_transformed)

    # Get raw shap values from explanation
    raw_values = explanation.values

    # Normalize to 3D
    values = _normalize_shap_values(raw_values)

    # Get base values
    base_values = np.atleast_1d(explanation.base_values)

    # Get feature names
    names = _get_feature_names(estimator, X, feature_names)

    # Validate feature names length
    if names is not None and len(names) != values.shape[1]:
        raise ValueError(
            f"feature_names length ({len(names)}) does not match "
            f"number of features ({values.shape[1]}). "
            f"Expected {values.shape[1]}, got {len(names)}."
        )

    return ExplainResult(
        values=values,
        base_values=base_values,
        data=X_transformed
        if isinstance(X_transformed, np.ndarray)
        else np.asarray(X_transformed),
        feature_names=names,
        raw=explanation,
    )
