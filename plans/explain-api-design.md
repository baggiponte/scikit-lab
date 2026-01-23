---
title: Explain API Design
description: SHAP integration via Experiment.explain() method
date: 2025-01-23
---

# Explain API Design

## Goal

Add model explanation capabilities to sklab through a new `Experiment.explain()` method that computes SHAP values with zero boilerplate, following the same inject-once-forget philosophy as the rest of the library.

## The Problem

Computing SHAP values requires:
1. Choosing the right explainer for your model type
2. Managing background data sampling
3. Handling the explainer → values → visualization pipeline
4. Integrating with experiment tracking (logging plots/values)

This is tedious boilerplate that sklab should remove.

## Design Principles

1. **One method, one job** — `explain()` does SHAP. Nothing else.
2. **SHAP does SHAP well** — Expose the `shap.Explanation` object. The `plot()` method is a thin passthrough to `shap.plots`, not a wrapper.
3. **Separate concerns** — Explanation (why this prediction?) is distinct from diagnostics (how well does it perform?) and feature analysis (what features matter globally?).
4. **Compute by default, plot on demand** — `explain()` computes values only. Plotting is user-initiated via `result.plot()`. No automatic plot logging (avoids matplotlib dependency issues in headless environments).

## API Design

### Type Definitions

```python
from enum import StrEnum, auto

class ExplainerMethod(StrEnum):
    """SHAP explainer selection strategy."""
    AUTO = auto()        # Select based on estimator type
    TREE = auto()        # TreeExplainer (RF, GBM, XGBoost, LightGBM, CatBoost)
    LINEAR = auto()      # LinearExplainer (LogisticRegression, Ridge, Lasso)
    KERNEL = auto()      # KernelExplainer (model-agnostic, slow)
    DEEP = auto()        # DeepExplainer (neural networks)

class ModelOutput(StrEnum):
    """What model output to explain."""
    AUTO = auto()        # probability for classifiers with predict_proba, raw otherwise
    RAW = auto()         # Raw model output (logits, regression values)
    PROBABILITY = auto() # Class probabilities (classifiers only)
    LOG_ODDS = auto()    # Log-odds (classifiers only, for coefficient comparison)
```

### Core Method

```python
class Experiment:
    def explain(
        self,
        X,
        *,
        method: ExplainerMethod | str = ExplainerMethod.AUTO,
        model_output: ModelOutput | str = ModelOutput.AUTO,
        background: ArrayLike | int | None = None,
        feature_names: Sequence[str] | None = None,
        run_name: str | None = None,
    ) -> ExplainResult:
        """
        Compute SHAP values for the fitted estimator.

        Parameters
        ----------
        X : array-like
            Samples to explain.
        method : ExplainerMethod or str, default="auto"
            Explainer type. "auto" selects based on estimator structure.
        model_output : ModelOutput or str, default="auto"
            What model output to explain. "auto" uses probability for classifiers
            with predict_proba, raw output otherwise. Use "log_odds" when comparing
            SHAP values to logistic regression coefficients.
        background : array-like or int, optional
            Background data for KernelExplainer/etc. If int, samples that many
            rows from X. If None, uses shap.maskers.Independent or model-appropriate default.
        feature_names : sequence of str, optional
            Feature names to use. If None, attempts to infer from pipeline
            transformers (best-effort; may fall back to generic names).
        run_name : str, optional
            Name for the logged run.

        Returns
        -------
        ExplainResult
            SHAP explanation with values, base values, and feature names.
        """
```

### Result Type

```python
@dataclass(slots=True)
class ExplainResult:
    """SHAP explanation result."""

    values: np.ndarray
    """SHAP values array. Shape: (n_samples, n_features) or (n_samples, n_features, n_classes)."""

    base_values: np.ndarray
    """Expected value(s) of the model output."""

    data: np.ndarray
    """The input data that was explained."""

    feature_names: list[str] | None
    """Feature names if available."""

    raw: shap.Explanation
    """The underlying shap.Explanation object for advanced use."""

    def plot(self, kind: str = "summary", **kwargs) -> None:
        """
        Convenience wrapper around SHAP plotting functions.

        Parameters
        ----------
        kind : str
            Plot type: "summary", "bar", "beeswarm", "waterfall", "force", "dependence".
        **kwargs
            Passed to the underlying shap plot function.
        """
        # Delegates to shap.plots.{kind}(self.raw, **kwargs)
```

### Explainer Selection ("auto" mode)

| Estimator Type | Explainer |
|----------------|-----------|
| Tree-based (RF, GBM, XGBoost, LightGBM, CatBoost) | `TreeExplainer` |
| Linear models (LogisticRegression, LinearRegression, Ridge, Lasso) | `LinearExplainer` |
| Neural networks (Keras, PyTorch via skorch) | `DeepExplainer` (if available) |
| Everything else | `KernelExplainer` (with background sampling) |

Detection uses **structural checks** (attributes like `tree_`, `coef_`) and sklearn's `is_classifier`/`is_regressor`, not brittle class name matching:

```python
from sklearn.base import is_classifier, is_regressor

def _select_explainer_method(estimator) -> ExplainerMethod:
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


def _default_model_output(estimator) -> ModelOutput:
    """Select model output based on estimator type."""
    if is_classifier(estimator) and hasattr(estimator, "predict_proba"):
        return ModelOutput.PROBABILITY
    return ModelOutput.RAW
```

### Pipeline Handling

For sklearn Pipelines:
1. Extract the final estimator
2. Transform X through all preprocessing steps
3. Apply explainer to final estimator with transformed X
4. Attempt to recover feature names (best-effort)

```python
# Internal logic
if hasattr(estimator, "steps"):  # It's a Pipeline
    preprocessor = Pipeline(estimator.steps[:-1])
    final_estimator = estimator.steps[-1][1]
    X_transformed = preprocessor.transform(X)
    explainer = select_explainer(final_estimator, X_transformed, method, background)
else:
    explainer = select_explainer(estimator, X, method, background)
```

#### Feature Name Recovery (Best-Effort)

Feature names after transformation are recovered using `get_feature_names_out()` (sklearn 1.0+):

```python
def _get_feature_names(preprocessor, X, user_provided: Sequence[str] | None) -> list[str] | None:
    """Attempt to recover feature names after preprocessing."""
    # 1. User override takes precedence
    if user_provided is not None:
        return list(user_provided)

    # 2. Try sklearn's get_feature_names_out
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except (ValueError, AttributeError):
            pass  # Some transformers don't support this

    # 3. Fall back to generic names
    n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
    return [f"x{i}" for i in range(n_features)]
```

**Limitations:**
- `ColumnTransformer` with `passthrough` and missing `get_feature_names_out` may fail
- Some custom transformers don't implement `get_feature_names_out`
- When recovery fails, generic names (`x0`, `x1`, ...) are used

**Mitigation:** Users can always pass explicit `feature_names` to override inference.

### Logger Integration

When a logger is configured, `explain()` logs **metrics only** (no plots):

```python
# Inside explain()
with self.logger.start_run(name=run_name, config={}, tags=self.tags):
    # ... compute shap_values ...

    # Log mean |SHAP| per feature as importance metrics
    mean_abs_shap = _compute_mean_abs_shap(shap_values)
    self.logger.log_metrics({
        f"shap_importance/{name}": float(val)
        for name, val in zip(feature_names, mean_abs_shap)
    })
```

#### Multi-class SHAP Value Handling

SHAP returns different shapes for binary vs multiclass:
- Binary/regression: `ndarray` of shape `(n_samples, n_features)`
- Multiclass: `list[ndarray]` of length `n_classes`, each `(n_samples, n_features)`

Normalize before aggregating:

```python
def _compute_mean_abs_shap(shap_values) -> np.ndarray:
    """Compute mean |SHAP| per feature, handling multiclass."""
    if isinstance(shap_values, list):
        # Multiclass: stack to (n_classes, n_samples, n_features)
        stacked = np.stack(shap_values)
        return np.abs(stacked).mean(axis=(0, 1))  # Mean over classes and samples
    return np.abs(shap_values).mean(axis=0)
```

#### No Automatic Plot Logging

Plots are **not** logged automatically because:
- SHAP plotting returns `Axes`, not `Figure` (inconsistent API)
- Requires matplotlib (fails in headless/minimal environments)
- Users may want different plot types or customizations

To log plots manually:

```python
result = exp.explain(X_test)

# User controls plotting and logging
import matplotlib.pyplot as plt
result.plot("beeswarm")
plt.savefig("shap_summary.png")
logger.log_artifact("shap_summary.png")
```

**Future iteration:** Investigate adding optional automatic plot logging with a unified plotting API. See [Open Questions: Unified Plotting API](#unified-plotting-api).

## What This Is NOT

### Not Feature Importance

Feature importance answers: "What features matter globally?"
- Permutation importance (model-agnostic, based on performance drop)
- Coefficient magnitude (linear models)
- Tree-based importance (impurity decrease / gain)

SHAP values answer: "Why did the model make *this* prediction?"
- Local explanations that can be aggregated globally
- Accounts for feature interactions
- Theoretically grounded (Shapley values)

**Recommendation:** Feature importance belongs in a separate `Experiment.feature_importance()` method or a diagnostics API, not in `explain()`.

### Not Partial Dependence Plots

PDPs answer: "How does changing feature X affect predictions on average?"
- Shows marginal effect of a feature
- Useful for understanding learned relationships
- sklearn has `PartialDependenceDisplay`

SHAP dependence plots show similar information but with interaction coloring and are SHAP-specific.

**Recommendation:** PDPs belong in a diagnostics API alongside learning curves, calibration plots, etc.

## Proposed API Landscape

```
Experiment.fit()           → FitResult           # Train
Experiment.evaluate()      → EvalResult          # Test metrics
Experiment.cross_validate() → CVResult           # CV metrics
Experiment.search()        → SearchResult        # Hyperparameter search
Experiment.explain()       → ExplainResult       # SHAP explanations (NEW)

# Future (separate from explain):
Experiment.diagnose()      → DiagnosticsResult   # Performance plots, calibration, etc.
Experiment.analyze()       → AnalysisResult      # Feature importance, PDPs, ICE
```

Or, if we want to keep the API minimal:

```
# Alternative: Single diagnostics entry point
Experiment.diagnose(
    X, y,
    include=["metrics", "importance", "pdp", "calibration", ...]
) → DiagnosticsResult
```

## Edge Cases

### Multi-output / Multi-class Models

Store `values` consistently as 3D array `(n_samples, n_features, n_outputs)`:
- Binary classification: `n_outputs=1` (single column)
- Multiclass: `n_outputs=n_classes`
- Regression: `n_outputs=1`

This avoids users having to handle both shapes. The `raw` shap.Explanation preserves SHAP's native format for advanced use.

### No Fitted Estimator
Raise `ValueError("Call fit() or cross_validate(refit=True) before explain()")`.

### Unsupported Estimator
Fall back to `KernelExplainer` with a warning about computational cost.

### Large X
For `KernelExplainer`, default background to `shap.kmeans(X, 100)` or similar summarization.

## Dependencies

SHAP is an optional dependency:
```toml
[project.optional-dependencies]
shap = ["shap>=0.42"]
```

Use the existing `LazyModule` pattern (see `src/sklab/_lazy.py`):

```python
# In src/sklab/_explain.py (or wherever explain logic lives)
from sklab._lazy import LazyModule

shap = LazyModule("shap", install_hint="Install with: pip install sklab[shap]")

# Usage - import happens on first attribute access
def _create_explainer(estimator, X, method):
    if method == "tree" or _is_tree_model(estimator):
        return shap.TreeExplainer(estimator)
    # ...
```

This matches how `mlflow` and `wandb` are handled in the logging adapters.

## Related: Composite Runs

To group `search()` + `explain()` under a single logged run, see `plans/run-context-api-design.md` for the `Experiment.run()` context manager proposal.

```python
# With run context (proposed)
with exp.run(name="grid-search-with-explanation") as e:
    result = e.search(GridSearchConfig(...), X, y)
    explanation = e.explain(X_test)
    # Both logged as nested runs under one parent
```

## How to Test

1. **Unit tests** for explainer selection logic (mock estimators with specific attributes)
2. **Integration tests** with real pipelines:
   - `LogisticRegression` → `LinearExplainer`
   - `RandomForestClassifier` → `TreeExplainer`
   - `SVC(kernel='rbf')` → `KernelExplainer`
3. **Pipeline tests** verifying preprocessing is handled correctly
4. **Logger tests** verifying artifacts and metrics are logged
5. **Edge case tests** for unfitted estimator, empty X, etc.

## Future Considerations

1. **Caching** — SHAP computation is expensive. Consider optional result caching.
2. **Sampling** — For large datasets, option to explain a sample.
3. **Batch explanations** — Stream explanations for very large X.
4. **Interaction values** — `shap_interaction_values` for tree models.
5. **Text/image data** — Would require different maskers and explainers.

## References

- [SHAP documentation](https://shap.readthedocs.io/)
- [Lundberg & Lee, 2017 - "A Unified Approach to Interpreting Model Predictions"](https://arxiv.org/abs/1705.07874)
- [sklearn inspection module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection) (permutation importance, PDPs)

---

## Open Questions

### Unified Plotting API

Currently, `result.plot()` is a thin passthrough to `shap.plots`. However, SHAP's plotting API has inconsistencies:
- Some functions return `Axes`, others return `Figure`
- Different functions have different parameter names
- Behavior varies between plot types

**Question:** Should sklab provide a unified plotting API that normalizes these inconsistencies?

**Option A: Thin passthrough (current design)**
- Pro: No maintenance burden, users get full SHAP flexibility
- Con: Users deal with SHAP's inconsistencies

**Option B: Unified wrapper**
- Pro: Consistent return types, consistent parameters, easier logging
- Con: Maintenance burden, may lag behind SHAP updates, limits advanced usage
- Would require matplotlib as a dependency (with shap extra)

**Decision:** Deferred. Start with thin passthrough, evaluate user feedback.

---

## Summary

| Question | Answer |
|----------|--------|
| What does `explain()` do? | Computes SHAP values for the fitted estimator |
| What does it return? | `ExplainResult` with values, base_values, data, feature_names, and raw `shap.Explanation` |
| Does it wrap SHAP plotting? | Thin passthrough only; exposes `raw` for full control |
| How is the explainer selected? | Structural checks (`tree_`, `coef_`, module name), returns `ExplainerMethod` StrEnum |
| How is model output selected? | `is_classifier()` + `predict_proba` check, returns `ModelOutput` StrEnum |
| What about feature names? | Best-effort via `get_feature_names_out()`, user can override |
| What about multiclass? | Values normalized to 3D array `(n_samples, n_features, n_outputs)` |
| Does it log plots? | No (future iteration). Users log plots manually. |
| What about feature importance? | Separate concern → separate method or diagnostics API |
| What about PDPs? | Separate concern → diagnostics API |
| Is SHAP required? | Optional dependency via `LazyModule` |
