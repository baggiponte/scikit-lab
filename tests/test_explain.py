"""Tests for Experiment.explain() SHAP integration."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklab import Experiment, ExplainerMethod, ExplainResult, ModelOutput
from sklab._explain import (
    _compute_mean_abs_shap,
    _default_model_output,
    _select_explainer_method,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def binary_data():
    """Binary classification dataset."""
    return load_breast_cancer(return_X_y=True)


@pytest.fixture
def iris_data():
    """Multiclass classification dataset."""
    return load_iris(return_X_y=True)


@pytest.fixture
def regression_data():
    """Regression dataset."""
    return load_diabetes(return_X_y=True)


class InMemoryLogger:
    """Logger that captures all calls for test assertions."""

    def __init__(self):
        self.runs = []
        self.metrics = {}
        self.params = {}
        self.artifacts = []
        self.models = []

    def start_run(self, name=None, config=None, tags=None, nested=False):
        self.runs.append({"name": name, "config": config, "tags": tags})
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.update(metrics)

    def set_tags(self, tags):
        pass

    def log_artifact(self, path, name=None):
        self.artifacts.append(path)

    def log_model(self, model, name=None):
        self.models.append(model)


@pytest.fixture
def in_memory_logger():
    """Logger that captures all calls for assertions."""
    return InMemoryLogger()


# =============================================================================
# 1. Explainer Selection
# =============================================================================


@pytest.mark.parametrize(
    "estimator,expected",
    [
        (DecisionTreeClassifier(), ExplainerMethod.TREE),
        (RandomForestClassifier(n_estimators=2), ExplainerMethod.TREE),
        (LogisticRegression(), ExplainerMethod.LINEAR),
        (Ridge(), ExplainerMethod.LINEAR),
        (SVC(), ExplainerMethod.KERNEL),
    ],
)
def test_select_explainer_method(estimator, expected, binary_data):
    X, y = binary_data
    estimator.fit(X, y)
    assert _select_explainer_method(estimator) == expected


# =============================================================================
# 2. Model Output Selection
# =============================================================================


@pytest.mark.parametrize(
    "estimator,expected",
    [
        (LogisticRegression(), ModelOutput.PROBABILITY),
        (SVC(probability=False), ModelOutput.RAW),
        (SVC(probability=True), ModelOutput.PROBABILITY),
        (Ridge(), ModelOutput.RAW),
    ],
)
def test_default_model_output(estimator, expected):
    assert _default_model_output(estimator) == expected


# =============================================================================
# 3. Feature Name Recovery
# =============================================================================


def test_feature_names_user_override(binary_data):
    X, y = binary_data
    n_features = X.shape[1]
    custom_names = [f"feature_{i}" for i in range(n_features)]

    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:5], feature_names=custom_names)

    assert result.feature_names == custom_names


def test_feature_names_fallback_generic(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:5])

    # Should fall back to x0, x1, ...
    expected = [f"x{i}" for i in range(X.shape[1])]
    assert result.feature_names == expected


# =============================================================================
# 4. Multi-class SHAP Value Handling
# =============================================================================


def test_explain_binary_values_shape(binary_data):
    """Binary classification: values should be (n_samples, n_features, 1)."""
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.ndim == 3
    assert result.values.shape[0] == 5
    assert result.values.shape[1] == X.shape[1]


def test_explain_regression_values_shape(regression_data):
    """Regression: values should be 3D."""
    X, y = regression_data
    exp = Experiment(pipeline=Ridge())
    exp.fit(X, y)
    result = exp.explain(X[:5])

    assert result.values.ndim == 3
    assert result.values.shape[0] == 5
    assert result.values.shape[1] == X.shape[1]


# =============================================================================
# 5. Mean |SHAP| Aggregation
# =============================================================================


def test_compute_mean_abs_shap_2d():
    shap_values = np.array([[0.1, -0.2], [0.3, -0.4]])
    result = _compute_mean_abs_shap(shap_values)
    expected = np.array([0.2, 0.3])  # mean of abs values per feature
    np.testing.assert_array_almost_equal(result, expected)


def test_compute_mean_abs_shap_multiclass():
    # 2 classes, 3 samples, 2 features
    shap_values = [
        np.array([[0.1, -0.2], [0.3, -0.4], [0.5, -0.6]]),
        np.array([[0.2, -0.3], [0.4, -0.5], [0.6, -0.7]]),
    ]
    result = _compute_mean_abs_shap(shap_values)
    # Should be (2,) - one value per feature
    assert result.shape == (2,)


# =============================================================================
# 6. Integration Tests
# =============================================================================


def test_explain_logistic_regression(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:10])

    assert isinstance(result, ExplainResult)
    assert result.values.shape[0] == 10
    assert result.base_values is not None
    assert result.raw is not None


def test_explain_random_forest(binary_data):
    X, y = binary_data
    exp = Experiment(
        pipeline=RandomForestClassifier(n_estimators=10, random_state=42)
    )
    exp.fit(X, y)
    result = exp.explain(X[:10])

    assert isinstance(result, ExplainResult)


def test_explain_pipeline_with_preprocessing(binary_data):
    X, y = binary_data
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
    )
    exp = Experiment(pipeline=pipeline)
    exp.fit(X, y)
    result = exp.explain(X[:10])

    assert isinstance(result, ExplainResult)
    assert result.feature_names is not None
    assert len(result.feature_names) == X.shape[1]


def test_explain_fallback_to_kernel_with_warning(binary_data):
    """Unsupported estimator should fall back to KernelExplainer with warning."""
    X, y = binary_data
    exp = Experiment(pipeline=SVC(kernel="rbf"))
    exp.fit(X, y)

    with pytest.warns(UserWarning, match="KernelExplainer"):
        result = exp.explain(X[:3])  # Small sample for speed

    assert isinstance(result, ExplainResult)


# =============================================================================
# 7. Logger Integration
# =============================================================================


def test_explain_logs_metrics(binary_data, in_memory_logger):
    X, y = binary_data
    exp = Experiment(
        pipeline=LogisticRegression(max_iter=1000),
        logger=in_memory_logger,
    )
    exp.fit(X, y)
    exp.explain(X[:10])

    # Check that shap_importance metrics were logged
    assert any("shap_importance" in key for key in in_memory_logger.metrics)


def test_explain_logs_correct_metric_count(binary_data, in_memory_logger):
    """Should log one shap_importance metric per feature."""
    X, y = binary_data
    n_features = X.shape[1]
    exp = Experiment(
        pipeline=LogisticRegression(max_iter=1000), logger=in_memory_logger
    )
    exp.fit(X, y)
    exp.explain(X[:10])

    importance_metrics = [
        k for k in in_memory_logger.metrics if "shap_importance" in k
    ]
    assert len(importance_metrics) == n_features


def test_explain_works_without_logger(binary_data):
    """explain() should work fine with NoOpLogger (default)."""
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:10])

    assert isinstance(result, ExplainResult)


# =============================================================================
# 8. Edge Cases
# =============================================================================


def test_explain_unfitted_raises(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression())
    with pytest.raises(ValueError, match="fit"):
        exp.explain(X[:5])


def test_explain_after_cross_validate_with_refit(binary_data):
    X, y = binary_data
    exp = Experiment(
        pipeline=LogisticRegression(max_iter=1000), scoring="accuracy"
    )
    exp.cross_validate(X, y, cv=3, refit=True)
    result = exp.explain(X[:10])
    assert isinstance(result, ExplainResult)


def test_explain_after_cross_validate_without_refit_raises(binary_data):
    X, y = binary_data
    exp = Experiment(
        pipeline=LogisticRegression(max_iter=1000), scoring="accuracy"
    )
    exp.cross_validate(X, y, cv=3, refit=False)
    with pytest.raises(ValueError, match="fit"):
        exp.explain(X[:10])


def test_explain_single_sample(binary_data):
    """Explaining a single sample should work."""
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:1])

    assert result.values.shape[0] == 1
    assert isinstance(result, ExplainResult)


# =============================================================================
# 8a. Input Variations
# =============================================================================


def test_explain_pandas_dataframe(binary_data):
    """Should accept pandas DataFrame input."""
    pd = pytest.importorskip("pandas")

    X, y = binary_data
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X_df, y)
    result = exp.explain(X_df.iloc[:10])

    assert isinstance(result, ExplainResult)
    # Feature names should come from DataFrame columns
    assert result.feature_names == list(X_df.columns)


# =============================================================================
# 9. StrEnum Parameter Acceptance
# =============================================================================


@pytest.mark.parametrize(
    "method",
    [
        ExplainerMethod.LINEAR,
        "linear",
    ],
)
def test_explain_accepts_method_string_and_enum(binary_data, method):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:5], method=method)
    assert isinstance(result, ExplainResult)


@pytest.mark.parametrize(
    "model_output",
    [
        ModelOutput.PROBABILITY,
        "probability",
    ],
)
def test_explain_accepts_model_output_string_and_enum(binary_data, model_output):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:5], model_output=model_output)
    assert isinstance(result, ExplainResult)


# =============================================================================
# 10. Plot Passthrough
# =============================================================================


def test_plot_passthrough_does_not_raise(binary_data):
    plt = pytest.importorskip("matplotlib.pyplot")

    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:5])

    # Should not raise
    result.plot("bar")
    plt.close("all")


# =============================================================================
# 11. Unhappy Paths
# =============================================================================


def test_explain_invalid_method_raises(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    with pytest.raises(ValueError):
        exp.explain(X[:5], method="invalid")


def test_explain_invalid_model_output_raises(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    with pytest.raises(ValueError):
        exp.explain(X[:5], model_output="invalid")


def test_explain_probability_on_regressor_raises(regression_data):
    X, y = regression_data
    exp = Experiment(pipeline=Ridge())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="regressor"):
        exp.explain(X[:5], model_output="probability")


def test_explain_log_odds_on_regressor_raises(regression_data):
    X, y = regression_data
    exp = Experiment(pipeline=Ridge())
    exp.fit(X, y)
    with pytest.raises(ValueError, match="regressor"):
        exp.explain(X[:5], model_output="log_odds")


def test_explain_feature_names_length_mismatch_raises(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    with pytest.raises(ValueError, match="feature_names"):
        exp.explain(X[:5], feature_names=["a", "b"])  # Wrong length


def test_explain_empty_x_raises(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    with pytest.raises(ValueError, match="empty"):
        exp.explain(X[:0])


def test_explain_background_larger_than_x_raises(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    with pytest.raises(ValueError, match="background"):
        exp.explain(X[:5], background=100)


def test_plot_invalid_kind_raises(binary_data):
    X, y = binary_data
    exp = Experiment(pipeline=LogisticRegression(max_iter=1000))
    exp.fit(X, y)
    result = exp.explain(X[:5])
    with pytest.raises(ValueError, match="invalid_plot"):
        result.plot("invalid_plot")


# =============================================================================
# 12. Regression Tests (Correctness)
# =============================================================================


def test_explain_deterministic_with_seed(binary_data):
    """Same model + data should produce identical SHAP values."""
    X, y = binary_data

    exp = Experiment(pipeline=LogisticRegression(random_state=42, max_iter=1000))
    exp.fit(X, y)

    result1 = exp.explain(X[:10])
    result2 = exp.explain(X[:10])

    np.testing.assert_array_almost_equal(result1.values, result2.values)
    np.testing.assert_array_almost_equal(result1.base_values, result2.base_values)
