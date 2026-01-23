"""Protocols for Result.raw types returned by Experiment methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from optuna.study import Study


@runtime_checkable
class ResultProtocol(Protocol):
    """Protocol for FitResult.raw - the fitted sklearn estimator."""

    raw: Any


@runtime_checkable
class SearchResultProtocol(Protocol):
    """Protocol for Study.trials - the trials of the study."""

    raw: Study
