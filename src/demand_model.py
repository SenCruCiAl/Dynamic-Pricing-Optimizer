"""Demand model training and elasticity estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

LOGGER = logging.getLogger(__name__)


class LogLogDemand:
    """Constant-elasticity demand model: log(D) = alpha + beta * log(P)."""

    def __init__(self) -> None:
        self._reg = LinearRegression()

    def fit(self, prices: np.ndarray, demand: np.ndarray) -> "LogLogDemand":
        self._reg.fit(np.log(prices).reshape(-1, 1), np.log(demand))
        return self

    @property
    def elasticity(self) -> float:
        """Price elasticity of demand (beta). Negative for normal goods."""
        return float(self._reg.coef_[0])

    def predict(self, prices: np.ndarray) -> np.ndarray:
        return np.exp(self._reg.predict(np.log(prices).reshape(-1, 1)))


class ForestDemand:
    """Non-parametric demand model on raw price."""

    def __init__(self, random_state: int = 42) -> None:
        self._rf = RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=random_state)

    def fit(self, prices: np.ndarray, demand: np.ndarray) -> "ForestDemand":
        self._rf.fit(prices.reshape(-1, 1), demand)
        return self

    def predict(self, prices: np.ndarray) -> np.ndarray:
        return self._rf.predict(np.asarray(prices).reshape(-1, 1))


@dataclass
class DemandModelArtifacts:
    """Container for trained models and metadata."""

    models: dict
    cv_rmse: dict
    selected_model_name: str


class DemandModeler:
    """Train demand estimators and select one by out-of-sample error.

    Selection uses K-fold cross-validated RMSE. In-sample R^2 is not used: a
    random forest nearly memorises training data (R^2 ~ 1.0), so in-sample
    comparison always favours it regardless of how well it generalises.
    """

    def __init__(self, random_state: int = 42, cv_folds: int = 5) -> None:
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.artifacts: DemandModelArtifacts | None = None

    def _candidates(self) -> dict:
        return {
            "log_log": LogLogDemand,
            "random_forest": lambda: ForestDemand(random_state=self.random_state),
        }

    def _cv_rmse(self, factory, prices: np.ndarray, demand: np.ndarray) -> float:
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        errors = []
        for train_idx, test_idx in kfold.split(prices):
            model = factory().fit(prices[train_idx], demand[train_idx])
            residual = model.predict(prices[test_idx]) - demand[test_idx]
            errors.append(np.sqrt(np.mean(residual**2)))
        return float(np.mean(errors))

    def fit(self, data: pd.DataFrame) -> DemandModelArtifacts:
        """Fit every candidate, score it by CV RMSE, and keep the best."""
        prices = data["price"].to_numpy(dtype=float)
        demand = data["demand"].to_numpy(dtype=float)

        models, cv_rmse = {}, {}
        for name, factory in self._candidates().items():
            cv_rmse[name] = self._cv_rmse(factory, prices, demand)
            models[name] = factory().fit(prices, demand)

        selected = min(cv_rmse, key=cv_rmse.get)
        self.artifacts = DemandModelArtifacts(models=models, cv_rmse=cv_rmse, selected_model_name=selected)

        LOGGER.info(
            "CV RMSE: %s; selected=%s",
            ", ".join(f"{k}={v:.2f}" for k, v in cv_rmse.items()),
            selected,
        )
        return self.artifacts

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Predict demand for given prices using selected model."""
        if self.artifacts is None:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.artifacts.models[self.artifacts.selected_model_name].predict(np.asarray(prices, dtype=float))

    @staticmethod
    def estimate_elasticity(data: pd.DataFrame) -> float:
        """Estimate price elasticity via log-log linear regression."""
        filtered = data[(data["price"] > 0) & (data["demand"] > 0)]
        model = LogLogDemand().fit(filtered["price"].to_numpy(), filtered["demand"].to_numpy())
        LOGGER.info("Estimated log-log elasticity coefficient: %.4f", model.elasticity)
        return model.elasticity

    def save(self, output_path: str) -> None:
        """Persist trained artifacts via joblib."""
        if self.artifacts is None:
            raise RuntimeError("Cannot save before fitting model.")
        joblib.dump(self.artifacts, output_path)
        LOGGER.info("Saved demand model artifacts to %s", output_path)
