"""Demand model training and elasticity estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

LOGGER = logging.getLogger(__name__)


@dataclass
class DemandModelArtifacts:
    """Container for trained models and metadata."""

    linear_model: LinearRegression
    rf_model: RandomForestRegressor
    selected_model_name: str


class DemandModeler:
    """Train demand estimators and expose prediction helpers."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.artifacts: DemandModelArtifacts | None = None

    def fit(self, data: pd.DataFrame) -> DemandModelArtifacts:
        """Fit linear and random forest regressors on price -> demand."""
        x = data[["price"]]
        y = data["demand"]

        linear = LinearRegression()
        linear.fit(x, y)

        rf = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=self.random_state,
        )
        rf.fit(x, y)

        linear_r2 = linear.score(x, y)
        rf_r2 = rf.score(x, y)
        selected = "random_forest" if rf_r2 >= linear_r2 else "linear_regression"

        self.artifacts = DemandModelArtifacts(
            linear_model=linear,
            rf_model=rf,
            selected_model_name=selected,
        )

        LOGGER.info(
            "Trained models. linear_r2=%.4f, rf_r2=%.4f, selected=%s",
            linear_r2,
            rf_r2,
            selected,
        )
        return self.artifacts

    def predict(self, prices: np.ndarray) -> np.ndarray:
        """Predict demand for given prices using selected model."""
        if self.artifacts is None:
            raise RuntimeError("Model must be fitted before prediction.")

        x = np.array(prices).reshape(-1, 1)
        if self.artifacts.selected_model_name == "random_forest":
            return self.artifacts.rf_model.predict(x)
        return self.artifacts.linear_model.predict(x)

    @staticmethod
    def estimate_elasticity(data: pd.DataFrame) -> float:
        """Estimate price elasticity via log-log linear regression."""
        filtered = data[(data["price"] > 0) & (data["demand"] > 0)]
        x = np.log(filtered["price"].to_numpy()).reshape(-1, 1)
        y = np.log(filtered["demand"].to_numpy())

        model = LinearRegression()
        model.fit(x, y)
        elasticity = float(model.coef_[0])
        LOGGER.info("Estimated log-log elasticity coefficient: %.4f", elasticity)
        return elasticity

    def save(self, output_path: str) -> None:
        """Persist trained artifacts via joblib."""
        if self.artifacts is None:
            raise RuntimeError("Cannot save before fitting model.")

        payload = {
            "selected_model_name": self.artifacts.selected_model_name,
            "linear_model": self.artifacts.linear_model,
            "rf_model": self.artifacts.rf_model,
        }
        joblib.dump(payload, output_path)
        LOGGER.info("Saved demand model artifacts to %s", output_path)
