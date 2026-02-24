"""Synthetic demand simulation utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class DemandSimulator:
    """Generate synthetic price-demand data.

    Demand function used:
        demand = base_demand * (price / reference_price) ** (-elasticity) + noise
    """

    def __init__(
        self,
        base_demand: float = 500.0,
        reference_price: float = 50.0,
        noise_std: float = 15.0,
        random_seed: int = 42,
    ) -> None:
        self.base_demand = base_demand
        self.reference_price = reference_price
        self.noise_std = noise_std
        self.random_seed = random_seed

    def generate(
        self,
        min_price: float,
        max_price: float,
        n_samples: int = 250,
        elasticity: float = 1.4,
    ) -> pd.DataFrame:
        """Generate a synthetic demand dataset."""
        rng = np.random.default_rng(self.random_seed)
        prices = rng.uniform(min_price, max_price, n_samples)

        deterministic_demand = self.base_demand * (
            prices / self.reference_price
        ) ** (-elasticity)
        noise = rng.normal(loc=0.0, scale=self.noise_std, size=n_samples)
        demand = np.clip(deterministic_demand + noise, 1.0, None)

        data = pd.DataFrame(
            {
                "price": prices,
                "demand": demand,
                "true_deterministic_demand": deterministic_demand,
                "noise": noise,
                "elasticity": elasticity,
            }
        ).sort_values("price", ignore_index=True)

        LOGGER.info(
            "Generated %s samples with elasticity %.3f and price range [%.2f, %.2f]",
            n_samples,
            elasticity,
            min_price,
            max_price,
        )
        return data

    @staticmethod
    def save(dataset: pd.DataFrame, output_path: Path) -> None:
        """Save simulated dataset to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path, index=False)
        LOGGER.info("Saved simulated data to %s", output_path)
