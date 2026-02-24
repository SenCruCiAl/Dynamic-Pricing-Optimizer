"""Revenue optimization helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimal price search."""

    optimal_price: float
    expected_demand: float
    expected_revenue: float
    success: bool


def revenue_function(price: float, demand_predictor: Callable[[np.ndarray], np.ndarray]) -> float:
    """Compute revenue for a single price using demand predictor."""
    demand = float(demand_predictor(np.array([price]))[0])
    demand = max(demand, 0.0)
    return price * demand


def optimize_price(
    demand_predictor: Callable[[np.ndarray], np.ndarray],
    min_price: float,
    max_price: float,
) -> OptimizationResult:
    """Maximize revenue with bounded price using scipy optimize."""

    def objective(price: float) -> float:
        return -revenue_function(price, demand_predictor)

    optimization = minimize_scalar(objective, bounds=(min_price, max_price), method="bounded")
    best_price = float(optimization.x)
    best_demand = float(demand_predictor(np.array([best_price]))[0])
    best_revenue = revenue_function(best_price, demand_predictor)

    result = OptimizationResult(
        optimal_price=best_price,
        expected_demand=max(best_demand, 0.0),
        expected_revenue=best_revenue,
        success=bool(optimization.success),
    )

    LOGGER.info(
        "Optimization complete. success=%s optimal_price=%.2f expected_revenue=%.2f",
        result.success,
        result.optimal_price,
        result.expected_revenue,
    )
    return result
