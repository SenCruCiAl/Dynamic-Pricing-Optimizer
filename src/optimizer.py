"""Price optimization helpers."""

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
    expected_profit: float
    success: bool


def profit_function(
    price: float,
    demand_predictor: Callable[[np.ndarray], np.ndarray],
    unit_cost: float = 0.0,
) -> float:
    """Profit (price - unit_cost) * demand for a single price. unit_cost=0 gives revenue."""
    demand = max(float(demand_predictor(np.array([price]))[0]), 0.0)
    return (price - unit_cost) * demand


def optimize_price(
    demand_predictor: Callable[[np.ndarray], np.ndarray],
    min_price: float,
    max_price: float,
    unit_cost: float = 0.0,
    grid_size: int = 400,
) -> OptimizationResult:
    """Maximize profit over [min_price, max_price].

    A bounded scalar optimizer alone assumes the objective is unimodal. That
    fails for step-shaped models (random forests) and for corner optima, so a
    dense grid search finds the best bracket first and Brent's method refines
    inside it. The bounds are always evaluated as candidates.
    """
    grid = np.linspace(min_price, max_price, grid_size)
    demand = np.maximum(demand_predictor(grid), 0.0)
    profits = (grid - unit_cost) * demand
    best_idx = int(np.argmax(profits))

    lo = grid[max(best_idx - 1, 0)]
    hi = grid[min(best_idx + 1, grid_size - 1)]
    refined = minimize_scalar(
        lambda p: -profit_function(p, demand_predictor, unit_cost),
        bounds=(lo, hi),
        method="bounded",
    )

    candidates = [(float(grid[best_idx]), float(profits[best_idx]))]
    if refined.success:
        candidates.append((float(refined.x), -float(refined.fun)))
    best_price, best_profit = max(candidates, key=lambda c: c[1])

    result = OptimizationResult(
        optimal_price=best_price,
        expected_demand=max(float(demand_predictor(np.array([best_price]))[0]), 0.0),
        expected_profit=best_profit,
        success=True,
    )
    LOGGER.info(
        "Optimization complete. optimal_price=%.2f expected_profit=%.2f",
        result.optimal_price,
        result.expected_profit,
    )
    return result


def analytic_optimal_price(elasticity: float, unit_cost: float, min_price: float, max_price: float) -> float:
    """Closed-form optimum for constant-elasticity demand.

    Profit (p - c) * A * p^(-e) has its interior maximum at the Lerner markup
    p* = c * e / (e - 1) when e > 1, and is concave around it, so the
    constrained optimum is p* clipped to the bounds. When e <= 1 demand is
    inelastic and profit rises with price everywhere, so the optimum is
    max_price.
    """
    if elasticity <= 1.0:
        return max_price
    return float(np.clip(unit_cost * elasticity / (elasticity - 1.0), min_price, max_price))
