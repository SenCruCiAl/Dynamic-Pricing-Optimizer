"""Sensitivity analysis and visualization utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from optimizer import analytic_optimal_price  # noqa: E402
from simulator import DemandSimulator  # noqa: E402

LOGGER = logging.getLogger(__name__)


def plot_demand_curve(data: pd.DataFrame, prices: np.ndarray, predicted: np.ndarray, true: np.ndarray,
                      output_path: Path) -> None:
    """Plot observations, fitted demand and the true demand curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(data["price"], data["demand"], s=8, color="0.6", label="Observed")
    ax.plot(prices, true, color="0.2", linestyle="--", linewidth=1.5, label="True demand")
    ax.plot(prices, predicted, color="tab:blue", linewidth=2, label="Fitted demand")
    ax.set_xlabel("Price")
    ax.set_ylabel("Demand")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved demand curve to %s", output_path)


def plot_profit_curve(prices: np.ndarray, predicted: np.ndarray, true: np.ndarray, chosen: float,
                      analytic: float, output_path: Path) -> None:
    """Plot predicted vs true profit with chosen and analytic optima."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(prices, true, color="0.2", linestyle="--", linewidth=1.5, label="True profit")
    ax.plot(prices, predicted, color="tab:green", linewidth=2, label="Predicted profit")
    ax.axvline(analytic, color="0.2", linewidth=1, label=f"Analytic optimum {analytic:.2f}")
    ax.axvline(chosen, color="tab:red", linewidth=1, linestyle=":", label=f"Chosen price {chosen:.2f}")
    ax.set_xlabel("Price")
    ax.set_ylabel("Profit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved profit curve to %s", output_path)


def run_sensitivity_analysis(
    min_price: float,
    max_price: float,
    unit_cost: float,
    elasticity_values: list[float],
    output_path: Path,
) -> None:
    """Plot true profit curves across elasticity values, marking each optimum."""
    simulator = DemandSimulator(noise_sigma=0.0)
    prices = np.linspace(min_price, max_price, 300)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for elasticity in elasticity_values:
        profit = (prices - unit_cost) * simulator.true_demand(prices, elasticity)
        line, = ax.plot(prices, profit, linewidth=2, label=f"e = {elasticity:.1f}")
        p_star = analytic_optimal_price(elasticity, unit_cost, min_price, max_price)
        p_profit = (p_star - unit_cost) * simulator.true_demand(np.array([p_star]), elasticity)[0]
        ax.plot([p_star], [p_profit], "o", color=line.get_color())

    ax.set_xlabel("Price")
    ax.set_ylabel("Profit")
    ax.set_title(f"Profit vs price by elasticity (unit cost {unit_cost:g}); dots mark optima")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    LOGGER.info("Saved elasticity sensitivity plot to %s", output_path)
