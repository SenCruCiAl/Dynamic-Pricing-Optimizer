"""Sensitivity analysis and visualization utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from simulator import DemandSimulator

LOGGER = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def plot_demand_curve(prices: np.ndarray, demand: np.ndarray, output_path: Path) -> None:
    """Plot modeled demand curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(prices, demand, color="tab:blue", linewidth=2)
    ax.set_title("Demand Curve")
    ax.set_xlabel("Price")
    ax.set_ylabel("Predicted Demand")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved demand curve to %s", output_path)


def plot_revenue_curve(prices: np.ndarray, revenue: np.ndarray, output_path: Path) -> None:
    """Plot modeled revenue curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(prices, revenue, color="tab:green", linewidth=2)
    ax.set_title("Revenue Curve")
    ax.set_xlabel("Price")
    ax.set_ylabel("Expected Revenue")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved revenue curve to %s", output_path)


def run_sensitivity_analysis(
    min_price: float,
    max_price: float,
    elasticity_values: list[float],
    output_path: Path,
) -> None:
    """Generate and plot revenue curves across multiple elasticity assumptions."""
    simulator = DemandSimulator(noise_std=0.0)
    prices = np.linspace(min_price, max_price, 200)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for elasticity in elasticity_values:
        synthetic = simulator.generate(
            min_price=min_price,
            max_price=max_price,
            n_samples=len(prices),
            elasticity=elasticity,
        )
        demand = synthetic["true_deterministic_demand"].to_numpy()
        revenue = synthetic["price"].to_numpy() * demand
        sort_idx = np.argsort(synthetic["price"].to_numpy())
        ax.plot(
            synthetic["price"].to_numpy()[sort_idx],
            revenue[sort_idx],
            linewidth=2,
            label=f"Elasticity={elasticity:.2f}",
        )

    ax.set_title("Revenue Sensitivity to Elasticity")
    ax.set_xlabel("Price")
    ax.set_ylabel("Revenue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    LOGGER.info("Saved elasticity sensitivity plot to %s", output_path)
