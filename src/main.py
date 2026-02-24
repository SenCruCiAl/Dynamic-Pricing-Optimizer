"""CLI entrypoint for the Dynamic Pricing Optimization Engine."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from analysis import plot_demand_curve, plot_revenue_curve, run_sensitivity_analysis
from demand_model import DemandModeler
from optimizer import optimize_price
from simulator import DemandSimulator
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, ensure_directories, save_json, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Dynamic Pricing Optimization Engine")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum allowed price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum allowed price")
    parser.add_argument(
        "--elasticity",
        type=float,
        default=1.4,
        help="Elasticity used to generate synthetic demand data",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of synthetic observations to generate",
    )
    return parser.parse_args()


def main() -> None:
    """Run full optimization workflow."""
    setup_logging()
    args = parse_args()

    if args.min_price <= 0 or args.max_price <= 0:
        raise ValueError("Price bounds must be positive values.")
    if args.min_price >= args.max_price:
        raise ValueError("min_price must be strictly lower than max_price.")

    ensure_directories()

    simulator = DemandSimulator()
    dataset = simulator.generate(
        min_price=args.min_price,
        max_price=args.max_price,
        n_samples=args.samples,
        elasticity=args.elasticity,
    )
    data_path = DATA_DIR / "demand_data.csv"
    simulator.save(dataset, data_path)

    modeler = DemandModeler()
    modeler.fit(dataset)
    estimated_elasticity = modeler.estimate_elasticity(dataset)

    model_path = MODELS_DIR / "demand_model.pkl"
    modeler.save(str(model_path))

    optimization_result = optimize_price(
        demand_predictor=modeler.predict,
        min_price=args.min_price,
        max_price=args.max_price,
    )

    price_grid = np.linspace(args.min_price, args.max_price, 200)
    predicted_demand = np.maximum(modeler.predict(price_grid), 0.0)
    expected_revenue = price_grid * predicted_demand

    plot_demand_curve(price_grid, predicted_demand, RESULTS_DIR / "demand_curve.png")
    plot_revenue_curve(price_grid, expected_revenue, RESULTS_DIR / "revenue_curve.png")
    run_sensitivity_analysis(
        min_price=args.min_price,
        max_price=args.max_price,
        elasticity_values=[0.8, 1.0, 1.2, 1.4, 1.8],
        output_path=RESULTS_DIR / "elasticity_sensitivity.png",
    )

    result_payload = {
        "min_price": args.min_price,
        "max_price": args.max_price,
        "simulated_elasticity": args.elasticity,
        "estimated_elasticity": estimated_elasticity,
        "optimal_price": optimization_result.optimal_price,
        "expected_demand": optimization_result.expected_demand,
        "expected_revenue": optimization_result.expected_revenue,
        "optimization_success": optimization_result.success,
    }
    output_json = RESULTS_DIR / "optimal_price.json"
    save_json(result_payload, output_json)

    LOGGER.info("Workflow complete. Results written to %s", output_json)


if __name__ == "__main__":
    main()
