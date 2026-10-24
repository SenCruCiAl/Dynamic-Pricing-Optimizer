"""CLI entrypoint for the Dynamic Pricing Optimization Engine."""

from __future__ import annotations

import argparse
import logging

import numpy as np

from analysis import plot_demand_curve, plot_profit_curve, run_sensitivity_analysis
from demand_model import DemandModeler
from optimizer import analytic_optimal_price, optimize_price
from simulator import DemandSimulator
from utils import DATA_DIR, MODELS_DIR, RESULTS_DIR, ensure_directories, save_json, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Dynamic Pricing Optimization Engine")
    parser.add_argument("--min_price", type=float, required=True, help="Minimum allowed price")
    parser.add_argument("--max_price", type=float, required=True, help="Maximum allowed price")
    parser.add_argument("--unit_cost", type=float, default=20.0, help="Marginal cost per unit (0 = maximize revenue)")
    parser.add_argument(
        "--elasticity",
        type=float,
        default=1.4,
        help="Elasticity used to generate synthetic demand data",
    )
    parser.add_argument("--noise", type=float, default=0.15, help="Std. dev. of log-demand noise")
    parser.add_argument(
        "--samples",
        type=int,
        default=300,
        help="Number of synthetic observations to generate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Run full optimization workflow."""
    setup_logging()
    args = parse_args()

    if args.min_price <= 0 or args.max_price <= 0:
        raise ValueError("Price bounds must be positive values.")
    if args.min_price >= args.max_price:
        raise ValueError("min_price must be strictly lower than max_price.")
    if args.unit_cost < 0:
        raise ValueError("unit_cost cannot be negative.")

    ensure_directories()

    simulator = DemandSimulator(noise_sigma=args.noise, random_seed=args.seed)
    dataset = simulator.generate(
        min_price=args.min_price,
        max_price=args.max_price,
        n_samples=args.samples,
        elasticity=args.elasticity,
    )
    simulator.save(dataset, DATA_DIR / "demand_data.csv")

    modeler = DemandModeler(random_state=args.seed)
    artifacts = modeler.fit(dataset)
    estimated_elasticity = modeler.estimate_elasticity(dataset)
    modeler.save(str(MODELS_DIR / "demand_model.pkl"))

    result = optimize_price(modeler.predict, args.min_price, args.max_price, args.unit_cost)

    # Ground truth is available because the data is simulated.
    true_price = analytic_optimal_price(args.elasticity, args.unit_cost, args.min_price, args.max_price)

    def true_profit(p: float) -> float:
        return float((p - args.unit_cost) * simulator.true_demand(np.array([p]), args.elasticity)[0])

    regret = 1.0 - true_profit(result.optimal_price) / true_profit(true_price) if true_profit(true_price) > 0 else 0.0

    price_grid = np.linspace(args.min_price, args.max_price, 200)
    predicted_demand = np.maximum(modeler.predict(price_grid), 0.0)
    plot_demand_curve(dataset, price_grid, predicted_demand, simulator.true_demand(price_grid, args.elasticity),
                      RESULTS_DIR / "demand_curve.png")
    plot_profit_curve(price_grid, (price_grid - args.unit_cost) * predicted_demand,
                      (price_grid - args.unit_cost) * simulator.true_demand(price_grid, args.elasticity),
                      result.optimal_price, true_price, RESULTS_DIR / "profit_curve.png")
    run_sensitivity_analysis(
        min_price=args.min_price,
        max_price=args.max_price,
        unit_cost=args.unit_cost,
        elasticity_values=[1.2, 1.4, 1.8, 2.5, 4.0],
        output_path=RESULTS_DIR / "elasticity_sensitivity.png",
    )

    payload = {
        "min_price": args.min_price,
        "max_price": args.max_price,
        "unit_cost": args.unit_cost,
        "simulated_elasticity": args.elasticity,
        "estimated_elasticity": estimated_elasticity,
        "selected_model": artifacts.selected_model_name,
        "cv_rmse": artifacts.cv_rmse,
        "optimal_price": result.optimal_price,
        "analytic_optimal_price": true_price,
        "expected_demand": result.expected_demand,
        "expected_profit": result.expected_profit,
        "profit_regret_vs_true_optimum": regret,
    }
    output_json = RESULTS_DIR / "optimal_price.json"
    save_json(payload, output_json)
    LOGGER.info("Chosen price %.2f vs analytic %.2f (profit regret %.2f%%)", result.optimal_price, true_price,
                100 * regret)
    LOGGER.info("Workflow complete. Results written to %s", output_json)


if __name__ == "__main__":
    main()
