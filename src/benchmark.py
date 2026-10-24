"""Monte Carlo benchmark: profit regret of each pricing strategy against the analytic optimum."""

from __future__ import annotations

import json
import logging

import numpy as np
from scipy.optimize import minimize_scalar

from demand_model import DemandModeler, ForestDemand
from optimizer import analytic_optimal_price, optimize_price
from simulator import DemandSimulator
from utils import RESULTS_DIR, ensure_directories

MIN_P, MAX_P, COST, N = 10.0, 100.0, 20.0, 300
ELASTICITIES = [1.3, 1.6, 2.0, 3.0]
NOISES = [0.15, 0.30]
SEEDS = range(30)


def bounded_only(predict, lo, hi, cost):
    """The original optimizer: Brent's method alone, which assumes a unimodal objective."""
    res = minimize_scalar(lambda p: -(p - cost) * max(float(predict(np.array([p]))[0]), 0.0),
                          bounds=(lo, hi), method="bounded")
    return float(res.x)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ensure_directories()
    rows = []
    for e in ELASTICITIES:
        for noise in NOISES:
            regrets = {"cv_selected_grid": [], "forest_grid": [], "forest_brent_only": []}
            for seed in SEEDS:
                sim = DemandSimulator(noise_sigma=noise, random_seed=seed)
                data = sim.generate(MIN_P, MAX_P, n_samples=N, elasticity=e)
                truth = analytic_optimal_price(e, COST, MIN_P, MAX_P)

                def profit(p):
                    return (p - COST) * sim.true_demand(np.array([p]), e)[0]

                modeler = DemandModeler(random_state=seed)
                modeler.fit(data)
                forest = ForestDemand(random_state=seed).fit(data["price"].to_numpy(), data["demand"].to_numpy())
                chosen = {
                    "cv_selected_grid": optimize_price(modeler.predict, MIN_P, MAX_P, COST).optimal_price,
                    "forest_grid": optimize_price(forest.predict, MIN_P, MAX_P, COST).optimal_price,
                    "forest_brent_only": bounded_only(forest.predict, MIN_P, MAX_P, COST),
                }
                for k, p in chosen.items():
                    regrets[k].append(1 - profit(p) / profit(truth))
            rows.append({
                "elasticity": e,
                "noise_sigma": noise,
                **{f"{k}_mean_regret": float(np.mean(v)) for k, v in regrets.items()},
                **{f"{k}_p90_regret": float(np.quantile(v, 0.9)) for k, v in regrets.items()},
            })
            print(f"e={e} noise={noise}: " + "  ".join(f"{k}={100*np.mean(v):.2f}%" for k, v in regrets.items()))

    with (RESULTS_DIR / "benchmark.json").open("w", encoding="utf-8") as f:
        json.dump({"settings": {"min_price": MIN_P, "max_price": MAX_P, "unit_cost": COST, "samples": N,
                                "seeds": len(SEEDS)}, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
