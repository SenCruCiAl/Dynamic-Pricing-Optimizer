import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from demand_model import DemandModeler  # noqa: E402
from optimizer import analytic_optimal_price, optimize_price  # noqa: E402
from simulator import DemandSimulator  # noqa: E402


@pytest.mark.parametrize("elasticity,cost", [(1.4, 20.0), (2.0, 15.0), (3.0, 30.0)])
def test_optimizer_matches_lerner_rule_on_true_demand(elasticity, cost):
    sim = DemandSimulator()
    result = optimize_price(lambda p: sim.true_demand(p, elasticity), 10, 200, cost)
    expected = analytic_optimal_price(elasticity, cost, 10, 200)
    assert result.optimal_price == pytest.approx(expected, rel=1e-3)


def test_revenue_objective_with_elastic_demand_is_a_corner_solution():
    # With zero cost and e > 1, revenue ~ p^(1-e) falls monotonically, so the floor wins.
    sim = DemandSimulator()
    result = optimize_price(lambda p: sim.true_demand(p, 1.4), 10, 100, unit_cost=0.0)
    assert result.optimal_price == pytest.approx(10.0)


def test_inelastic_demand_pushes_price_to_ceiling():
    assert analytic_optimal_price(0.8, 20, 10, 100) == 100


def test_cv_selection_prefers_correctly_specified_model():
    data = DemandSimulator(random_seed=0).generate(10, 100, n_samples=300, elasticity=1.6)
    artifacts = DemandModeler().fit(data)
    assert artifacts.selected_model_name == "log_log"


def test_elasticity_estimate_recovers_truth():
    data = DemandSimulator(random_seed=1).generate(10, 100, n_samples=500, elasticity=1.8)
    assert DemandModeler.estimate_elasticity(data) == pytest.approx(-1.8, abs=0.05)


def test_end_to_end_regret_is_small():
    sim = DemandSimulator(random_seed=3)
    data = sim.generate(10, 100, n_samples=300, elasticity=1.5)
    modeler = DemandModeler()
    modeler.fit(data)
    chosen = optimize_price(modeler.predict, 10, 100, unit_cost=20).optimal_price
    truth = analytic_optimal_price(1.5, 20, 10, 100)

    def profit(p):
        return (p - 20) * sim.true_demand(np.array([p]), 1.5)[0]

    assert 1 - profit(chosen) / profit(truth) < 0.01
