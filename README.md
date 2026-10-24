# Dynamic Pricing Optimizer

[![CI](https://github.com/SenCruCiAl/Dynamic-Pricing-Optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/SenCruCiAl/Dynamic-Pricing-Optimizer/actions/workflows/ci.yml)

**Interactive page:** https://sencrucial.github.io/Dynamic-Pricing-Optimizer/

Simulates sales from a demand curve with a known answer, estimates the curve
from noisy data, picks a profit-maximising price, and scores that price
against the closed-form optimum. Because the data is simulated, every number
here can be checked against ground truth.

## The problem being solved

Demand has constant elasticity `e` with multiplicative noise:

```
D(p) = A · (p / p0)^(-e) · exp(eps),   eps ~ N(0, sigma^2)
```

Profit is `(p - c) · D(p)` for unit cost `c`. Setting its derivative to zero
gives the Lerner markup (Lerner, 1934):

```
p* = c · e / (e - 1)     for e > 1, clipped to [min_price, max_price]
```

Two corner cases follow directly and are tested:

- **Zero cost (pure revenue maximisation) with elastic demand:** revenue
  `∝ p^(1-e)` falls at every price, so the optimum is always `min_price`.
- **Inelastic demand (`e ≤ 1`):** profit rises with price everywhere, so the
  optimum is always `max_price`.

## Pipeline

1. `simulator.py`: draw prices uniformly in the allowed range and demand from
   the curve above.
2. `demand_model.py`: fit a log-log regression (constant elasticity) and a
   random forest; **select by 5-fold cross-validated RMSE**.
3. `optimizer.py`: dense grid search over the price range, then Brent's
   method inside the best bracket; `analytic_optimal_price` gives the truth.
4. `main.py`: run it all and report the chosen price, the analytic optimum and
   the profit regret.
5. `benchmark.py`: Monte Carlo comparison of strategies (below).

## What was wrong in the first version, and the evidence

| Issue | Effect | Fix |
|---|---|---|
| Objective was revenue with zero cost | Under constant elasticity the true optimum is always a price bound, so the "optimal price" was meaningless | Profit objective with a unit cost (`--unit_cost`, default 20) |
| Model chosen by in-sample R² | Random forest always won (R² 0.998 vs 0.650) by memorising the data | 5-fold CV RMSE (log-log 172 vs forest 210 on the default run) |
| Brent's method alone on a step-shaped forest curve | Returned 13.93 when the true optimum for that setup was the floor, 10.00 | Grid search first, Brent refinement inside the best bracket |

Mean profit regret against the analytic optimum, 30 seeds per row, 300
observations, prices 10–100, unit cost 20 (`python src/benchmark.py`):

| Elasticity | σ | CV-selected (this version) | Forest + grid | Forest + Brent only (original) |
|---|---|---|---|---|
| 1.3 | 0.15 | **0.02%** | 0.82% | 0.92% |
| 1.3 | 0.30 | **0.09%** | 0.98% | 3.31% |
| 1.6 | 0.15 | **0.13%** | 1.02% | 1.64% |
| 1.6 | 0.30 | **0.15%** | 2.62% | 2.12% |
| 2.0 | 0.15 | **0.03%** | 1.38% | 2.91% |
| 2.0 | 0.30 | **0.05%** | 3.21% | 6.16% |
| 3.0 | 0.15 | **0.14%** | 2.09% | 2.40% |
| 3.0 | 0.30 | **0.16%** | 3.37% | 3.66% |

**Caveat.** The simulated demand is log-linear, so the log-log model is
correctly specified here by construction, and its win is expected. The
pipeline still selects by cross-validation rather than hard-coding log-log,
because on real data with a different demand shape the forest could win.
Nothing in this repository is fitted to real sales data.

## Run it

```bash
pip install -r requirements.txt
python src/main.py --min_price 10 --max_price 100            # unit cost 20, e = 1.4
python src/main.py --min_price 10 --max_price 100 --unit_cost 0   # revenue: lands on the floor
pytest -q tests
```

Options: `--unit_cost`, `--elasticity`, `--noise`, `--samples`, `--seed`.

Output of the default run (`results/optimal_price.json`, trimmed):

```json
{
  "selected_model": "log_log",
  "cv_rmse": {"log_log": 172.42, "random_forest": 209.67},
  "estimated_elasticity": -1.407,
  "optimal_price": 69.11,
  "analytic_optimal_price": 70.0,
  "profit_regret_vs_true_optimum": 0.00005
}
```

The profit curve is flat near its peak, so a price 0.9 away from the optimum
loses only 0.005% of profit.

Figures go to `results/`: `demand_curve.png` (observations, fitted and true
demand), `profit_curve.png` (predicted vs true profit with both optima) and
`elasticity_sensitivity.png`.

## Project page

`docs/index.html` reimplements the simulator, log-log fit and optimiser in
plain JavaScript so the page runs with no backend. It is deployed by
`.github/workflows/pages.yml`.

## Reference

Lerner, A. P. (1934). The concept of monopoly and the measurement of monopoly
power. *Review of Economic Studies*, 1(3), 157–175.
