# Dynamic Pricing Optimization Engine

A production-ready Python project that simulates demand behavior, learns a demand curve from data, estimates elasticity, and computes the revenue-maximizing price under configurable price bounds.

## Why this project matters

Dynamic pricing allows businesses to respond to demand patterns and maximize topline revenue instead of relying on static prices. This engine demonstrates an end-to-end workflow from synthetic data generation to optimization and business reporting.

---

## Revenue Maximization Explained

The objective function is:

\[
\text{Revenue}(p) = p \times \hat{D}(p)
\]

Where:
- \(p\) = candidate price
- \(\hat{D}(p)\) = model-predicted demand at price \(p\)

The optimizer searches within constraints \([\text{min\_price}, \text{max\_price}]\) and selects the price with maximum expected revenue.

---

## Price Elasticity Explained

Price elasticity of demand measures how sensitive demand is to price changes.

- Elasticity magnitude **> 1**: demand is highly price-sensitive.
- Elasticity magnitude **< 1**: demand is less price-sensitive.

This project estimates elasticity from log-log regression:

\[
\log(D) = \alpha + \beta\log(P)
\]

Where \(\beta\) is the estimated elasticity coefficient (usually negative in real markets).

---

## System Architecture

1. **Data Simulation (`src/simulator.py`)**
   - Generates synthetic demand data with controllable elasticity and noise.
   - Writes dataset to `data/demand_data.csv`.

2. **Demand Modeling (`src/demand_model.py`)**
   - Trains two models: Linear Regression and Random Forest Regressor.
   - Selects the better model using in-sample R².
   - Estimates elasticity from log-log regression.
   - Saves trained model artifacts to `models/demand_model.pkl`.

3. **Optimization (`src/optimizer.py`)**
   - Defines revenue as `price * predicted_demand`.
   - Uses `scipy.optimize.minimize_scalar` to maximize revenue in bounded price interval.

4. **Analysis & Visualization (`src/analysis.py`)**
   - Demand curve plot.
   - Revenue curve plot.
   - Elasticity sensitivity comparison plot.

5. **CLI Orchestration (`src/main.py`)**
   - Runs the full pipeline end-to-end.
   - Stores summary output in `results/optimal_price.json`.

---

## Repository Structure

```text
dynamic-pricing-engine/
│
├── data/
├── src/
│   ├── simulator.py
│   ├── demand_model.py
│   ├── optimizer.py
│   ├── analysis.py
│   ├── utils.py
│   └── main.py
│
├── results/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Engine

```bash
python src/main.py --min_price 10 --max_price 100
```

Optional parameters:
- `--elasticity` (default: `1.4`)
- `--samples` (default: `300`)

---

## Outputs

After execution, you will get:

- `data/demand_data.csv`
- `models/demand_model.pkl`
- `results/optimal_price.json`
- `results/demand_curve.png`
- `results/revenue_curve.png`
- `results/elasticity_sensitivity.png`

### Example `results/optimal_price.json`

```json
{
  "min_price": 10.0,
  "max_price": 100.0,
  "simulated_elasticity": 1.4,
  "estimated_elasticity": -1.35,
  "optimal_price": 34.72,
  "expected_demand": 760.11,
  "expected_revenue": 26391.15,
  "optimization_success": true
}
```

---

## Business Implications

- **Higher margin capture:** identifies price points where demand drop-off is offset by higher unit prices.
- **Demand-aware decisions:** elasticity estimation helps forecast customer sensitivity by category.
- **Scenario planning:** sensitivity analysis reveals how robust pricing strategy is under different market reactions.
- **Operational deployment:** saved model and JSON outputs can be integrated into BI dashboards or automated pricing workflows.

---

## Code Quality

- PEP8-compliant modules
- Docstrings for public classes/functions
- Structured logging for observability
- Deterministic simulation via random seed
