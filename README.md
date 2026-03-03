# BAWS: Bootstrap-Based Adaptive Window Selection for Forecasting

This repository implements the Bootstrap-Based Adaptive Window Selection (BAWS) algorithm for time series forecasting under structural breaks and regime changes.

## Overview

BAWS is an adaptive forecasting method that automatically selects the optimal window size for rolling-window estimation based on bootstrap-based admissibility testing. The algorithm is designed to handle:

- Discrete structural breaks in mean and variance
- Continuous parameter drift
- Volatility regime shifts
- Time-varying distributions

## Repository Structure

```
.
├── src/
│   ├── __init__.py
│   ├── baws.py                  # Core BAWS algorithm
│   ├── baselines.py             # Baseline methods (Fixed, Full, SAWS)
│   ├── bootstrap.py             # Bootstrap threshold computation
│   ├── data_generation.py       # Synthetic data generation
│   ├── experiment1.py           # Experiment 1: Discrete breaks
│   ├── experiment2.py           # Experiment 2: Continuous drift
│   ├── experiment3.py           # Experiment 3: GARCH volatility
│   ├── loss_functions.py        # Loss functions and estimators
│   ├── metrics.py               # Performance metrics
│   └── window_grid.py           # Window grid construction
├── tests/
│   ├── __init__.py
│   └── test_baws.py             # Test suite
├── results/
│   └── RESULTS.md               # Experimental results
├── run_experiments.py           # Main runner script
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

## Experiments

### Experiment 1: Discrete Structural Breaks
- **Settings**: A1, A2, A3 (i.i.d. Normal with mean/variance breaks)
- **Methods**: BAWS, SAWS, Fixed-250, Fixed-500, Fixed-750, Full window
- **Forecasts**: Mean and VaR (α=0.95)
- **Metrics**: MAB, Variance, MSE, Cumulative Risk, Cumulative Loss

### Experiment 2: Continuous Mean Drift
- **Settings**: B1 (sinusoidal), B2 (random walk), B3 (geometric Brownian motion)
- **Methods**: Same as Experiment 1
- **Forecasts**: Mean and VaR (α=0.95)

### Experiment 3: GARCH Volatility Regime Shifts
- **Data**: GARCH(1,1) with Fernandez-Steel skewed-t innovations
- **Regime shift**: Persistence parameter changes from 0.7 to 0.95 at t=1000
- **Bootstrap**: Moving block bootstrap (for temporal dependence)
- **Forecasts**: VaR (α=0.95)

### Experiment 4: Empirical S&P 500 Analysis (Placeholder)
- **Data**: S&P 500 daily returns (2005-2025)
- **Forecasts**: VaR and Expected Shortfall
- **Sub-periods**: GFC, COVID-19, Tariff events

## Installation

```bash
# Clone repository
git clone <repository-url>
cd qca-simulation-scenario-1--baws-vs-baselines-under-dis

# Install dependencies (already available in container)
# pip install -r requirements.txt
```

## Usage

### Running All Experiments

```bash
python run_experiments.py
```

### Running Tests

```bash
pytest tests/ -v
```

### Running Individual Experiments

```python
from src.experiment1 import run_experiment1

# Run Experiment 1 with setting A1
results = run_experiment1(
    settings=['A1'],
    T=2000,
    t_start=501,
    n_replications=1000,
    base_seed=42
)
```

## Key Features

### BAWS Algorithm

1. **Dynamic Window Grid**: Piecewise candidate grid with increments of 5/10/20/50/100
2. **Bootstrap Thresholds**: i.i.d. or moving block bootstrap (B=500 replications)
3. **Admissibility Testing**: Systematic comparison against all smaller reference windows
4. **Maximum Selection**: Chooses the largest admissible window for bias-variance tradeoff

### Loss Functions

- **Mean Forecasting**: Squared error loss with closed-form minimizer (sample mean)
- **VaR Forecasting**: Quantile scoring function with empirical α-quantile minimizer

### Performance Metrics

- **MAB**: Mean Absolute Bias
- **Var**: Variance of estimates
- **MSE**: Mean Squared Error
- **CR**: Cumulative Risk (population excess risk)
- **CL**: Cumulative Loss (sample forecast loss)

## Implementation Notes

### Computational Efficiency

The full experiments with n=1000 replications are computationally intensive. For testing purposes:
- Reduce `n_replications` to 100-500
- Reduce bootstrap replications `B` to 100-200
- Limit settings (e.g., run only A1 instead of A1, A2, A3)

### Reproducibility

All random number generation uses deterministic seeding:
- Master seed for each experiment (42, 43, 44 for Exp 1-3)
- Per-replication seeds derived deterministically
- Bootstrap seeds based on time and reference window indices

### Methodology Adherence

This implementation strictly follows the paper's methodology:
- Window grid construction exactly as specified
- Bootstrap procedures (i.i.d. and block) as described
- Loss functions and minimizers per equations
- Metrics computation matching paper definitions

## Results

Results are saved to `results/RESULTS.md` after running experiments. Key findings:

- BAWS achieves lowest MSE, CR, and CL for VaR forecasting under discrete breaks
- BAWS window sizes contract at structural breaks and expand in stable periods
- BAWS outperforms fixed windows in settings with regime changes
- Moving block bootstrap enables BAWS to handle temporal dependence (GARCH)

## References

Implementation based on the BAWS methodology for adaptive window selection in time series forecasting with structural breaks.

## License

MIT License
