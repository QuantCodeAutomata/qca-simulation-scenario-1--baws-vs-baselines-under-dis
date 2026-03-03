"""
Experiment 3: BAWS vs Baselines Under GARCH Volatility Regime Shifts
Scenario 3 with GARCH(1,1) and skewed-t innovations
"""

import numpy as np
from typing import Dict
from .data_generation import generate_scenario3_data
from .window_grid import build_base_grid
from .loss_functions import var_estimator, var_scoring_loss
from .baws import BAWSForecaster
from .baselines import FixedWindowForecaster, FullWindowForecaster
from .metrics import compute_var_metrics_garch


def run_experiment3(
    T: int = 2000,
    t_start: int = 501,
    n_replications: int = 1000,
    base_seed: int = 44,
    output_dir: str = 'results'
) -> Dict:
    """
    Run Experiment 3: GARCH volatility regime shifts.
    """
    print("=" * 80)
    print("EXPERIMENT 3: GARCH Volatility Regime Shifts")
    print("=" * 80)
    
    # Generate data
    print(f"\nGenerating {n_replications} GARCH replications...")
    losses, sigmas, true_vars, q_epsilon_005 = generate_scenario3_data(
        T=T,
        n_replications=n_replications,
        base_seed=base_seed
    )
    
    base_grid = build_base_grid(max_window=2000)
    n_steps = T - t_start + 1
    alpha = 0.95
    
    def var_est(x):
        return var_estimator(x, alpha)
    
    def var_loss(x, v):
        return var_scoring_loss(x, v, alpha)
    
    results = {}
    
    # Run BAWS with block bootstrap (limit replications for speed)
    print("\nRunning BAWS with moving block bootstrap...")
    n_run = min(n_replications, 100)
    baws_estimates = np.zeros((n_run, n_steps))
    baws_windows = np.zeros((n_run, n_steps), dtype=int)
    
    for l in range(n_run):
        if (l + 1) % 10 == 0:
            print(f"  Replication {l+1}/{n_run}")
        
        forecaster = BAWSForecaster(
            estimator_fn=var_est,
            loss_fn=var_loss,
            base_grid=base_grid,
            bootstrap_type='block',
            B=500,
            beta=0.9
        )
        
        est, win = forecaster.forecast(
            losses[l], t_start=t_start, k_init=250,
            base_seed=base_seed + l * 1000
        )
        baws_estimates[l] = est
        baws_windows[l] = win
    
    results['BAWS'] = {
        'estimates': baws_estimates,
        'windows': baws_windows,
        'metrics': compute_var_metrics_garch(
            baws_estimates, true_vars[:n_run, t_start-1:],
            sigmas[:n_run], losses[:n_run], q_epsilon_005, alpha
        )
    }
    
    # Fixed-250
    print("\nRunning Fixed-250...")
    fixed_estimates = np.zeros((n_run, n_steps))
    for l in range(n_run):
        forecaster = FixedWindowForecaster(var_est, 250)
        est, _ = forecaster.forecast(losses[l], t_start=t_start)
        fixed_estimates[l] = est
    
    results['Fixed-250'] = {
        'estimates': fixed_estimates,
        'metrics': compute_var_metrics_garch(
            fixed_estimates, true_vars[:n_run, t_start-1:],
            sigmas[:n_run], losses[:n_run], q_epsilon_005, alpha
        )
    }
    
    # Print results
    print("\n  VaR Forecasting Metrics (GARCH):")
    print(f"  {'Method':<12} {'MSE':>10} {'Var':>10} {'CR':>12} {'CL':>12}")
    print("  " + "-" * 56)
    for name, result in results.items():
        m = result['metrics']
        print(f"  {name:<12} {m['MSE']:10.6f} {m['Var']:10.6f} "
              f"{m['CR']:12.4f} {m['CL']:12.4f}")
    
    return results
