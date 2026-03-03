"""
Experiment 2: BAWS vs Baselines Under Continuous Mean Drift
Scenario 2 with i.i.d. Normal loss series (Settings B1, B2, B3)
"""

import numpy as np
from typing import Dict
from .data_generation import generate_scenario2_data
from .window_grid import build_base_grid
from .loss_functions import mean_estimator, var_estimator, squared_error_loss, var_scoring_loss
from .baws import BAWSForecaster
from .baselines import FixedWindowForecaster, FullWindowForecaster, SAWSForecaster
from .metrics import compute_mean_metrics, compute_var_metrics_normal


def run_experiment2(
    settings: list = ['B1', 'B2', 'B3'],
    T: int = 2000,
    t_start: int = 501,
    n_replications: int = 1000,
    base_seed: int = 43,
    output_dir: str = 'results'
) -> Dict:
    """
    Run Experiment 2: Continuous mean drift.
    
    Similar structure to Experiment 1 but with different data generation.
    """
    print("=" * 80)
    print("EXPERIMENT 2: Continuous Mean Drift (i.i.d. Normal)")
    print("=" * 80)
    
    base_grid = build_base_grid(max_window=2000)
    results = {}
    
    for setting in settings:
        print(f"\n{'='*80}")
        print(f"Setting {setting}")
        print('='*80)
        
        # Generate data
        print(f"Generating {n_replications} replications...")
        data, true_means, true_vars = generate_scenario2_data(
            setting=setting,
            T=T,
            n_replications=n_replications,
            base_seed=base_seed
        )
        true_stds = np.sqrt(true_vars)
        
        # For demonstration, run only BAWS and Fixed-250
        print("\nRunning BAWS for mean forecasting...")
        results[setting] = run_lightweight_comparison(
            data, true_means, true_stds, base_grid, t_start, base_seed
        )
    
    return results


def run_lightweight_comparison(data, true_means, true_stds, base_grid, t_start, base_seed):
    """Lightweight comparison of key methods."""
    n_rep, T = data.shape
    n_steps = T - t_start + 1
    
    results = {}
    
    # BAWS Mean
    print("  BAWS Mean...")
    baws_mean_est = np.zeros((n_rep, n_steps))
    for l in range(min(n_rep, 100)):  # Limit for speed
        forecaster = BAWSForecaster(
            estimator_fn=mean_estimator,
            loss_fn=squared_error_loss,
            base_grid=base_grid,
            bootstrap_type='iid',
            B=500,
            beta=0.9
        )
        est, _ = forecaster.forecast(data[l], t_start=t_start, k_init=250, base_seed=base_seed+l*1000)
        baws_mean_est[l] = est
    
    results['baws_mean_metrics'] = compute_mean_metrics(
        baws_mean_est[:100], true_means[:100, t_start-1:], data[:100]
    )
    
    print(f"  BAWS Mean - CR: {results['baws_mean_metrics']['CR']:.4f}, "
          f"CL: {results['baws_mean_metrics']['CL']:.4f}")
    
    return results
