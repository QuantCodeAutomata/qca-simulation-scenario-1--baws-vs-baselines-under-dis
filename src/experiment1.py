"""
Experiment 1: BAWS vs Baselines Under Discrete Structural Breaks
Scenario 1 with i.i.d. Normal loss series (Settings A1, A2, A3)
"""

import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from .data_generation import generate_scenario1_data
from .window_grid import build_base_grid
from .loss_functions import mean_estimator, var_estimator, squared_error_loss, var_scoring_loss
from .baws import BAWSForecaster
from .baselines import FixedWindowForecaster, FullWindowForecaster, SAWSForecaster
from .metrics import compute_mean_metrics, compute_var_metrics_normal


def run_experiment1(
    settings: list = ['A1', 'A2', 'A3'],
    T: int = 2000,
    t_start: int = 501,
    n_replications: int = 1000,
    base_seed: int = 42,
    output_dir: str = 'results'
) -> Dict:
    """
    Run Experiment 1: Discrete structural breaks.
    
    Parameters
    ----------
    settings : list
        List of settings to run ('A1', 'A2', 'A3')
    T : int
        Time series length
    t_start : int
        Starting time for forecasting
    n_replications : int
        Number of Monte Carlo replications
    base_seed : int
        Base random seed
    output_dir : str
        Directory to save results
        
    Returns
    -------
    Dict
        Results dictionary with metrics and figures
    """
    print("=" * 80)
    print("EXPERIMENT 1: Discrete Structural Breaks (i.i.d. Normal)")
    print("=" * 80)
    
    # Build base grid
    base_grid = build_base_grid(max_window=2000)
    
    results = {}
    
    for setting in settings:
        print(f"\n{'='*80}")
        print(f"Setting {setting}")
        print('='*80)
        
        # Generate data
        print(f"Generating {n_replications} replications...")
        data, true_means, true_vars = generate_scenario1_data(
            setting=setting,
            T=T,
            n_replications=n_replications,
            base_seed=base_seed
        )
        true_stds = np.sqrt(true_vars)
        
        setting_results = {}
        
        # Run mean forecasting
        print("\nMean Forecasting:")
        mean_results = run_mean_forecasting(
            data, true_means, base_grid, t_start, base_seed, setting
        )
        setting_results['mean'] = mean_results
        
        # Run VaR forecasting
        print("\nVaR Forecasting:")
        var_results = run_var_forecasting(
            data, true_means, true_stds, base_grid, t_start, base_seed, setting
        )
        setting_results['var'] = var_results
        
        results[setting] = setting_results
    
    return results


def run_mean_forecasting(
    data: np.ndarray,
    true_means: np.ndarray,
    base_grid: list,
    t_start: int,
    base_seed: int,
    setting: str
) -> Dict:
    """Run mean forecasting for all methods."""
    n_rep, T = data.shape
    n_steps = T - t_start + 1
    
    methods = {}
    
    # BAWS
    print("  Running BAWS...")
    baws_estimates = np.zeros((n_rep, n_steps))
    baws_windows = np.zeros((n_rep, n_steps), dtype=int)
    
    for l in range(n_rep):
        if (l + 1) % 100 == 0:
            print(f"    Replication {l+1}/{n_rep}")
        
        forecaster = BAWSForecaster(
            estimator_fn=mean_estimator,
            loss_fn=squared_error_loss,
            base_grid=base_grid,
            bootstrap_type='iid',
            B=500,
            beta=0.9
        )
        
        est, win = forecaster.forecast(
            data[l], t_start=t_start, k_init=250,
            base_seed=base_seed + l * 1000
        )
        baws_estimates[l] = est
        baws_windows[l] = win
    
    methods['BAWS'] = {
        'estimates': baws_estimates,
        'windows': baws_windows,
        'metrics': compute_mean_metrics(
            baws_estimates, true_means[:, t_start-1:], data
        )
    }
    
    # Fixed windows
    for k in [250, 500, 750]:
        print(f"  Running Fixed-{k}...")
        fixed_estimates = np.zeros((n_rep, n_steps))
        
        for l in range(n_rep):
            forecaster = FixedWindowForecaster(mean_estimator, k)
            est, _ = forecaster.forecast(data[l], t_start=t_start)
            fixed_estimates[l] = est
        
        methods[f'Fixed-{k}'] = {
            'estimates': fixed_estimates,
            'metrics': compute_mean_metrics(
                fixed_estimates, true_means[:, t_start-1:], data
            )
        }
    
    # Full window
    print("  Running Full...")
    full_estimates = np.zeros((n_rep, n_steps))
    
    for l in range(n_rep):
        forecaster = FullWindowForecaster(mean_estimator)
        est, _ = forecaster.forecast(data[l], t_start=t_start)
        full_estimates[l] = est
    
    methods['Full'] = {
        'estimates': full_estimates,
        'metrics': compute_mean_metrics(
            full_estimates, true_means[:, t_start-1:], data
        )
    }
    
    # SAWS
    print("  Running SAWS...")
    saws_estimates = np.zeros((n_rep, n_steps))
    saws_windows = np.zeros((n_rep, n_steps), dtype=int)
    
    for l in range(n_rep):
        forecaster = SAWSForecaster(
            estimator_fn=mean_estimator,
            loss_fn=squared_error_loss,
            base_grid=base_grid,
            alpha_tau=0.1,
            C_tau=0.3
        )
        est, win = forecaster.forecast(data[l], t_start=t_start, k_init=250)
        saws_estimates[l] = est
        saws_windows[l] = win
    
    methods['SAWS'] = {
        'estimates': saws_estimates,
        'windows': saws_windows,
        'metrics': compute_mean_metrics(
            saws_estimates, true_means[:, t_start-1:], data
        )
    }
    
    # Print metrics
    print("\n  Mean Forecasting Metrics:")
    print(f"  {'Method':<12} {'MAB':>10} {'Var':>10} {'MSE':>10} {'CR':>12} {'CL':>12}")
    print("  " + "-" * 68)
    for name, result in methods.items():
        m = result['metrics']
        print(f"  {name:<12} {m['MAB']:10.4f} {m['Var']:10.4f} {m['MSE']:10.4f} "
              f"{m['CR']:12.4f} {m['CL']:12.4f}")
    
    return methods


def run_var_forecasting(
    data: np.ndarray,
    true_means: np.ndarray,
    true_stds: np.ndarray,
    base_grid: list,
    t_start: int,
    base_seed: int,
    setting: str,
    alpha: float = 0.95
) -> Dict:
    """Run VaR forecasting for all methods."""
    n_rep, T = data.shape
    n_steps = T - t_start + 1
    
    methods = {}
    
    # Create VaR-specific estimator and loss
    def var_est(x):
        return var_estimator(x, alpha)
    
    def var_loss(x, v):
        return var_scoring_loss(x, v, alpha)
    
    # BAWS
    print("  Running BAWS...")
    baws_estimates = np.zeros((n_rep, n_steps))
    baws_windows = np.zeros((n_rep, n_steps), dtype=int)
    
    for l in range(n_rep):
        if (l + 1) % 100 == 0:
            print(f"    Replication {l+1}/{n_rep}")
        
        forecaster = BAWSForecaster(
            estimator_fn=var_est,
            loss_fn=var_loss,
            base_grid=base_grid,
            bootstrap_type='iid',
            B=500,
            beta=0.9
        )
        
        est, win = forecaster.forecast(
            data[l], t_start=t_start, k_init=250,
            base_seed=base_seed + l * 1000
        )
        baws_estimates[l] = est
        baws_windows[l] = win
    
    methods['BAWS'] = {
        'estimates': baws_estimates,
        'windows': baws_windows,
        'metrics': compute_var_metrics_normal(
            baws_estimates, true_means[:, t_start-1:],
            true_stds[:, t_start-1:], data, alpha
        )
    }
    
    # Fixed windows
    for k in [250, 500, 750]:
        print(f"  Running Fixed-{k}...")
        fixed_estimates = np.zeros((n_rep, n_steps))
        
        for l in range(n_rep):
            forecaster = FixedWindowForecaster(var_est, k)
            est, _ = forecaster.forecast(data[l], t_start=t_start)
            fixed_estimates[l] = est
        
        methods[f'Fixed-{k}'] = {
            'estimates': fixed_estimates,
            'metrics': compute_var_metrics_normal(
                fixed_estimates, true_means[:, t_start-1:],
                true_stds[:, t_start-1:], data, alpha
            )
        }
    
    # Full window
    print("  Running Full...")
    full_estimates = np.zeros((n_rep, n_steps))
    
    for l in range(n_rep):
        forecaster = FullWindowForecaster(var_est)
        est, _ = forecaster.forecast(data[l], t_start=t_start)
        full_estimates[l] = est
    
    methods['Full'] = {
        'estimates': full_estimates,
        'metrics': compute_var_metrics_normal(
            full_estimates, true_means[:, t_start-1:],
            true_stds[:, t_start-1:], data, alpha
        )
    }
    
    # SAWS
    print("  Running SAWS...")
    saws_estimates = np.zeros((n_rep, n_steps))
    saws_windows = np.zeros((n_rep, n_steps), dtype=int)
    
    for l in range(n_rep):
        forecaster = SAWSForecaster(
            estimator_fn=var_est,
            loss_fn=var_loss,
            base_grid=base_grid,
            alpha_tau=0.1,
            C_tau=0.5
        )
        est, win = forecaster.forecast(data[l], t_start=t_start, k_init=250)
        saws_estimates[l] = est
        saws_windows[l] = win
    
    methods['SAWS'] = {
        'estimates': saws_estimates,
        'windows': saws_windows,
        'metrics': compute_var_metrics_normal(
            saws_estimates, true_means[:, t_start-1:],
            true_stds[:, t_start-1:], data, alpha
        )
    }
    
    # Print metrics
    print("\n  VaR Forecasting Metrics:")
    print(f"  {'Method':<12} {'MAB':>10} {'Var':>10} {'MSE':>10} {'CR':>12} {'CL':>12}")
    print("  " + "-" * 68)
    for name, result in methods.items():
        m = result['metrics']
        print(f"  {name:<12} {m['MAB']:10.4f} {m['Var']:10.4f} {m['MSE']:10.4f} "
              f"{m['CR']:12.4f} {m['CL']:12.4f}")
    
    return methods
