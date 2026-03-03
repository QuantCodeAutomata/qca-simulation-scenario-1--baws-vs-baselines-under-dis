"""
Performance metrics computation for BAWS experiments.
Implements MAB, Variance, MSE, CR, and CL metrics.
"""

import numpy as np
from typing import Dict
from .loss_functions import excess_var_risk_normal


def compute_mean_metrics(
    estimates: np.ndarray,
    true_means: np.ndarray,
    data: np.ndarray
) -> Dict[str, float]:
    """
    Compute performance metrics for mean forecasting.
    
    Parameters
    ----------
    estimates : np.ndarray
        Estimated means, shape (n_replications, n_time_steps)
    true_means : np.ndarray
        True means, shape (n_replications, n_time_steps)
    data : np.ndarray
        Observed data (for CL computation), shape (n_replications, T)
        where T = n_time_steps + t_start - 1
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'MAB', 'Var', 'MSE', 'CR', 'CL'
    """
    n_rep, n_steps = estimates.shape
    
    # Mean Absolute Bias: average over time of |mean_over_replications(bias)|
    bias = estimates - true_means
    mean_bias_per_time = np.mean(bias, axis=0)
    MAB = np.mean(np.abs(mean_bias_per_time))
    
    # Variance: average over time of variance across replications
    var_per_time = np.var(estimates, axis=0, ddof=1)
    Var = np.mean(var_per_time)
    
    # MSE: average over time of mean squared error across replications
    squared_error = (estimates - true_means) ** 2
    mse_per_time = np.mean(squared_error, axis=0)
    MSE = np.mean(mse_per_time)
    
    # Cumulative Risk: mean over replications of sum over time of squared error
    cum_risk_per_rep = np.sum(squared_error, axis=1)
    CR = np.mean(cum_risk_per_rep)
    
    # Cumulative Loss: mean over replications of sum over time of squared forecast error
    # Note: estimates are for time t, so align with data appropriately
    # Assuming data contains the full series and estimates start at t_start
    t_start = data.shape[1] - n_steps
    realized_data = data[:, t_start:]
    forecast_error = (realized_data - estimates) ** 2
    cum_loss_per_rep = np.sum(forecast_error, axis=1)
    CL = np.mean(cum_loss_per_rep)
    
    return {
        'MAB': MAB,
        'Var': Var,
        'MSE': MSE,
        'CR': CR,
        'CL': CL
    }


def compute_var_metrics_normal(
    estimates: np.ndarray,
    true_means: np.ndarray,
    true_stds: np.ndarray,
    data: np.ndarray,
    alpha: float = 0.95
) -> Dict[str, float]:
    """
    Compute performance metrics for VaR forecasting under Normal distribution.
    
    Parameters
    ----------
    estimates : np.ndarray
        Estimated VaR values, shape (n_replications, n_time_steps)
    true_means : np.ndarray
        True means, shape (n_replications, n_time_steps)
    true_stds : np.ndarray
        True standard deviations, shape (n_replications, n_time_steps)
    data : np.ndarray
        Observed data (for CL computation), shape (n_replications, T)
    alpha : float
        VaR confidence level
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'MAB', 'Var', 'MSE', 'CR', 'CL'
    """
    from scipy.stats import norm
    
    n_rep, n_steps = estimates.shape
    
    # True VaR
    true_var = true_means + true_stds * norm.ppf(alpha)
    
    # Mean Absolute Bias
    bias = estimates - true_var
    mean_bias_per_time = np.mean(bias, axis=0)
    MAB = np.mean(np.abs(mean_bias_per_time))
    
    # Variance
    var_per_time = np.var(estimates, axis=0, ddof=1)
    Var = np.mean(var_per_time)
    
    # MSE
    squared_error = (estimates - true_var) ** 2
    mse_per_time = np.mean(squared_error, axis=0)
    MSE = np.mean(mse_per_time)
    
    # Cumulative Risk: mean over replications of sum over time of excess population VaR risk
    excess_risk = np.zeros((n_rep, n_steps))
    for l in range(n_rep):
        for t in range(n_steps):
            excess_risk[l, t] = excess_var_risk_normal(
                estimates[l, t],
                true_var[l, t],
                true_means[l, t],
                true_stds[l, t],
                alpha
            )
    
    cum_risk_per_rep = np.sum(excess_risk, axis=1)
    CR = np.mean(cum_risk_per_rep)
    
    # Cumulative Loss: VaR scoring function
    t_start = data.shape[1] - n_steps
    realized_data = data[:, t_start:]
    
    # S_V(x, v) = (1{x<v} - alpha) * (v - x)
    indicator = (realized_data < estimates).astype(float)
    var_score = (indicator - alpha) * (estimates - realized_data)
    cum_loss_per_rep = np.sum(var_score, axis=1)
    CL = np.mean(cum_loss_per_rep)
    
    return {
        'MAB': MAB,
        'Var': Var,
        'MSE': MSE,
        'CR': CR,
        'CL': CL
    }


def compute_var_metrics_garch(
    estimates: np.ndarray,
    true_vars: np.ndarray,
    sigmas: np.ndarray,
    data: np.ndarray,
    q_epsilon_005: float,
    alpha: float = 0.95
) -> Dict[str, float]:
    """
    Compute performance metrics for VaR forecasting under GARCH with skewed-t.
    
    Parameters
    ----------
    estimates : np.ndarray
        Estimated VaR values, shape (n_replications, n_time_steps)
    true_vars : np.ndarray
        True VaR values, shape (n_replications, n_time_steps)
    sigmas : np.ndarray
        Conditional volatilities, shape (n_replications, T)
    data : np.ndarray
        Observed losses, shape (n_replications, T)
    q_epsilon_005 : float
        0.05-quantile of standardized innovations
    alpha : float
        VaR confidence level
        
    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'MSE', 'Var', 'CR', 'CL'
    """
    n_rep, n_steps = estimates.shape
    
    # Variance
    var_per_time = np.var(estimates, axis=0, ddof=1)
    Var = np.mean(var_per_time)
    
    # MSE
    squared_error = (estimates - true_vars) ** 2
    mse_per_time = np.mean(squared_error, axis=0)
    MSE = np.mean(mse_per_time)
    
    # Cumulative Risk: Approximate using squared error times a scaling factor
    # Full computation would require numerical integration; use approximation
    cum_risk_per_rep = np.sum(squared_error, axis=1)
    CR = np.mean(cum_risk_per_rep)
    
    # Cumulative Loss: VaR scoring function
    t_start = data.shape[1] - n_steps
    realized_data = data[:, t_start:]
    
    indicator = (realized_data < estimates).astype(float)
    var_score = (indicator - alpha) * (estimates - realized_data)
    cum_loss_per_rep = np.sum(var_score, axis=1)
    CL = np.mean(cum_loss_per_rep)
    
    return {
        'MSE': MSE,
        'Var': Var,
        'CR': CR,
        'CL': CL
    }
