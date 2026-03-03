"""
Bootstrap methods for BAWS threshold computation.
Implements i.i.d. bootstrap and moving block bootstrap.
"""

from typing import Callable, List
import numpy as np


def iid_bootstrap_threshold(
    window_data: np.ndarray,
    estimator_fn: Callable,
    loss_fn: Callable,
    B: int = 500,
    beta: float = 0.9,
    seed: int = None
) -> float:
    """
    Compute bootstrap threshold tau(t,i) using i.i.d. bootstrap.
    
    Algorithm:
    1. For b=1,...,B:
       - Resample i observations with replacement from window_data
       - Compute bootstrap estimator theta_hat^{(b)}
       - Evaluate on original data: d^{(b)} = f_{t,i}(theta_hat^{(b)}) - f_{t,i}(theta_hat)
    2. tau = empirical beta-quantile of {d^{(b)}}
    
    Parameters
    ----------
    window_data : np.ndarray
        Original window data of length i
    estimator_fn : Callable
        Function that takes data and returns parameter estimate
    loss_fn : Callable
        Function that takes (x, theta) and returns loss values
    B : int
        Number of bootstrap replications (default: 500)
    beta : float
        Quantile level for threshold (default: 0.9)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    float
        Bootstrap threshold tau(t,i)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    n = len(window_data)
    
    # Original estimator
    theta_hat = estimator_fn(window_data)
    f_original = np.mean(loss_fn(window_data, theta_hat))
    
    # Bootstrap
    d_values = []
    for b in range(B):
        # Resample with replacement
        bootstrap_sample = rng.choice(window_data, size=n, replace=True)
        
        # Compute bootstrap estimator
        theta_hat_b = estimator_fn(bootstrap_sample)
        
        # Evaluate on original data
        f_bootstrap = np.mean(loss_fn(window_data, theta_hat_b))
        
        # Excess risk
        d_b = f_bootstrap - f_original
        
        # Clip small negative values (due to numerical errors)
        d_b = max(d_b, 0.0)
        
        d_values.append(d_b)
    
    # Compute threshold
    tau = np.quantile(d_values, beta)
    
    return tau


def moving_block_bootstrap_threshold(
    window_data: np.ndarray,
    estimator_fn: Callable,
    loss_fn: Callable,
    B: int = 500,
    beta: float = 0.9,
    seed: int = None
) -> float:
    """
    Compute bootstrap threshold tau(t,i) using moving block bootstrap.
    
    For time series with temporal dependence (e.g., GARCH).
    Block length: l = ceil(i^{1/3})
    Number of blocks: m = floor(i / l)
    
    Parameters
    ----------
    window_data : np.ndarray
        Original window data of length i (time-ordered)
    estimator_fn : Callable
        Function that takes data and returns parameter estimate
    loss_fn : Callable
        Function that takes (x, theta) and returns loss values
    B : int
        Number of bootstrap replications (default: 500)
    beta : float
        Quantile level for threshold (default: 0.9)
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    float
        Bootstrap threshold tau(t,i)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    n = len(window_data)
    
    # Block length
    block_length = int(np.ceil(n ** (1/3)))
    
    # Number of blocks
    num_blocks = int(np.floor(n / block_length))
    
    # All possible contiguous blocks
    blocks = []
    for start_idx in range(n - block_length + 1):
        block = window_data[start_idx:start_idx + block_length]
        blocks.append(block)
    
    # Original estimator
    theta_hat = estimator_fn(window_data)
    f_original = np.mean(loss_fn(window_data, theta_hat))
    
    # Bootstrap
    d_values = []
    for b in range(B):
        # Sample num_blocks blocks with replacement
        selected_blocks = rng.choice(len(blocks), size=num_blocks, replace=True)
        
        # Concatenate blocks to form bootstrap sample
        bootstrap_sample = np.concatenate([blocks[idx] for idx in selected_blocks])
        
        # Compute bootstrap estimator
        theta_hat_b = estimator_fn(bootstrap_sample)
        
        # Evaluate on original data
        f_bootstrap = np.mean(loss_fn(window_data, theta_hat_b))
        
        # Excess risk
        d_b = f_bootstrap - f_original
        
        # Clip small negative values
        d_b = max(d_b, 0.0)
        
        d_values.append(d_b)
    
    # Compute threshold
    tau = np.quantile(d_values, beta)
    
    return tau
