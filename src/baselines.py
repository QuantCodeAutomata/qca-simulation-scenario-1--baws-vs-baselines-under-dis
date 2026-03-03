"""
Baseline forecasting methods for comparison with BAWS.
Implements fixed rolling windows, full recursive window, and SAWS.
"""

from typing import Callable, Tuple
import numpy as np


class FixedWindowForecaster:
    """
    Fixed rolling window forecaster.
    
    Uses a constant window size k for all time steps.
    """
    
    def __init__(self, estimator_fn: Callable, window_size: int):
        """
        Initialize fixed window forecaster.
        
        Parameters
        ----------
        estimator_fn : Callable
            Function to compute parameter estimate from window data
        window_size : int
            Fixed window size
        """
        self.estimator_fn = estimator_fn
        self.window_size = window_size
    
    def forecast(
        self,
        data: np.ndarray,
        t_start: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform recursive forecasting with fixed window.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series data
        t_start : int
            Starting time index for forecasts
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (estimates, window_sizes) arrays
        """
        T = len(data)
        n_forecasts = T - t_start + 1
        
        estimates = np.zeros(n_forecasts)
        windows = np.full(n_forecasts, self.window_size, dtype=int)
        
        for idx, t in enumerate(range(t_start, T + 1)):
            # Use min(window_size, t-1) to handle initial periods
            k = min(self.window_size, t - 1)
            window_data = data[t-k:t]
            estimates[idx] = self.estimator_fn(window_data)
            windows[idx] = k
        
        return estimates, windows


class FullWindowForecaster:
    """
    Full recursive window forecaster.
    
    Uses all available historical data at each time step.
    """
    
    def __init__(self, estimator_fn: Callable):
        """
        Initialize full window forecaster.
        
        Parameters
        ----------
        estimator_fn : Callable
            Function to compute parameter estimate from window data
        """
        self.estimator_fn = estimator_fn
    
    def forecast(
        self,
        data: np.ndarray,
        t_start: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform recursive forecasting with full window.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series data
        t_start : int
            Starting time index for forecasts
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (estimates, window_sizes) arrays
        """
        T = len(data)
        n_forecasts = T - t_start + 1
        
        estimates = np.zeros(n_forecasts)
        windows = np.zeros(n_forecasts, dtype=int)
        
        for idx, t in enumerate(range(t_start, T + 1)):
            # Use all data from 0 to t-1
            window_data = data[:t]
            estimates[idx] = self.estimator_fn(window_data)
            windows[idx] = t - 1
        
        return estimates, windows


class SAWSForecaster:
    """
    SAWS (Smoothed Adaptive Window Selection) forecaster.
    
    Simplified implementation based on Huang & Wang (2025) methodology.
    Uses a heuristic adaptive threshold with smoothing.
    
    Note: This is a placeholder implementation. Full SAWS requires
    the complete methodology from the referenced paper.
    """
    
    def __init__(
        self,
        estimator_fn: Callable,
        loss_fn: Callable,
        base_grid: list,
        alpha_tau: float = 0.1,
        C_tau: float = 0.3,
        max_window: int = None
    ):
        """
        Initialize SAWS forecaster.
        
        Parameters
        ----------
        estimator_fn : Callable
            Function to compute parameter estimate
        loss_fn : Callable
            Loss function for evaluation
        base_grid : list
            Base candidate window grid
        alpha_tau : float
            Threshold parameter alpha
        C_tau : float
            Threshold parameter C
        max_window : int, optional
            Maximum window size
        """
        self.estimator_fn = estimator_fn
        self.loss_fn = loss_fn
        self.base_grid = base_grid
        self.alpha_tau = alpha_tau
        self.C_tau = C_tau
        self.max_window = max_window
    
    def forecast(
        self,
        data: np.ndarray,
        t_start: int,
        k_init: int = 250,
        base_seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform recursive forecasting with SAWS.
        
        This is a simplified heuristic implementation that adaptively
        selects windows based on loss comparisons with a smoothed threshold.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series data
        t_start : int
            Starting time index
        k_init : int
            Initial window size
        base_seed : int, optional
            Random seed (for compatibility)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (estimates, selected_windows) arrays
        """
        T = len(data)
        n_forecasts = T - t_start + 1
        
        estimates = np.zeros(n_forecasts)
        windows = np.zeros(n_forecasts, dtype=int)
        
        k_prev = k_init
        
        for idx, t in enumerate(range(t_start, T + 1)):
            # Build candidate grid
            max_k = self.max_window if self.max_window else t - 1
            K_t = [k for k in self.base_grid if k <= max_k]
            
            if not K_t:
                K_t = [min(t - 1, k_init)]
            
            # Compute estimates and losses for candidates
            best_k = K_t[0]
            best_loss = float('inf')
            
            for k in K_t:
                window_data = data[t-k:t]
                theta_k = self.estimator_fn(window_data)
                loss_k = np.mean(self.loss_fn(window_data, theta_k))
                
                # Simple adaptive threshold: accept if loss is below threshold
                # Threshold increases with window size (penalize large windows)
                threshold = self.C_tau * (1 + self.alpha_tau * np.log(1 + k / 100))
                
                if loss_k < best_loss + threshold:
                    best_loss = loss_k
                    best_k = k
            
            # Use exponential smoothing with previous window
            # to avoid erratic jumps (SAWS smoothing heuristic)
            smoothing_factor = 0.3
            k_selected = int(smoothing_factor * k_prev + (1 - smoothing_factor) * best_k)
            k_selected = max(min(K_t), min(k_selected, max_k))
            
            # Compute final estimate
            window_data = data[t-k_selected:t]
            theta_hat = self.estimator_fn(window_data)
            
            estimates[idx] = theta_hat
            windows[idx] = k_selected
            
            k_prev = k_selected
        
        return estimates, windows
