"""
Bootstrap-Based Adaptive Window Selection (BAWS) algorithm.
Main implementation of the BAWS forecasting method.
"""

from typing import Callable, List, Tuple, Dict
import numpy as np
from .window_grid import build_dynamic_grid
from .bootstrap import iid_bootstrap_threshold, moving_block_bootstrap_threshold


class BAWSForecaster:
    """
    BAWS adaptive window selection forecaster.
    
    Attributes
    ----------
    estimator_fn : Callable
        Function to compute parameter estimate from window data
    loss_fn : Callable
        Loss function for evaluation
    base_grid : List[int]
        Base candidate window grid
    bootstrap_type : str
        Type of bootstrap ('iid' or 'block')
    B : int
        Number of bootstrap replications
    beta : float
        Bootstrap threshold quantile level
    max_window : int, optional
        Maximum allowed window size
    """
    
    def __init__(
        self,
        estimator_fn: Callable,
        loss_fn: Callable,
        base_grid: List[int],
        bootstrap_type: str = 'iid',
        B: int = 500,
        beta: float = 0.9,
        max_window: int = None
    ):
        """
        Initialize BAWS forecaster.
        
        Parameters
        ----------
        estimator_fn : Callable
            Function that takes window data and returns parameter estimate
        loss_fn : Callable
            Function that takes (x, theta) and returns loss values
        base_grid : List[int]
            Base candidate window grid
        bootstrap_type : str
            'iid' for i.i.d. bootstrap, 'block' for moving block bootstrap
        B : int
            Number of bootstrap replications
        beta : float
            Bootstrap threshold quantile level
        max_window : int, optional
            Maximum allowed window size
        """
        self.estimator_fn = estimator_fn
        self.loss_fn = loss_fn
        self.base_grid = base_grid
        self.bootstrap_type = bootstrap_type
        self.B = B
        self.beta = beta
        self.max_window = max_window
        
        # History tracking
        self.selected_windows = []
        self.estimates = []
    
    def _compute_window_estimate(self, data: np.ndarray, t: int, k: int) -> float:
        """
        Compute parameter estimate for window of size k ending at t-1.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series data
        t : int
            Current time index
        k : int
            Window size
            
        Returns
        -------
        float
            Parameter estimate
        """
        window_data = data[t-k:t]
        return self.estimator_fn(window_data)
    
    def _compute_window_loss(self, data: np.ndarray, t: int, k: int, theta: float) -> float:
        """
        Compute average loss over window of size k using estimate theta.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series data
        t : int
            Current time index
        k : int
            Window size
        theta : float
            Parameter estimate
            
        Returns
        -------
        float
            Average loss over window
        """
        window_data = data[t-k:t]
        return np.mean(self.loss_fn(window_data, theta))
    
    def _compute_bootstrap_threshold(
        self,
        data: np.ndarray,
        t: int,
        i: int,
        seed: int = None
    ) -> float:
        """
        Compute bootstrap threshold tau(t,i) for reference window i.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series data
        t : int
            Current time index
        i : int
            Reference window size
        seed : int, optional
            Random seed
            
        Returns
        -------
        float
            Bootstrap threshold
        """
        window_data = data[t-i:t]
        
        if self.bootstrap_type == 'iid':
            tau = iid_bootstrap_threshold(
                window_data,
                self.estimator_fn,
                self.loss_fn,
                B=self.B,
                beta=self.beta,
                seed=seed
            )
        elif self.bootstrap_type == 'block':
            tau = moving_block_bootstrap_threshold(
                window_data,
                self.estimator_fn,
                self.loss_fn,
                B=self.B,
                beta=self.beta,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown bootstrap type: {self.bootstrap_type}")
        
        return tau
    
    def forecast_one_step(
        self,
        data: np.ndarray,
        t: int,
        k_prev: int,
        seed: int = None
    ) -> Tuple[float, int]:
        """
        Perform one-step-ahead forecast at time t using BAWS.
        
        Parameters
        ----------
        data : np.ndarray
            Time series data up to t-1
        t : int
            Current time index (forecast for t, using data up to t-1)
        k_prev : int
            Previously selected window size
        seed : int, optional
            Random seed for bootstrap
            
        Returns
        -------
        Tuple[float, int]
            (estimate, selected_window)
        """
        # Build dynamic candidate grid
        K_t = build_dynamic_grid(t, k_prev, self.base_grid, self.max_window)
        
        # Compute estimates for all candidates
        estimates = {}
        for k in K_t:
            estimates[k] = self._compute_window_estimate(data, t, k)
        
        # Compute bootstrap thresholds for all reference windows
        # Use deterministic seed based on t if provided
        thresholds = {}
        for i in K_t:
            ref_seed = None if seed is None else (seed + i * 1000)
            thresholds[i] = self._compute_bootstrap_threshold(data, t, i, ref_seed)
        
        # Check admissibility for each candidate
        admissible = []
        for k in K_t:
            is_admissible = True
            
            # Test against all smaller reference windows
            for i in K_t:
                if i >= k:
                    continue
                
                # Compute Delta_{t;i,k} = f_{t,i}(theta_hat_{t,k}) - f_{t,i}(theta_hat_{t,i})
                loss_k = self._compute_window_loss(data, t, i, estimates[k])
                loss_i = self._compute_window_loss(data, t, i, estimates[i])
                delta = loss_k - loss_i
                
                # Check if admissible
                if delta > thresholds[i]:
                    is_admissible = False
                    break
            
            if is_admissible:
                admissible.append(k)
        
        # Select maximum admissible window
        if len(admissible) == 0:
            # Fallback to minimum window if none admissible (should be rare)
            k_hat = min(K_t)
        else:
            k_hat = max(admissible)
        
        theta_hat = estimates[k_hat]
        
        return theta_hat, k_hat
    
    def forecast(
        self,
        data: np.ndarray,
        t_start: int,
        k_init: int = 250,
        base_seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform recursive forecasting from t_start to end of data.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series data
        t_start : int
            Starting time index for forecasts
        k_init : int
            Initial window size (k_{t_start-1})
        base_seed : int, optional
            Base random seed
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (estimates, selected_windows) arrays of length T - t_start + 1
        """
        T = len(data)
        n_forecasts = T - t_start + 1
        
        estimates_arr = np.zeros(n_forecasts)
        windows_arr = np.zeros(n_forecasts, dtype=int)
        
        k_prev = k_init
        
        for idx, t in enumerate(range(t_start, T + 1)):
            # Deterministic seed for reproducibility
            seed = None if base_seed is None else (base_seed + t * 10000)
            
            theta_hat, k_hat = self.forecast_one_step(data[:t], t, k_prev, seed)
            
            estimates_arr[idx] = theta_hat
            windows_arr[idx] = k_hat
            
            k_prev = k_hat
        
        self.estimates = estimates_arr
        self.selected_windows = windows_arr
        
        return estimates_arr, windows_arr
