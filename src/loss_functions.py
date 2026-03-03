"""
Loss functions and their minimizers for BAWS algorithm.
Implements mean forecasting (squared error) and VaR forecasting (scoring function).
"""

from typing import Callable, Tuple
import numpy as np
from scipy import optimize


def squared_error_loss(x: np.ndarray, theta: float) -> np.ndarray:
    """
    Squared error loss function: (x - theta)^2
    
    Parameters
    ----------
    x : np.ndarray
        Observed values
    theta : float
        Parameter (mean estimate)
        
    Returns
    -------
    np.ndarray
        Loss values
    """
    return (x - theta) ** 2


def mean_estimator(x: np.ndarray) -> float:
    """
    Closed-form minimizer of squared error loss: sample mean.
    
    Parameters
    ----------
    x : np.ndarray
        Window data
        
    Returns
    -------
    float
        Sample mean
    """
    return np.mean(x)


def var_scoring_loss(x: np.ndarray, v: float, alpha: float = 0.95) -> np.ndarray:
    """
    VaR scoring function (quantile scoring): S_V(x,v) = (1{x<v} - alpha) * (v - x)
    
    Parameters
    ----------
    x : np.ndarray
        Observed values
    v : float
        VaR estimate
    alpha : float
        Confidence level (default: 0.95)
        
    Returns
    -------
    np.ndarray
        Scoring loss values
    """
    indicator = (x < v).astype(float)
    return (indicator - alpha) * (v - x)


def var_estimator(x: np.ndarray, alpha: float = 0.95) -> float:
    """
    Empirical alpha-quantile of x using the infimum convention.
    
    For alpha = 0.95, returns inf{v: F_hat(v) >= 0.95}
    
    Parameters
    ----------
    x : np.ndarray
        Window data
    alpha : float
        Confidence level (default: 0.95)
        
    Returns
    -------
    float
        Empirical VaR estimate
    """
    # Sort data
    x_sorted = np.sort(x)
    n = len(x)
    
    # Compute the index: ceil(alpha * n) - 1 (0-indexed)
    # This gives the smallest value such that at least alpha fraction of data is <= it
    idx = int(np.ceil(alpha * n)) - 1
    idx = min(max(idx, 0), n - 1)  # Clip to valid range
    
    return x_sorted[idx]


def compute_window_loss(x: np.ndarray, theta: float, loss_fn: Callable) -> float:
    """
    Compute average loss over a window.
    
    Parameters
    ----------
    x : np.ndarray
        Window data
    theta : float
        Parameter estimate
    loss_fn : Callable
        Loss function that takes (x, theta) and returns loss values
        
    Returns
    -------
    float
        Average loss
    """
    return np.mean(loss_fn(x, theta))


def true_var_normal(mu: float, sigma: float, alpha: float = 0.95) -> float:
    """
    True VaR for Normal distribution: VaR = mu + sigma * Phi^{-1}(alpha)
    
    Parameters
    ----------
    mu : float
        Mean
    sigma : float
        Standard deviation
    alpha : float
        Confidence level
        
    Returns
    -------
    float
        True VaR
    """
    from scipy.stats import norm
    return mu + sigma * norm.ppf(alpha)


def excess_var_risk_normal(v_hat: float, v_true: float, mu: float, 
                           sigma: float, alpha: float = 0.95) -> float:
    """
    Excess population VaR risk under Normal distribution.
    
    F(v) = E[S_V(X, v)] where X ~ N(mu, sigma^2)
    Excess risk = F(v_hat) - F(v_true)
    
    Parameters
    ----------
    v_hat : float
        Estimated VaR
    v_true : float
        True VaR
    mu : float
        True mean
    sigma : float
        True standard deviation
    alpha : float
        Confidence level
        
    Returns
    -------
    float
        Excess risk
    """
    from scipy.stats import norm
    
    def population_var_score(v, mu, sigma, alpha):
        """Expected VaR score E[(1{X<v}-alpha)(v-X)] under N(mu, sigma^2)"""
        z = (v - mu) / sigma
        cdf = norm.cdf(z)
        pdf = norm.pdf(z)
        
        # E[S_V(X,v)] = alpha*v - alpha*mu - sigma*pdf(z) - v*(1-cdf) + (mu + sigma*z)*(1-cdf)
        # Simplifies to: (cdf - alpha)*v - sigma*pdf(z) - (1-cdf)*mu
        score = (cdf - alpha) * v - sigma * pdf - (1 - cdf) * mu
        return score
    
    risk_hat = population_var_score(v_hat, mu, sigma, alpha)
    risk_true = population_var_score(v_true, mu, sigma, alpha)
    
    return risk_hat - risk_true
