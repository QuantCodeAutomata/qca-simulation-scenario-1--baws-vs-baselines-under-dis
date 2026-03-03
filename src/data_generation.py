"""
Synthetic data generation for BAWS experiments.
Implements data generation for Scenarios 1, 2, and 3.
"""

from typing import Tuple
import numpy as np
from scipy import stats


def generate_scenario1_data(
    setting: str,
    T: int = 2000,
    n_replications: int = 1000,
    base_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for Scenario 1: Discrete structural breaks in i.i.d. Normal series.
    
    Settings:
    A1: Single mean break at t=1000 (mu: 1->2), sigma^2=0.25
    A2: Double mean break (mu: 1->0->2 at t=800,1400), sigma^2=0.25
    A3: Double mean+variance break (mu: 1->0->2, sigma^2: 0.25->1.0->0.49 at t=800,1400)
    
    Parameters
    ----------
    setting : str
        'A1', 'A2', or 'A3'
    T : int
        Time series length
    n_replications : int
        Number of Monte Carlo replications
    base_seed : int
        Base random seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (data, true_means, true_vars) each of shape (n_replications, T)
    """
    data = np.zeros((n_replications, T))
    true_means = np.zeros((n_replications, T))
    true_vars = np.zeros((n_replications, T))
    
    for l in range(n_replications):
        seed = base_seed * 10000 + l
        rng = np.random.RandomState(seed)
        
        # Generate true parameters
        mu_t = np.zeros(T)
        sigma_t = np.zeros(T)
        
        if setting == 'A1':
            mu_t[:1000] = 1.0
            mu_t[1000:] = 2.0
            sigma_t[:] = 0.5  # std dev
            
        elif setting == 'A2':
            mu_t[:800] = 1.0
            mu_t[800:1400] = 0.0
            mu_t[1400:] = 2.0
            sigma_t[:] = 0.5
            
        elif setting == 'A3':
            mu_t[:800] = 1.0
            mu_t[800:1400] = 0.0
            mu_t[1400:] = 2.0
            sigma_t[:800] = 0.5  # sqrt(0.25)
            sigma_t[800:1400] = 1.0  # sqrt(1.0)
            sigma_t[1400:] = 0.7  # sqrt(0.49)
            
        else:
            raise ValueError(f"Unknown setting: {setting}")
        
        # Generate data
        x_t = mu_t + sigma_t * rng.randn(T)
        
        data[l] = x_t
        true_means[l] = mu_t
        true_vars[l] = sigma_t ** 2
    
    return data, true_means, true_vars


def generate_scenario2_data(
    setting: str,
    T: int = 2000,
    n_replications: int = 1000,
    base_seed: int = 43
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for Scenario 2: Continuous mean drift in i.i.d. Normal series.
    
    Settings:
    B1: Sinusoidal drift: mu_t = sin(2*pi*t/T)
    B2: Random walk drift: mu_t ~ RW with innovation std = 1/sqrt(T)
    B3: Geometric Brownian motion drift
    
    Noise variance: 0.25 for all settings
    
    Parameters
    ----------
    setting : str
        'B1', 'B2', or 'B3'
    T : int
        Time series length
    n_replications : int
        Number of Monte Carlo replications
    base_seed : int
        Base random seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (data, true_means, true_vars)
    """
    data = np.zeros((n_replications, T))
    true_means = np.zeros((n_replications, T))
    true_vars = np.zeros((n_replications, T))
    
    for l in range(n_replications):
        # Separate seeds for drift and noise
        seed_drift = base_seed * 100000 + l * 10 + 1
        seed_noise = base_seed * 100000 + l * 10 + 2
        rng_drift = np.random.RandomState(seed_drift)
        rng_noise = np.random.RandomState(seed_noise)
        
        # Generate true mean path
        if setting == 'B1':
            # Sinusoidal (deterministic)
            t_arr = np.arange(1, T + 1)
            mu_t = np.sin(2 * np.pi * t_arr / T)
            
        elif setting == 'B2':
            # Random walk
            delta_t = rng_drift.randn(T) / np.sqrt(T)
            mu_t = np.cumsum(delta_t)
            
        elif setting == 'B3':
            # Geometric Brownian motion
            mu_0 = 1.0
            mu_drift = 0.5
            sigma_gbm = 0.5  # sqrt(0.25)
            
            eta_t = rng_drift.randn(T) / np.sqrt(T)
            W_t = np.cumsum(eta_t)
            t_arr = np.arange(1, T + 1) / T
            
            mu_t = mu_0 * np.exp((mu_drift - 0.5 * sigma_gbm**2) * t_arr + sigma_gbm * W_t)
            
        else:
            raise ValueError(f"Unknown setting: {setting}")
        
        # Noise variance (constant)
        sigma_noise = 0.5  # sqrt(0.25)
        
        # Generate observations
        x_t = mu_t + sigma_noise * rng_noise.randn(T)
        
        data[l] = x_t
        true_means[l] = mu_t
        true_vars[l, :] = sigma_noise ** 2
    
    return data, true_means, true_vars


def generate_fernandez_steel_skewtdist(
    n: int,
    nu: float = 5.0,
    r: float = 0.95,
    standardize: bool = True,
    seed: int = None
) -> np.ndarray:
    """
    Generate samples from Fernandez-Steel skewed Student-t distribution.
    
    Parameters
    ----------
    n : int
        Number of samples
    nu : float
        Degrees of freedom
    r : float
        Skewness parameter (ratio of scale right/left)
    standardize : bool
        If True, standardize to mean 0, variance 1
    seed : int, optional
        Random seed
        
    Returns
    -------
    np.ndarray
        Samples from FS skewed-t distribution
    """
    rng = np.random.RandomState(seed)
    
    # Generate standard Student-t samples
    t_samples = stats.t.rvs(df=nu, size=n, random_state=rng)
    
    # Apply skewing transformation
    # With probability r^2/(1+r^2), take -|T|/r; else take |T|*r
    prob_left = r**2 / (1 + r**2)
    u = rng.uniform(0, 1, n)
    
    z_samples = np.where(
        u < prob_left,
        -np.abs(t_samples) / r,  # Left tail
        np.abs(t_samples) * r     # Right tail
    )
    
    if standardize:
        # Compute mean and std from large sample (already done, use empirical)
        mean_z = np.mean(z_samples)
        std_z = np.std(z_samples, ddof=1)
        z_samples = (z_samples - mean_z) / std_z
    
    return z_samples


def generate_scenario3_data(
    T: int = 2000,
    n_replications: int = 1000,
    base_seed: int = 44
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate data for Scenario 3: GARCH(1,1) with skewed-t innovations and regime shift.
    
    L_t = -sigma_t * epsilon_t
    sigma_t^2 = 0.00001 + 0.04*L_{t-1}^2 + gamma_t*sigma_{t-1}^2
    gamma_t = 0.7 for t<=1000, 0.95 for t>1000
    epsilon_t ~ standardized Fernandez-Steel skewed-t(nu=5, r=0.95)
    
    Parameters
    ----------
    T : int
        Time series length
    n_replications : int
        Number of Monte Carlo replications
    base_seed : int
        Base random seed
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]
        (losses, sigmas, true_vars, q_epsilon_005)
        losses: (n_rep, T)
        sigmas: (n_rep, T) conditional volatilities
        true_vars: (n_rep, T) true VaR values
        q_epsilon_005: scalar 0.05-quantile of standardized innovations
    """
    # First, compute the quantile of standardized innovations using large sample
    print("Computing epsilon quantile from large sample...")
    large_sample = generate_fernandez_steel_skewtdist(
        n=10000000,
        nu=5.0,
        r=0.95,
        standardize=True,
        seed=0
    )
    q_epsilon_005 = np.percentile(large_sample, 5)
    print(f"q_epsilon(0.05) = {q_epsilon_005:.6f}")
    
    # GARCH parameters
    omega = 0.00001
    alpha_garch = 0.04
    gamma_1 = 0.7  # t <= 1000
    gamma_2 = 0.95  # t > 1000
    
    # Unconditional variance for first regime
    sigma_1_sq = omega / (1 - alpha_garch - gamma_1)
    
    # Generate data
    losses = np.zeros((n_replications, T))
    sigmas = np.zeros((n_replications, T))
    true_vars = np.zeros((n_replications, T))
    
    for l in range(n_replications):
        if (l + 1) % 100 == 0:
            print(f"  Generating replication {l+1}/{n_replications}...")
        
        seed = base_seed * 100000 + l
        
        # Generate innovations
        epsilon_t = generate_fernandez_steel_skewtdist(
            n=T,
            nu=5.0,
            r=0.95,
            standardize=True,
            seed=seed
        )
        
        # Initialize
        sigma_sq = np.zeros(T + 1)
        sigma_sq[0] = sigma_1_sq
        L_t = np.zeros(T + 1)
        L_t[0] = 0.0
        
        # Simulate GARCH path
        for t in range(T):
            # Conditional volatility
            sigma_t_val = np.sqrt(sigma_sq[t])
            
            # Loss
            L_t[t + 1] = -sigma_t_val * epsilon_t[t]
            
            # True VaR
            true_vars[l, t] = -sigma_t_val * q_epsilon_005
            
            # Next period variance
            gamma_t = gamma_1 if (t + 1) <= 1000 else gamma_2
            sigma_sq[t + 1] = omega + alpha_garch * L_t[t + 1]**2 + gamma_t * sigma_sq[t]
            
            # Store volatility
            sigmas[l, t] = sigma_t_val
        
        # Store losses (skip initialization)
        losses[l] = L_t[1:]
    
    return losses, sigmas, true_vars, q_epsilon_005
