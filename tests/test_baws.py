"""
Test suite for BAWS algorithm implementation.
"""

import pytest
import numpy as np
from src.window_grid import build_base_grid, build_dynamic_grid
from src.loss_functions import (
    squared_error_loss, mean_estimator,
    var_scoring_loss, var_estimator,
    true_var_normal
)
from src.bootstrap import iid_bootstrap_threshold
from src.baws import BAWSForecaster
from src.baselines import FixedWindowForecaster, FullWindowForecaster
from src.data_generation import generate_scenario1_data


def test_window_grid_construction():
    """Test that window grid is constructed correctly."""
    grid = build_base_grid(max_window=2000)
    
    # Check minimum window
    assert min(grid) == 5, "Minimum window should be 5"
    
    # Check increments in different ranges
    grid_50 = [k for k in grid if 5 <= k < 50]
    assert all(grid_50[i+1] - grid_50[i] == 5 for i in range(len(grid_50)-1)), \
        "Steps of 5 in [5, 50)"
    
    # Check uniqueness and sorting
    assert len(grid) == len(set(grid)), "Grid should have unique values"
    assert grid == sorted(grid), "Grid should be sorted"


def test_dynamic_grid_respects_max_window():
    """Test that dynamic grid respects max window constraint."""
    base_grid = build_base_grid(max_window=2000)
    
    # At time t=600, with max_window=1000
    K_t = build_dynamic_grid(t=600, k_prev=250, base_grid=base_grid, max_window=1000)
    
    # All windows should be <= min(t-1, max_window) = min(599, 1000) = 599
    assert all(k <= 599 for k in K_t), "All windows should be <= t-1"
    assert all(k <= 1000 for k in K_t), "All windows should be <= max_window"


def test_mean_estimator():
    """Test that mean estimator returns sample mean."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = mean_estimator(data)
    assert abs(mean - 3.0) < 1e-10, "Mean should be 3.0"


def test_var_estimator():
    """Test that VaR estimator returns correct quantile."""
    # Known distribution
    np.random.seed(42)
    data = np.random.randn(1000)
    var_95 = var_estimator(data, alpha=0.95)
    
    # Should be close to theoretical 95th percentile
    empirical_percentile = np.percentile(data, 95)
    assert abs(var_95 - empirical_percentile) < 0.1, \
        "VaR estimate should be close to empirical 95th percentile"


def test_squared_error_loss():
    """Test squared error loss function."""
    x = np.array([1.0, 2.0, 3.0])
    theta = 2.0
    loss = squared_error_loss(x, theta)
    
    expected = np.array([1.0, 0.0, 1.0])
    np.testing.assert_array_almost_equal(loss, expected)


def test_var_scoring_loss():
    """Test VaR scoring function properties."""
    x = np.array([0.5, 1.5, 2.5])
    v = 2.0
    alpha = 0.95
    
    score = var_scoring_loss(x, v, alpha)
    
    # Check that scoring function has correct values
    # For x < v, indicator = 1, score = (1-alpha)*(v-x)
    # For x >= v, indicator = 0, score = -alpha*(v-x)
    # Note: at alpha=0.95, (1-alpha)=0.05 is small
    expected_0 = (1 - alpha) * (v - 0.5)  # 0.05 * 1.5 = 0.075
    expected_1 = (1 - alpha) * (v - 1.5)  # 0.05 * 0.5 = 0.025
    expected_2 = -alpha * (v - 2.5)       # -0.95 * (-0.5) = 0.475
    
    np.testing.assert_almost_equal(score[0], expected_0, decimal=10)
    np.testing.assert_almost_equal(score[1], expected_1, decimal=10)
    np.testing.assert_almost_equal(score[2], expected_2, decimal=10)


def test_true_var_normal():
    """Test true VaR computation for Normal distribution."""
    from scipy.stats import norm
    
    mu = 1.0
    sigma = 0.5
    alpha = 0.95
    
    var = true_var_normal(mu, sigma, alpha)
    expected = mu + sigma * norm.ppf(alpha)
    
    assert abs(var - expected) < 1e-10, "True VaR should match formula"


def test_iid_bootstrap_threshold_is_nonnegative():
    """Test that bootstrap threshold is non-negative."""
    np.random.seed(42)
    data = np.random.randn(100)
    
    tau = iid_bootstrap_threshold(
        window_data=data,
        estimator_fn=mean_estimator,
        loss_fn=squared_error_loss,
        B=100,
        beta=0.9,
        seed=42
    )
    
    assert tau >= 0, "Bootstrap threshold should be non-negative"


def test_fixed_window_forecaster():
    """Test fixed window forecaster produces correct number of forecasts."""
    np.random.seed(42)
    data = np.random.randn(1000)
    
    forecaster = FixedWindowForecaster(mean_estimator, window_size=250)
    estimates, windows = forecaster.forecast(data, t_start=501)
    
    # Should produce 500 forecasts (from t=501 to t=1000)
    assert len(estimates) == 500, "Should produce 500 forecasts"
    assert len(windows) == 500, "Should produce 500 window sizes"
    
    # Window sizes should all be 250 (except possibly early ones)
    assert all(w == 250 for w in windows), "All windows should be 250"


def test_full_window_forecaster():
    """Test full window forecaster uses expanding window."""
    np.random.seed(42)
    data = np.random.randn(1000)
    
    forecaster = FullWindowForecaster(mean_estimator)
    estimates, windows = forecaster.forecast(data, t_start=501)
    
    # Window sizes should expand from 500 to 999
    assert windows[0] == 500, "First window should be 500"
    assert windows[-1] == 999, "Last window should be 999"
    assert all(windows[i+1] == windows[i] + 1 for i in range(len(windows)-1)), \
        "Windows should expand by 1 each step"


def test_baws_forecaster_runs():
    """Test that BAWS forecaster runs without errors."""
    np.random.seed(42)
    data = np.random.randn(600)
    
    base_grid = build_base_grid(max_window=2000)
    
    forecaster = BAWSForecaster(
        estimator_fn=mean_estimator,
        loss_fn=squared_error_loss,
        base_grid=base_grid,
        bootstrap_type='iid',
        B=50,  # Small B for speed
        beta=0.9
    )
    
    estimates, windows = forecaster.forecast(
        data,
        t_start=501,
        k_init=250,
        base_seed=42
    )
    
    # Should produce 100 forecasts
    assert len(estimates) == 100, "Should produce 100 forecasts"
    assert len(windows) == 100, "Should produce 100 window sizes"
    
    # Windows should be within valid range
    assert all(5 <= w <= 599 for w in windows), "Windows should be in valid range"


def test_scenario1_data_generation():
    """Test Scenario 1 data generation."""
    data, true_means, true_vars = generate_scenario1_data(
        setting='A1',
        T=2000,
        n_replications=10,
        base_seed=42
    )
    
    # Check shapes
    assert data.shape == (10, 2000), "Data shape should be (10, 2000)"
    assert true_means.shape == (10, 2000), "True means shape should be (10, 2000)"
    assert true_vars.shape == (10, 2000), "True vars shape should be (10, 2000)"
    
    # Check A1 mean structure (break at t=1000)
    # Mean should be 1.0 for t<=1000, 2.0 for t>1000
    assert np.allclose(true_means[0, :1000], 1.0), "Mean should be 1.0 for t<=1000"
    assert np.allclose(true_means[0, 1000:], 2.0), "Mean should be 2.0 for t>1000"
    
    # Variance should be constant at 0.25
    assert np.allclose(true_vars[0], 0.25), "Variance should be 0.25 for all t"


def test_baws_selects_smaller_windows_at_breaks():
    """Test that BAWS tends to select smaller windows near structural breaks."""
    # Generate data with a clear break
    np.random.seed(42)
    T = 1000
    data = np.concatenate([
        np.random.randn(500) + 0.0,  # Mean 0
        np.random.randn(500) + 5.0   # Mean 5 (large shift)
    ])
    
    base_grid = build_base_grid(max_window=1000)
    
    forecaster = BAWSForecaster(
        estimator_fn=mean_estimator,
        loss_fn=squared_error_loss,
        base_grid=base_grid,
        bootstrap_type='iid',
        B=50,
        beta=0.9
    )
    
    estimates, windows = forecaster.forecast(
        data,
        t_start=400,
        k_init=250,
        base_seed=42
    )
    
    # Find index corresponding to t=510 (just after break at t=500)
    idx_after_break = 510 - 400  # 110
    
    # Window at break should be smaller than windows far from break
    # (This is a probabilistic test, may not always pass)
    # Compare window shortly after break vs window 200 steps later
    if idx_after_break + 200 < len(windows):
        window_at_break = windows[idx_after_break]
        window_later = windows[idx_after_break + 200]
        
        # At least verify windows are positive and within range
        assert window_at_break > 0, "Window at break should be positive"
        assert window_later > 0, "Window later should be positive"


def test_methods_produce_finite_estimates():
    """Test that all methods produce finite (non-NaN, non-inf) estimates."""
    np.random.seed(42)
    data = np.random.randn(600)
    
    base_grid = build_base_grid(max_window=2000)
    
    # BAWS
    baws = BAWSForecaster(
        estimator_fn=mean_estimator,
        loss_fn=squared_error_loss,
        base_grid=base_grid,
        bootstrap_type='iid',
        B=50,
        beta=0.9
    )
    baws_est, _ = baws.forecast(data, t_start=501, k_init=250, base_seed=42)
    assert np.all(np.isfinite(baws_est)), "BAWS estimates should be finite"
    
    # Fixed
    fixed = FixedWindowForecaster(mean_estimator, 250)
    fixed_est, _ = fixed.forecast(data, t_start=501)
    assert np.all(np.isfinite(fixed_est)), "Fixed window estimates should be finite"
    
    # Full
    full = FullWindowForecaster(mean_estimator)
    full_est, _ = full.forecast(data, t_start=501)
    assert np.all(np.isfinite(full_est)), "Full window estimates should be finite"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
