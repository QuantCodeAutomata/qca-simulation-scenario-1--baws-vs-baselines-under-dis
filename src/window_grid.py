"""
Window grid construction for BAWS algorithm.
Implements the piecewise increment strategy for candidate window selection.
"""

from typing import List, Set
import numpy as np


def build_base_grid(max_window: int = 2000) -> List[int]:
    """
    Build the base candidate window grid with piecewise increments.
    
    Steps:
    - Increments of 5 for windows in [5, 50)
    - Increments of 10 for windows in [50, 110]
    - Increments of 20 for windows in (110, 320]
    - Increments of 50 for windows in (320, 1050]
    - Increments of 100 for windows > 1050
    
    Parameters
    ----------
    max_window : int
        Maximum window size to include in the grid
        
    Returns
    -------
    List[int]
        Sorted list of candidate window sizes
    """
    grid = []
    
    # Steps of 5 for [5, 50)
    grid.extend(range(5, 50, 5))
    
    # Steps of 10 for [50, 110]
    grid.extend(range(50, 110, 10))
    
    # Steps of 20 for (110, 320]
    grid.extend(range(120, 320, 20))
    
    # Steps of 50 for (320, 1050]
    grid.extend(range(350, min(1050, max_window + 1), 50))
    
    # Steps of 100 for > 1050
    if max_window > 1050:
        grid.extend(range(1100, max_window + 1, 100))
    
    # Remove duplicates and sort
    grid = sorted(set(grid))
    
    # Ensure max_window is included if specified
    if max_window not in grid and max_window >= 5:
        grid.append(max_window)
        grid = sorted(grid)
    
    return grid


def build_dynamic_grid(t: int, k_prev: int, base_grid: List[int], 
                       max_window: int = None) -> List[int]:
    """
    Build the dynamic candidate window grid at time t.
    
    K_t includes:
    - All base grid windows <= t-1
    - The previous selected window k_{t-1}
    - Exploratory windows: k_{t-1}+50, k_{t-1}+100, ... up to min(t-1, max_window)
    
    Parameters
    ----------
    t : int
        Current time index
    k_prev : int
        Previously selected window size
    base_grid : List[int]
        Base candidate grid
    max_window : int, optional
        Maximum allowed window size (defaults to t-1)
        
    Returns
    -------
    List[int]
        Sorted list of candidate windows at time t
    """
    if max_window is None:
        max_window = t - 1
    else:
        max_window = min(max_window, t - 1)
    
    # Start with base grid elements <= max_window
    K_t = [k for k in base_grid if k <= max_window]
    
    # Add previous window if valid
    if k_prev <= max_window and k_prev >= 5:
        K_t.append(k_prev)
    
    # Add exploratory windows above k_prev in steps of 50
    if k_prev < max_window:
        exploratory = list(range(k_prev + 50, max_window + 1, 50))
        K_t.extend(exploratory)
    
    # Remove duplicates and sort
    K_t = sorted(set(K_t))
    
    return K_t
