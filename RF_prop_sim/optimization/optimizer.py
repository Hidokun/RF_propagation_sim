import numpy as np
from scipy.optimize import minimize
import sys
import os

# Import the propagation models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from propagation_model import free_space_path_loss

def objective_function(tx_params, rx_grid, target_power_dbm=-80, tx_power_dbm=40, freq_mhz=900):
    """
    Objective function to minimize.
    Goal: Maximize coverage area and minimize energy variance (even distribution).
    
    tx_params: [tx_x, tx_y, tx_height]
    rx_grid: List of receiver coordinates [[x1, y1], [x2, y2], ...] in km (relative to center)
    """
    tx_x, tx_y, tx_height = tx_params
    
    losses = []
    for rx in rx_grid:
        # Calculate 3D distance in km
        # height difference in km
        dh = (tx_height - 1.5) / 1000.0 
        dist_km = np.sqrt((tx_x - rx[0])**2 + (tx_y - rx[1])**2 + dh**2)
        
        # Using free space as a fast optimizer proxy
        # In a full simulation, we could use ITM or Ray Tracing if compute allows
        loss = free_space_path_loss(freq_mhz, dist_km)
        losses.append(loss)
        
    losses = np.array(losses)
    rx_power = tx_power_dbm - losses
    
    # We want rx_power to be close to target_power_dbm, but not overly powerful (wasting energy)
    # and not too weak (lack of coverage)
    
    # Penalty for being below target (coverage loss)
    under_coverage_penalty = np.sum(np.maximum(0, target_power_dbm - rx_power)**2)
    
    # Penalty for being significantly above target (wasted energy/high cost)
    over_coverage_penalty = np.sum(np.maximum(0, rx_power - (target_power_dbm + 10))**2) * 0.5
    
    # Variance penalty to ensure even energy repartition
    variance_penalty = np.var(rx_power) * 10
    
    total_cost = under_coverage_penalty + over_coverage_penalty + variance_penalty
    
    return total_cost

def optimize_antenna_placement(area_bounds_km=1.0, freq_mhz=900, tx_power_dbm=40):
    """
    Find the optimal [x, y, height] for an antenna to cover a grid.
    """
    print("Initializing Optimization Engine...")
    
    # Create a grid of receivers in a 1km x 1km area
    rx_grid = []
    steps = np.linspace(-area_bounds_km/2, area_bounds_km/2, 10)
    for x in steps:
        for y in steps:
            rx_grid.append([x, y])
            
    # Initial guess: Center of the area, 30m high
    initial_guess = [0.0, 0.0, 30.0]
    
    # Bounds: Keep within the area, height between 10m and 100m
    bnds = ((-area_bounds_km/2, area_bounds_km/2), 
            (-area_bounds_km/2, area_bounds_km/2), 
            (10.0, 100.0))
            
    print("Running optimization algorithm (minimizing energy waste & maximizing coverage)...")
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(rx_grid, -80, tx_power_dbm, freq_mhz),
        bounds=bnds,
        method='L-BFGS-B'
    )
    
    if result.success:
        print("\nOptimization Successful!")
        print(f"Optimal Tx Position: X={result.x[0]:.3f} km, Y={result.x[1]:.3f} km")
        print(f"Optimal Antenna Height: {result.x[2]:.1f} meters")
        print(f"Minimum Cost Achieved: {result.fun:.2f}")
        return result.x
    else:
        print("\nOptimization Failed to converge.")
        return initial_guess
