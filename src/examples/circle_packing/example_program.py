import numpy as np
from scipy.optimize import minimize, dual_annealing

def check_overlap(circles, tolerance=1e-9):
    """Checks for overlaps between circles."""
    num_circles = circles.shape[0]
    if num_circles < 2:
        return False
    
    # Vectorized calculation of squared distances
    centers = circles[:, :2]
    radii = circles[:, 2]

    # Calculate all pairwise distances
    # From https://stackoverflow.com/questions/37009647/compute-pairwise-euclidean-distances-using-numpy
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)

    # Calculate sum of radii squared for all pairs
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = min_dist**2

    # Check for overlaps (excluding self-interaction)
    # Set diagonal to infinity to ignore self-comparison
    np.fill_diagonal(dist_sq, np.inf) 
    
    # Check if any squared distance is less than (sum of radii)^2 - tolerance
    # Only need to check the upper triangle (since dist_sq is symmetric)
    return np.any(dist_sq < min_dist_sq - tolerance)

def check_bounds(circles, tolerance=1e-9):
    """Checks if circles are within the unit square [0,1]x[0,1]."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Check if any circle is out of bounds
    out_of_bounds_x = (x - r < 0 - tolerance) | (x + r > 1 + tolerance)
    out_of_bounds_y = (y - r < 0 - tolerance) | (y + r > 1 + tolerance)
    
    return np.any(out_of_bounds_x) or np.any(out_of_bounds_y)

def score_packing(circles):
    """Calculates the sum of radii if the packing is valid, else 0."""
    if check_overlap(circles) or check_bounds(circles):
        return 0.0
    return np.sum(circles[:, 2])

def _initialize_circles(num_circles: int, initial_radius_factor: float = 1.0, rows_factor: float = 1.0) -> np.ndarray:
    """Initializes circles in a hexagonal packing arrangement."""
    circles = np.zeros((num_circles, 3))
    # Initial radius based on number of circles and area
    initial_radius = np.sqrt(1.0 / (num_circles * np.pi)) * initial_radius_factor

    # Determine a hexagonal grid layout
    aspect_ratio = np.sqrt(3)/2 
    
    rows = int(np.ceil(np.sqrt(num_circles / aspect_ratio) * rows_factor))
    if rows == 0: rows = 1 # Ensure at least one row
    cols = int(np.ceil(num_circles / rows))
    
    base_spacing_x = np.sqrt(3) 
    base_spacing_y = 1.5 
    
    effective_cols_width = cols + 0.5 if rows > 1 else cols 
    
    scale_x = 1.0 / (effective_cols_width * base_spacing_x * initial_radius + initial_radius) if (effective_cols_width * base_spacing_x * initial_radius + initial_radius) > 0 else 1.0
    scale_y = 1.0 / (rows * base_spacing_y * initial_radius + initial_radius) if (rows * base_spacing_y * initial_radius + initial_radius) > 0 else 1.0
    
    scale_factor = min(scale_x, scale_y) * 0.95 

    initial_radius *= scale_factor
    spacing_x = base_spacing_x * initial_radius
    spacing_y = base_spacing_y * initial_radius

    idx = 0
    for i in range(rows):
        offset_x = (i % 2) * spacing_x / 2
        for j in range(cols):
            if idx < num_circles:
                x_center = offset_x + j * spacing_x + initial_radius
                y_center = i * spacing_y + initial_radius
                
                circles[idx, 0] = x_center
                circles[idx, 1] = y_center
                circles[idx, 2] = initial_radius
                idx += 1
    
    # Center the entire packing within the unit square
    min_x = np.min(circles[:, 0] - circles[:, 2])
    max_x = np.max(circles[:, 0] + circles[:, 2])
    min_y = np.min(circles[:, 1] - circles[:, 2])
    max_y = np.max(circles[:, 1] + circles[:, 2])
    
    translate_x = 0.5 - (min_x + max_x) / 2
    translate_y = 0.5 - (min_y + max_y) / 2
    
    circles[:, 0] += translate_x
    circles[:, 1] += translate_y
    
    return circles

def pack_26() -> np.ndarray:
    num_circles = 26
    
    def _calculate_penalties(circles, penalty_weight):
        """Calculates overlap and boundary penalties for a given set of circles."""
        penalty = 0.0
        
        centers = circles[:, :2]
        radii = circles[:, 2]

        # Overlap penalty
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist = np.sqrt(dist_sq)
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        overlap_mask = dist < min_dist - 1e-9 
        overlap_values = np.triu(min_dist - dist, k=1)
        penalty += penalty_weight * np.sum(overlap_values[overlap_mask]**2)

        # Boundary penalties
        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        penalty_x_lower = np.maximum(0, r - x)**2
        penalty_x_upper = np.maximum(0, x + r - 1)**2
        penalty += penalty_weight * np.sum(penalty_x_lower + penalty_x_upper)

        penalty_y_lower = np.maximum(0, r - y)**2
        penalty_y_upper = np.maximum(0, y + r - 1)**2
        penalty += penalty_weight * np.sum(penalty_y_lower + penalty_y_upper)
                
        return penalty

    # Define the objective function for scipy.optimize.minimize
    def objective_function(packed_params, penalty_weight=1000): # Default penalty for L-BFGS-B
        circles = packed_params.reshape(num_circles, 3)
        circles[:, 2] = np.maximum(circles[:, 2], 1e-7)
        sum_radii = np.sum(circles[:, 2])
        penalty = _calculate_penalties(circles, penalty_weight)
        return -sum_radii + penalty

    # Define bounds for x, y, and r for each circle
    bounds = []
    for i in range(num_circles):
        bounds.append((0.0, 1.0)) 
        bounds.append((0.0, 1.0)) 
        bounds.append((1e-7, 0.5)) 
    
    # Define objective function for dual_annealing (same as for L-BFGS-B but with adaptable penalty)
    def objective_function_sa(packed_params, penalty_weight_sa):
        circles = packed_params.reshape(num_circles, 3)
        circles[:, 2] = np.maximum(circles[:, 2], 1e-7)
        sum_radii = np.sum(circles[:, 2])
        penalty = _calculate_penalties(circles, penalty_weight_sa)
        return -sum_radii + penalty

    # Define objective function for radii-only optimization
    def objective_function_radii_only(radii_params, fixed_centers_val, penalty_weight_radii):
        radii = np.maximum(radii_params, 1e-7)
        circles_temp = np.hstack((fixed_centers_val, radii.reshape(-1, 1)))
        
        sum_radii = np.sum(radii)
        penalty = _calculate_penalties(circles_temp, penalty_weight_radii)
        return -sum_radii + penalty

    best_score = 0.0
    best_circles = None

    # Use dual_annealing for global optimization, then L-BFGS-B for refinement
    sa_penalty_weight = 1000 
    
    # Initialize circles using the helper function
    initial_circles = _initialize_circles(num_circles, initial_radius_factor=1.0, rows_factor=1.1)
    
    # The objective_function passed to dual_annealing needs to take a single array argument.
    def sa_objective_wrapper(x):
        return objective_function_sa(x, sa_penalty_weight)

    # Run dual_annealing with increased iterations and temperature for broader exploration
    # Use the initialized circles as a starting point for dual_annealing if possible,
    # though dual_annealing samples randomly within bounds.
    sa_result = dual_annealing(func=sa_objective_wrapper, bounds=bounds, maxiter=5000, initial_temp=15000, seed=42)
    
    x0_refined = sa_result.x
    
    # Introduce a small random perturbation to the SA result for L-BFGS-B
    # Scale perturbation by the average radius from SA result
    avg_radius_sa = np.mean(x0_refined.reshape(num_circles, 3)[:, 2])
    perturbation_scale = 1e-3 * avg_radius_sa 
    x0_refined_perturbed = x0_refined + np.random.uniform(-perturbation_scale, perturbation_scale, size=x0_refined.shape)
    
    # Now run L-BFGS-B with the higher penalty weight for precision and increased maxiter
    result_lbfgsb = minimize(objective_function, x0_refined_perturbed, method='L-BFGS-B', bounds=bounds, options={'disp': False, 'maxiter': 50000}) 
    
    current_optimized_circles = result_lbfgsb.x.reshape(num_circles, 3)

    # Iterative "Squash and Expand" refinement
    num_squash_expand_iterations = 5
    shrink_factor_per_iter = 0.995 # Shrink radii by 0.5% each iteration
    lbfgsb_penalty_base = 1000
    lbfgsb_maxiter_per_iter = 10000

    for iter_idx in range(num_squash_expand_iterations):
        # 1. Slightly shrink existing circles
        current_optimized_circles[:, 2] *= shrink_factor_per_iter
        
        # Ensure circles are still within bounds after shrinking
        for i in range(num_circles):
            x, y, r = current_optimized_circles[i]
            r = max(r, 1e-7) 
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            current_optimized_circles[i] = [x, y, r]

        # 2. Re-optimize all parameters (x, y, r) with L-BFGS-B from this slightly shrunk state
        # Increase penalty weight for stricter adherence as we refine
        current_lbfgsb_penalty = lbfgsb_penalty_base * (1 + iter_idx * 0.5) 

        result_iter_lbfgsb = minimize(objective_function, current_optimized_circles.flatten(), 
                                      args=(current_lbfgsb_penalty,), # Pass penalty weight as arg
                                      method='L-BFGS-B', bounds=bounds, options={'disp': False, 'maxiter': lbfgsb_maxiter_per_iter})
        current_optimized_circles = result_iter_lbfgsb.x.reshape(num_circles, 3)

    # Final clamping after all L-BFGS-B iterations
    for i in range(num_circles):
        x, y, r = current_optimized_circles[i]
        r = max(r, 1e-7) 
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        current_optimized_circles[i] = [x, y, r]

    # Final refinement of radii with fixed centers using BFGS
    fixed_centers = current_optimized_circles[:, :2] 
    radii_bounds = [(1e-7, 0.5) for _ in range(num_circles)]

    # Use a higher penalty weight for this final step to ensure no overlaps/bounds issues
    penalty_weight_radii = 2500 
    
    # Wrap radii-only objective for `minimize`
    def radii_objective_wrapper(radii_params):
        return objective_function_radii_only(radii_params, fixed_centers, penalty_weight_radii)

    result_bfgs_radii = minimize(radii_objective_wrapper, current_optimized_circles[:, 2], 
                                 method='L-BFGS-B', bounds=radii_bounds, options={'disp': False, 'maxiter': 20000})

    final_radii = np.maximum(result_bfgs_radii.x, 1e-7)
    best_circles = np.hstack((fixed_centers, final_radii.reshape(-1, 1)))

    # One final very slight shrinkage if any overlap or out-of-bounds still exists
    shrinkage_factor = 0.999
    max_attempts = 100
    attempts = 0
    while (check_overlap(best_circles, tolerance=1e-9) or 
           check_bounds(best_circles, tolerance=1e-9)) and attempts < max_attempts:
        best_circles[:, 2] *= shrinkage_factor
        for i in range(num_circles):
            x, y, r = best_circles[i]
            best_circles[i, 0] = np.clip(x, r, 1 - r)
            best_circles[i, 1] = np.clip(y, r, 1 - r)
        attempts += 1
        if np.any(best_circles[:, 2] < 1e-10): 
            break
    
    return best_circles
