import numpy as np
from scipy.optimize import minimize

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

def _initialize_circles(num_circles: int) -> np.ndarray:
    """Initializes circles in a hexagonal packing arrangement."""
    circles = np.zeros((num_circles, 3))
    # Initial radius based on number of circles and area
    # A slightly larger initial radius might help fill space more aggressively
def _initialize_circles(num_circles: int, initial_radius_factor: float, rows_factor: float = 1.0) -> np.ndarray:
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
    
    # Define the objective function for scipy.optimize.minimize
    def objective_function(packed_params):
        circles = packed_params.reshape(num_circles, 3)
        
        circles[:, 2] = np.maximum(circles[:, 2], 1e-7)

        sum_radii = np.sum(circles[:, 2])
        
        penalty = 0.0
        
        centers = circles[:, :2]
        radii = circles[:, 2]

        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        dist = np.sqrt(dist_sq)

        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        overlap_mask = dist < min_dist - 1e-9 
        overlap_values = np.triu(min_dist - dist, k=1)
        
        penalty_weight = 750 # Slightly reduced penalty weight
        penalty += penalty_weight * np.sum(overlap_values[overlap_mask]**2)

        x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
        
        penalty_x_lower = np.maximum(0, r - x)**2
        penalty_x_upper = np.maximum(0, x + r - 1)**2
        penalty += penalty_weight * np.sum(penalty_x_lower + penalty_x_upper)

        penalty_y_lower = np.maximum(0, r - y)**2
        penalty_y_upper = np.maximum(0, y + r - 1)**2
        penalty += penalty_weight * np.sum(penalty_y_lower + penalty_y_upper)
                
        return -sum_radii + penalty

    # Define bounds for x, y, and r for each circle
    bounds = []
    for i in range(num_circles):
        bounds.append((0.0, 1.0)) 
        bounds.append((0.0, 1.0)) 
        bounds.append((1e-7, 0.5)) 
    
    best_score = 0.0
    best_circles = None

    # Try multiple initializations
    initialization_configs = [
        (0.4, 1.0),
        (0.45, 0.9),
        (0.35, 1.1),
        (0.42, 1.05),
        (0.38, 0.95),
        (0.41, 0.98), # New variations
        (0.39, 1.02),
        (0.43, 0.97),
        (0.37, 1.03)
    ]

    for factor, rows_factor in initialization_configs:
        initial_circles = _initialize_circles(num_circles, initial_radius_factor=factor, rows_factor=rows_factor)
        x0 = initial_circles.flatten()

        result = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds, options={'disp': False, 'maxiter': 20000}) # Increased maxiter
        
        current_optimized_circles = result.x.reshape(num_circles, 3)

        # Final clamping to ensure strict validity after optimization
        for i in range(num_circles):
            x, y, r = current_optimized_circles[i]
            r = max(r, 1e-7) 
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            current_optimized_circles[i] = [x, y, r]

        # One final very slight shrinkage if any overlap or out-of-bounds still exists
        shrinkage_factor = 0.999
        max_attempts = 100
        attempts = 0
        while (check_overlap(current_optimized_circles, tolerance=1e-9) or 
               check_bounds(current_optimized_circles, tolerance=1e-9)) and attempts < max_attempts:
            current_optimized_circles[:, 2] *= shrinkage_factor
            for i in range(num_circles):
                x, y, r = current_optimized_circles[i]
                current_optimized_circles[i, 0] = np.clip(x, r, 1 - r)
                current_optimized_circles[i, 1] = np.clip(y, r, 1 - r)
            attempts += 1
            if np.any(current_optimized_circles[:, 2] < 1e-10): 
                break
        
        current_score = score_packing(current_optimized_circles)
        if current_score > best_score:
            best_score = current_score
            best_circles = current_optimized_circles

    return best_circles
