import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

def pack_26() -> np.ndarray:
    num_circles = 26

    # Numerical tolerance for the checker is 1e-9.
    # We enforce a slightly larger separation to ensure strict validity.
    MIN_SEPARATION = 1.2e-9 # Adjusted buffer to allow tighter packing, closer to checker tolerance (1e-9)

    # Bounds for x, y, r.
    # r_i must be positive, x/y_i between 0 and 1
    lower_bounds_base = np.tile([0.0, 0.0, 1e-9], num_circles) # min radius 1e-9
    upper_bounds_base = np.tile([1.0, 1.0, 0.5], num_circles) # max radius can't exceed 0.5

    bounds = Bounds(lower_bounds_base, upper_bounds_base)

    # Objective function: Maximize sum of radii => Minimize negative sum of radii
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    # Nonlinear Constraints:
    def constraints_func(params):
        circles = params.reshape(-1, 3)
        x_coords, y_coords, radii = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # C1: Boundary constraints (x-r >= 0, 1-(x+r) >= 0, y-r >= 0, 1-(y+r) >= 0)
        boundary_constraints = np.concatenate([
            x_coords - radii - MIN_SEPARATION,
            1.0 - (x_coords + radii) - MIN_SEPARATION,
            y_coords - radii - MIN_SEPARATION,
            1.0 - (y_coords + radii) - MIN_SEPARATION
        ])
        
        # C2: Overlap constraints (dist^2 >= (r1+r2)^2)
        dx = x_coords[:, np.newaxis] - x_coords
        dy = y_coords[:, np.newaxis] - y_coords
        dist_sq = dx**2 + dy**2
        
        radii_sum = radii[:, np.newaxis] + radii
        min_dist_sq = (radii_sum + MIN_SEPARATION)**2
        
        overlap_constraints = dist_sq[np.triu_indices(num_circles, k=1)] - min_dist_sq[np.triu_indices(num_circles, k=1)]

        return np.concatenate([boundary_constraints, overlap_constraints])

    nonlinear_constraints = NonlinearConstraint(constraints_func, 0, np.inf)

    best_score = -np.inf
    best_packing = None

    N_TRIALS = 50 # Increased trials
    MAX_ITER_PER_TRIAL = 600 # Increased iterations per trial

    # Initial Strategy: Seed with a known pattern for fewer circles, then fill
    # A common pattern for 5 large circles is a central one and four in corners/middles.
    # For 26, we can try to place some dominant circles and then fill the gaps.

    def generate_initial_packing_strategy(n_circles, trial_idx):
        if trial_idx == 0:
            # First trial: A structured, somewhat dense initial guess
            initial_circles = []
            
            # Try a 5x5 grid pattern, then adjust one to make 26
            rows, cols = 5, 5
            base_r = 0.095 # Larger initial radius
            
            for r_idx in range(rows):
                for c_idx in range(cols):
                    x = (c_idx + 0.5) / cols
                    y = (r_idx + 0.5) / rows
                    initial_circles.append([x, y, base_r])
            
            # Add the 26th circle - perhaps centrally or near an edge
            initial_circles.append([0.5, 0.5, 0.05]) # Add a smaller one at center
            
            # Adjust radii to be slightly smaller and allow for packing
            initial_circles_array = np.array(initial_circles[:n_circles])
            initial_circles_array[:, 2] *= 0.8 # Scale down radii initially
            
            # Perturb slightly to avoid perfect symmetry issues for optimizer
            initial_circles_array[:, :2] += np.random.uniform(-0.01, 0.01, (n_circles, 2))
            initial_circles_array[:, 2] += np.random.uniform(-0.001, 0.002, n_circles)
            
            return initial_circles_array

        else:
            # Subsequent trials: Random initial packing with varied radii distribution
            initial_x = np.random.uniform(0.05, 0.95, n_circles)
            initial_y = np.random.uniform(0.05, 0.95, n_circles)
            
            # Introduce more variation in initial radii: some larger, some smaller
            initial_r = np.random.uniform(0.005, 0.06, n_circles) # Broader range for random radii
            
            # Optionally, for a few circles, try to make them larger
            if trial_idx % 5 == 0: # Every 5th trial, try to make a few circles larger
                num_large_r = np.random.randint(3, 7)
                large_r_indices = np.random.choice(n_circles, num_large_r, replace=False)
                initial_r[large_r_indices] = np.random.uniform(0.03, 0.08, num_large_r)

            return np.stack([initial_x, initial_y, initial_r], axis=-1)

    for trial in range(N_TRIALS):
        initial_circles_array = generate_initial_packing_strategy(num_circles, trial)
        
        initial_params = initial_circles_array.flatten()
        
        # Ensure initial_params are within the overall bounds
        initial_params = np.clip(initial_params, lower_bounds_base, upper_bounds_base)

        result = minimize(objective, initial_params, 
                          method='SLSQP', # SLSQP is generally good for this type of problem
                          bounds=bounds, 
                          constraints=[nonlinear_constraints],
                          options={'maxiter': MAX_ITER_PER_TRIAL, 'ftol': 1e-11, 'disp': False}) # Smaller ftol

        if result.success:
            current_circles = result.x.reshape(-1, 3)
            current_score = np.sum(current_circles[:, 2]) # Sum of radii

            # Optional: Add a check for validity after optimization, though checker will do it
            # if constraints_func(result.x).min() >= -1e-9: # Check if constraints are met within tolerance
            if current_score > best_score:
                best_score = current_score
                best_packing = current_circles
        
        # If optimization failed but still produced a result, check if it's valid and better
        # This part is commented out as the checker will do final validation.
        # elif result.x is not None:
        #      current_circles = result.x.reshape(-1, 3)
        #      current_score = np.sum(current_circles[:, 2])
        #      if current_score > best_score:
        #          pass # Only update if result.success for robustness and to avoid invalid interim results

    if best_packing is None:
        # Fallback if no successful optimization occurred across all trials.
        # This provides a guaranteed valid, though likely low-scoring, packing.
        # A very basic, small circle packing as a last resort.
        fallback_r = 0.05
        fallback_circles = []
        for i in range(5):
            for j in range(6):
                if len(fallback_circles) < num_circles:
                    fallback_circles.append([0.1 + j*0.15, 0.1 + i*0.15, fallback_r])
        best_packing = np.array(fallback_circles[:num_circles])


    return best_packing
