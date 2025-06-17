# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# %%
root_dir = Path(__file__).parent.parent.parent.parent
program_db_path = root_dir / "data" / "flash.db"
program_db_path.exists()

# %%
from alfred_evolve.database.base import Program, Score
from alfred_evolve.database.sql import SQLDatabase

sql_db = SQLDatabase(f"sqlite:///{program_db_path}")

with sql_db.get_session() as session:
    n_islands = session.query(Program.island_id).distinct().count()
print(f"Number of islands in the database: {n_islands}")

initial_program_content = sql_db.get(
    Program, filter_by={"island_id": 0}, order_by="id"
).content
print(f"Initial program content:\n{initial_program_content}")

with sql_db.get_session() as session:
    from sqlalchemy.orm import aliased

    ParentProgram = aliased(Program)
    migration_ids = (
        session.query(Program.id)
        .join(ParentProgram, Program.parent)
        .filter(Program.island_id != ParentProgram.island_id)
        .all()
    )
print(f"Migration IDs: {migration_ids}")

for migration_id in migration_ids:
    program = sql_db.get(Program, filter_by={"id": migration_id[0]})
    scores = sql_db.get_n(Score, filter_by={"program_id": program.id}, order_by="name")
    print(
        f"Program ID: {program.id}, Island ID: {program.island_id}, Scores: {scores}, Parent ID: {program.parent_id}"
    )
    if program.parent_id:
        parent_program = sql_db.get(Program, filter_by={"id": program.parent_id})
        parent_scores = sql_db.get_n(
            Score, filter_by={"program_id": parent_program.id}, order_by="name"
        )
        print(
            f"Parent Program ID: {parent_program.id}, Island ID: {parent_program.island_id}, Scores: {parent_scores}"
        )

sql_db.close()

# %%
def plot_packing(packing):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    for x, y, r in packing:
        circle = plt.Circle((x, y), r, edgecolor="black", facecolor="none")
        ax.add_artist(circle)

    plt.title(f"Circle Packing (r = {packing[:, 2].sum():.6f})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.grid(False)
    plt.show()

# %%
packing = np.array(
    [
        [0.08492626295299072, 0.08492626295298925, 0.08492626175298813],
        [0.1051825607425316, 0.273952839895209, 0.1051825595425286],
        [0.8892209867422185, 0.8892209867422205, 0.11077901205778062],
        [0.29769047515361124, 0.13325857321090293, 0.13325857201089847],
        [0.5005716308602726, 0.9060726622352698, 0.09392733656472874],
        [0.7602894717601135, 0.7636735690669769, 0.06918067567421826],
        [0.8969394793820907, 0.48460080267039746, 0.10306051941790755],
        [0.8932098548995672, 0.27478328362107973, 0.10679014390043018],
        [0.9042676702078575, 0.6832585347531728, 0.09573232859214245],
        [0.5027155537926319, 0.07886037342133083, 0.07886037222132991],
        [0.7052539407047457, 0.38692355354529584, 0.11207708821576053],
        [0.10346723383379158, 0.482595582231428, 0.10346723263379044],
        [0.6868841898053353, 0.9076084479399097, 0.09239155086008782],
        [0.2973903965441072, 0.3816658445946462, 0.11514887942184346],
        [0.5013319244949968, 0.5299634197172355, 0.13701042935933302],
        [0.5966412162813498, 0.7424170492107359, 0.09584232503049345],
        [0.31405697823634754, 0.9074079045596762, 0.09259209424032307],
        [0.2730942859606913, 0.5960427017928825, 0.1006003670979893],
        [0.5044682393202817, 0.2753426170410285, 0.11762968730538587],
        [0.24064759875028952, 0.7629588633384494, 0.06944019302792762],
        [0.09615133453038453, 0.682080042714095, 0.09615133333038138],
        [0.7053905109717038, 0.13022110150895724, 0.13022110030895526],
        [0.9153604988057937, 0.08463950119420646, 0.0846394999942054],
        [0.40478026717453036, 0.7420494431701331, 0.09601897504302981],
        [0.7283701482399528, 0.5976347962705023, 0.09989834987287692],
        [0.11115617987705803, 0.8888438201229436, 0.11115617867705682],
    ]
)
plot_packing(packing)

# %%
import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from tqdm.auto import tqdm

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

    N_TRIALS = 1000 # Increased trials
    MAX_ITER_PER_TRIAL = 1000 # Increased iterations per trial

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

    for trial in tqdm(range(N_TRIALS)):
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
                print(f"New best score: {best_score:.9f} at trial {trial + 1}")
        
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

packing = pack_26()
plot_packing(packing)

# %%
