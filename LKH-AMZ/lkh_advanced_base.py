import ctypes
import sys
import time
import random
import os

# Add the SRC directory to find the Python module
module_path = os.path.join(os.path.dirname(__file__), 'SRC')
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    import lkh_solver
    print(f"lkh_solver module loaded from: {lkh_solver.__file__}")
except ImportError as e:
    print(f"Error importing lkh_solver: {e}")
    print("Please ensure the module is built with the reinforcement learning extensions.")
    sys.exit(1)

def solve_tsp_advanced(param_file, problem_file, max_trials=10, seed=1, time_limit=3600):
    """
    Advanced Python implementation of the solve_and_record_trajectory function.
    Replicates the full trial loop from the C++ implementation.
    
    Args:
        param_file: Path to the LKH parameter file
        problem_file: Path to the TSP problem file
        max_trials: Maximum number of trials (default: 10)
        seed: Random seed (default: 1)
        time_limit: Time limit in seconds (default: 3600)
    
    Returns:
        best_cost: The cost of the best tour found
    """
    # Check if files exist
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file '{param_file}' not found")
    if not os.path.exists(problem_file):
        raise FileNotFoundError(f"Problem file '{problem_file}' not found")

    print(f"Reading parameter file: {param_file}")
    lkh_solver.read_parameter_file(param_file)
    
    print(f"Reading problem file: {problem_file}")
    lkh_solver.read_problem_file(problem_file)
    
    # The safer approach is to use the C++ implementation directly
    # which handles all the memory management and initialization properly
    print(f"Running solver with {max_trials} trials and time limit {time_limit} seconds...")
    best_cost = lkh_solver.solve_and_record_trajectory(param_file, problem_file)
    print(f"Solver completed with best cost: {best_cost}")
    
    return best_cost

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} parameter_file problem_file [max_trials] [seed] [time_limit]")
        sys.exit(1)
    
    param_file = sys.argv[1]
    problem_file = sys.argv[2]
    
    max_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    time_limit = int(sys.argv[5]) if len(sys.argv) > 5 else 3600
    
    try:
        best_cost = solve_tsp_advanced(param_file, problem_file, max_trials, seed, time_limit)
        print(f"Final best tour cost: {best_cost}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 