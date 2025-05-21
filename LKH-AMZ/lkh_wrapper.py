import os
import sys
import numpy as np
import random
import time
from collections import deque

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

def solve_tsp(param_file, problem_file, seed=1, time_limit=None):
    """
    Python implementation of the solve_and_record_trajectory function.
    
    Args:
        param_file: Path to the LKH parameter file
        problem_file: Path to the TSP problem file
        seed: Random seed (default: 1)
        time_limit: Time limit in seconds (default: None, uses value from parameter file)
    
    Returns:
        best_cost: The cost of the best tour found
    """
    print(f"Reading parameter file: {param_file}")
    lkh_solver.read_parameter_file(param_file)
    
    print(f"Reading problem file: {problem_file}")
    lkh_solver.read_problem_file(problem_file)
    
    # Set up the solver structures
    lkh_solver.AllocateStructures()
    lkh_solver.CreateCandidateSet()
    lkh_solver.InitializeStatistics()
    
    # Run the optimization process
    start_time = time.time()
    
    # Choose initial tour
    lkh_solver.ChooseInitialTour()
    
    # Run the Lin-Kernighan algorithm
    best_cost = lkh_solver.LinKernighan()
    
    # Record the best tour
    lkh_solver.RecordBestTour()
    
    print(f"Best tour cost: {best_cost}")
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    
    return best_cost

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} parameter_file problem_file")
        sys.exit(1)
    
    param_file = sys.argv[1]
    problem_file = sys.argv[2]
    
    best_cost = solve_tsp(param_file, problem_file)
    print(f"Final best tour cost: {best_cost}") 