import os
import sys
import numpy as np
import random
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

def main():
    # Define the parameter file path
    param_file = "/home/kaiz/jpt-amz-meets-policy-driver/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.par"
    problem_file = "/home/kaiz/jpt-amz-meets-policy-driver/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.ctsptw"

    # Run the solver and record the trajectory
    best_cost = lkh_solver.solve_and_record_trajectory(param_file, problem_file)
    print(f"Best cost: {best_cost}")
    
if __name__ == "__main__":
    main()
    