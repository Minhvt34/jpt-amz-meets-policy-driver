import os
import sys


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
    """Test the LKH solver Python bindings."""
    print("Testing LKH solver Python bindings...")
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set paths to parameter and problem files
    # param_file = os.path.join(current_dir, "LKH-AMZ/LKH.par")
    # problem_file = os.path.join(current_dir, "LKH-AMZ/test.tsp")

    param_file = "/home/kaiz/jpt-amz-meets-policy-driver/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.par"
    problem_file = "/home/kaiz/jpt-amz-meets-policy-driver/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.ctsptw"
    
    # # Alternative paths (if you want to switch to amz0000 later)
    # param_file = "/home/kaiz/jpt-amz-meets-policy-driver/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.par"
    # problem_file = "/home/kaiz/jpt-amz-meets-policy-driver/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.ctsptw"
    
    # Check if the files exist
    if not os.path.exists(param_file):
        print(f"Error: Parameter file '{param_file}' not found.")
        return 1
    
    if not os.path.exists(problem_file):
        print(f"Error: Problem file '{problem_file}' not found.")
        return 1
    
    # Run the solver with detailed logging
    print(f"Using parameter file: {param_file}")
    print(f"Using problem file: {problem_file}")
    
    try:
        print("\n--- Testing Step-by-Step Initialization ---")
        # 1. Set parameter file name (Python wrapper)
        print("Setting parameter file path...")
        lkh_solver.read_parameter_file(param_file)
        # 2. Call LKH to read parameters
        print("Calling LKH_ReadParameters()...")
        lkh_solver.LKH_ReadParameters()
        
        # 3. Set problem file name (Python wrapper)
        print("\nSetting problem file path...")
        lkh_solver.read_problem_file(problem_file)
        # 4. Call LKH to read problem data
        print("Calling LKH_ReadProblem()...")
        lkh_solver.LKH_ReadProblem()
        
        # 5. Allocate structures
        print("\nAllocating structures...")
        lkh_solver.AllocateStructures()
        
        # Validate solver state
        print("\nValidating solver state after step-by-step init...")
        if not lkh_solver.validate_solver_state(True):
            print("Solver state validation FAILED after step-by-step init. Aborting this part.")
            # return 1 # Optionally abort here if this is critical
        else:
            print("Solver state validation PASSED after step-by-step init.")

        # 6. Create candidate set (using the safe wrapper which calls the explicit C++ one)
        print("\nCreating candidate set (step-by-step)...")
        if not lkh_solver.CreateCandidateSet():
            print("Failed to create candidate set (step-by-step), aborting.")
            # return 1 # Optionally abort
        else:
            print("Candidate set created successfully (step-by-step).")

        # 7. Initialize statistics
        print("\nInitializing statistics (step-by-step)...")
        lkh_solver.InitializeStatistics()
        print("Statistics initialized (step-by-step).")

        # Optionally, you could try running LinKernighan here, 
        # but solve_and_record_trajectory does a full run which is a better test.

        print("\n--- Freeing Structures Before Full Trajectory Test ---")
        lkh_solver.FreeStructures()
        print("LKH Structures Freed.")

        print("\n--- Testing Full Solver Trajectory ---")
        # Now run the full solver which handles its own internal initialization sequence
        print("Running solve_and_record_trajectory...")
        best_cost = lkh_solver.solve_and_record_trajectory(param_file, problem_file)
        
        if best_cost == sys.maxsize: # LLONG_MAX in C++ is often sys.maxsize in Python
            print("Solver failed with error (solve_and_record_trajectory).")
            return 1
        
        print(f"\nBest cost from solve_and_record_trajectory: {best_cost}")
        
        # Get the best tour
        print("\nGetting best tour (after solve_and_record_trajectory)...")
        try:
            best_tour = lkh_solver.get_best_tour()
            print(f"Best tour (first 10 nodes): {best_tour[:10]}")
            print(f"Tour length: {len(best_tour)}")
        except Exception as e:
            print(f"Error getting best tour: {e}")
        
        print("\nTest completed successfully.")
        return 0
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # You might want to print a traceback here for more detailed debugging
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 