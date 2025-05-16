import sys
import os

# Add the SRC directory to Python's path to find the .so file
# Assuming this script is in LKH-AMZ/ and the .so is in LKH-AMZ/SRC/
module_path = os.path.join(os.path.dirname(__file__), 'SRC')
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    import lkh_solver
except ImportError as e:
    print(f"Error importing lkh_solver: {e}")
    print("Please ensure 'lkh_solver.cpython-....so' is in the LKH-AMZ/SRC directory.")
    print(f"Current sys.path: {sys.path}")
    exit()

def main():
    # The .par file path provided by the user
    # This path is relative to the workspace root /home/minhvt/jpt-amz
    # So, if this script is run from LKH-AMZ/, the path needs to be adjusted
    # or an absolute path should be used.
    # For robustness, let's construct the path relative to the assumed workspace root.
    
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # par_file = "data-evaluation/model_apply_outputs/TSPLIB_1/amz3051.par" # Relative to workspace root
    # For the Python script, it's safer to define it relative to this script's location or use an absolute path.
    # Let's assume the user means it's relative to the workspace root.
    # The script is in LKH-AMZ, workspace is parent of LKH-AMZ.
    
    par_file_relative_to_workspace = "data-evaluation/model_apply_outputs/TSPLIB_1/amz3051.par"
    
    # Construct absolute path if current PWD is not workspace_root
    # The problem_file_path needs to be absolute or relative to where LKH will be "run" from (conceptually).
    # The C code ReadParameters will try to open this path.
    # It's best to give an absolute path to the C code.
    
    # The par file path should be workspace_root + par_file_relative_to_workspace
    # Workspace root is /home/minhvt/jpt-amz/
    # The path given by user is relative to /home/minhvt/jpt-amz
    
    # current_script_dir = os.path.dirname(os.path.abspath(__file__)) # LKH-AMZ
    # workspace_dir = os.path.dirname(current_script_dir) # /home/minhvt/jpt-amz

    # For the par file path itself, the LKH C code will interpret it.
    # Let's assume the CWD for the Python process will be LKH-AMZ/
    # So, the path from LKH-AMZ/ to the par file is ../data-evaluation/...
    
    # Path for the .par file.
    # The user specified: data-evaluation/model_apply_outputs/TSPLIB_1/amz3051.par
    # This path looks like it's from the workspace root.
    # When we run `python LKH-AMZ/test_lkh_module.py`, CWD might be /home/minhvt/jpt-amz
    # If the script is run from /home/minhvt/jpt-amz, then the path is fine.
    # If run from LKH-AMZ/, then it should be "../data-evaluation/..."

    # Let's assume the script will be run from workspace root for simplicity of path.
    # If not, adjust this path.
    param_file_path = "data-evaluation/model_apply_outputs/TSPLIB_1/amz3051.par"

    print(f"Attempting to solve using LKH with parameter file: {param_file_path}")
    
    try:
        # Check if par file exists before calling
        if not os.path.exists(param_file_path):
            print(f"ERROR: Parameter file not found at {os.path.abspath(param_file_path)}")
            print(f"Please ensure the path is correct. Current CWD: {os.getcwd()}")
            # Try to locate it from workspace root explicitly
            abs_param_file_path = os.path.join(workspace_root, par_file_relative_to_workspace)
            if os.path.exists(abs_param_file_path):
                print(f"Found it at: {abs_param_file_path}. Using this path.")
                param_file_path = abs_param_file_path
            else:
                print(f"Still not found at {abs_param_file_path}. Exiting.")
                return

        # Read the problem file path from the parameter file
        problem_file_path = None
        with open(param_file_path, 'r') as f:
            for line in f:
                if line.startswith('PROBLEM_FILE'):
                    problem_file_path = line.split('=')[1].strip()
                    break
        
        if not problem_file_path:
            print("ERROR: Could not find PROBLEM_FILE in parameter file")
            return
            
        print(f"Problem file path from parameter file: {problem_file_path}")
        
        # Check if problem file exists
        if not os.path.exists(problem_file_path):
            print(f"ERROR: Problem file not found at {problem_file_path}")
            return

        # Call the solve_tsp function with both problem and parameter files
        best_cost = lkh_solver.solve_tsp(problem_file_path, param_file_path)
        print(f"LKH solver finished.")
        print(f"Best cost found: {best_cost}")
        
        if best_cost == 0 or best_cost == LLONG_MAX_PLACEHOLDER: # LLONG_MAX might be returned on error/no solution
             print("Warning: Cost is 0 or max value, this might indicate an issue or no solution found.")

    except Exception as e:
        print(f"An error occurred while running lkh_solver.solve: {e}")

if __name__ == "__main__":
    # A placeholder for LLONG_MAX if needed for comparison, Python doesn't have it directly.
    # The C++ wrapper returns long long. Pybind11 converts it to Python int.
    # A very large number can be used as a proxy if needed for checks.
    LLONG_MAX_PLACEHOLDER = 9223372036854775807 
    main() 