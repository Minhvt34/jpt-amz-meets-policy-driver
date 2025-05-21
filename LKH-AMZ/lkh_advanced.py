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

class LKHSolver:
    """
    A wrapper class for the LKH solver that ensures proper handling of shared variables.
    """
    def __init__(self, param_file, problem_file, seed=1):
        self.param_file = param_file
        self.problem_file = problem_file
        self.seed = seed
        self.initialized = False
        self.best_cost = float('inf')
        
    def initialize(self):
        """
        Initialize the solver by reading parameters and problem.
        """
        try:
            # Check files exist
            if not os.path.exists(self.param_file):
                raise FileNotFoundError(f"Parameter file '{self.param_file}' not found")
            if not os.path.exists(self.problem_file):
                raise FileNotFoundError(f"Problem file '{self.problem_file}' not found")
            
            # Set up file parameters (C++ bindings handle the memory management)

            print("\n--- Testing Step-by-Step Initialization ---")
            # 1. Set parameter file name (Python wrapper)
            print("Setting parameter file path...")
            lkh_solver.read_parameter_file(self.param_file)
            # 2. Call LKH to read parameters
            print("Calling LKH_ReadParameters()...")
            lkh_solver.LKH_ReadParameters()
            
            # 3. Set problem file name (Python wrapper)
            print("\nSetting problem file path...")
            lkh_solver.read_problem_file(self.problem_file)
            # 4. Call LKH to read problem data
            print("Calling LKH_ReadProblem()...")
            lkh_solver.LKH_ReadProblem()

            # print(f"Reading parameter file: {self.param_file}")
            # lkh_solver.read_parameter_file(self.param_file)
            
            # print(f"Reading problem file: {self.problem_file}")
            # lkh_solver.read_problem_file(self.problem_file)
            
            # Allocate memory structures
            print("Allocating structures...")
            lkh_solver.AllocateStructures()
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
            
    def solve_and_record_trajectory(self, max_trials=10, time_limit=3600):
        """
        Python implementation of the C++ solve_and_record_trajectory function.
        
        This function follows the same structure as the C++ equivalent,
        but calls the appropriate binding functions for each step.
        
        Args:
            max_trials: Maximum number of trials
            time_limit: Time limit in seconds
            
        Returns:
            best_cost: The cost of the best tour found
        """
        if not self.initialized and not self.initialize():
            raise RuntimeError("Failed to initialize the solver")
        
        # Initialize LKH global variables for the run (seed, BestCost, etc.)
        print(f"Initializing LKH run globals with seed: {self.seed}")
        lkh_solver.initialize_lkh_run_globals(self.seed) # Sets BestCost = LLONG_MAX, etc.

        # Fallback for direct C++ solver (can be un-commented for comparison or if Python path has issues)
        # print("Note: For a stable run, consider using the direct C++ solve_and_record_trajectory.")
        # return lkh_solver.solve_and_record_trajectory(self.param_file, self.problem_file)
        
        try:
            print("Creating candidate set using safe wrapper...")
            if not lkh_solver.CreateCandidateSet(): # Uses safe_create_candidate_set
                raise RuntimeError("Failed to create candidate set using safe_create_candidate_set")
            
            print("Initializing statistics...")
            lkh_solver.InitializeStatistics()
            
            # --- Python takes over FindTour's main loop logic --- 
            print("Python is now controlling the LKH trial loop.")

            # Reset node tour fields (OldPred, OldSuc, NextBestSuc, BestSuc = 0)
            lkh_solver.py_reset_node_tour_fields()

            # Initialize run-specific bests (BetterCost, BetterPenalty)
            # Using sys.maxsize as a proxy for LLONG_MAX; C++ bindings use actual LLONG_MAX.
            # initialize_lkh_run_globals already set BestCost, this is for *this run's* best *before* global update.
            lkh_solver.set_better_cost(sys.maxsize) 
            lkh_solver.set_better_penalty(sys.maxsize)
            # CurrentPenalty is usually set by ChooseInitialTour or Penalty() within the loop

            if max_trials > 0: # Corresponds to LKH's if (MaxTrials > 0)
                if lkh_solver.is_hashing_used():
                    print("Initializing LKH Hash Table (HTable)...")
                    lkh_solver.HashInitialize() # HTable is a global pointer in LKH, init once per run
            else: # Special LKH logic for MaxTrials == 0 (usually for debugging or simple cases)
                print("MaxTrials is 0. Following LKH's specific path for this scenario.")
                lkh_solver.set_trial_number(1) # Trial is 1 for this path
                print("Choosing initial tour for MaxTrials=0 case...")
                if not lkh_solver.ChooseInitialTour(): # This sets up an initial tour
                    raise RuntimeError("ChooseInitialTour failed in MaxTrials=0 case")
                
                # In LKH, for MaxTrials=0, CurrentPenalty and BetterPenalty are often set based on this initial tour's penalty.
                current_p = lkh_solver.py_calculate_penalty()
                lkh_solver.set_current_penalty(current_p)
                lkh_solver.set_better_penalty(current_p)
                # BetterCost would be the cost of this initial tour. LinKernighan is not run.
                # This path is less common for full optimization. We might need to get the cost associated 
                # with ChooseInitialTour if it were to be returned directly. The current structure 
                # will proceed to finalize_run_and_get_cost which expects LinKernighan to have run.
                # For simplicity, we'll let it run through, but acknowledge this edge case.
                print(f"MaxTrials=0: Initial tour chosen. Penalty: {current_p}. No LK trials will run.")

            print("Preparing initial kicking strategy...")
            if not lkh_solver.PrepareKicking(): # Initial kick setup before the main loop
                raise RuntimeError("Failed to prepare initial kicking strategy")
            
            start_time = time.time()
            print(f"Starting LKH trials loop: 1 to {max_trials}")
            for trial_num_py in range(1, max_trials + 1):
                # Time limit check (as in original C++ and previous Python versions)
                if trial_num_py > 1 and (time.time() - start_time) >= time_limit:
                    print("*** Time limit exceeded ***")
                    break
                
                lkh_solver.set_trial_number(trial_num_py) # Sync LKH's global Trial variable
                print(f"-- Trial {lkh_solver.get_trial_number()}/{max_trials} --")

                # Choose FirstNode at random (mimicking LKH's FindTour)
                lkh_solver.py_select_random_first_node()
                # first_node_id = lkh_solver.get_first_node_id()
                # print(f"Selected FirstNode ID: {first_node_id}")

                print(f"Choosing initial tour for trial {trial_num_py}...")
                if not lkh_solver.ChooseInitialTour(): # This modifies Suc, Pred, and potentially CurrentPenalty
                    print(f"Warning: ChooseInitialTour failed for trial {trial_num_py}. Skipping LKH run for this trial.")
                    continue # Or handle error more robustly
                
                # CurrentPenalty is updated by ChooseInitialTour or LinKernighan internally in LKH.
                # We fetch it after these calls.

                print(f"Running Lin-Kernighan for trial {trial_num_py}...")
                cost_after_lk = lkh_solver.LinKernighan() # Modifies Suc, Pred, and CurrentPenalty

                if cost_after_lk == sys.maxsize: # LLONG_MAX from C++ indicates an error in safe_lin_kernighan
                    print(f"Error in LinKernighan algorithm for trial {trial_num_py}. Skipping trial.")
                    continue
                
                # Fetch current penalty *after* LK, as LK might update it (e.g., via StoreTour -> Penalty)
                current_penalty_val = lkh_solver.get_current_penalty()
                
                # Python now implements the logic to check for improvement and update run bests
                better_penalty_val = lkh_solver.get_better_penalty()
                better_cost_val = lkh_solver.get_better_cost()

                print(f"Trial {trial_num_py}: Cost from LK = {cost_after_lk}, CurrentPenalty = {current_penalty_val}")
                print(f"Comparing with: BetterCost = {better_cost_val}, BetterPenalty = {better_penalty_val}")

                improved_this_trial = False
                if current_penalty_val < better_penalty_val or \
                   (current_penalty_val == better_penalty_val and cost_after_lk < better_cost_val):
                    improved_this_trial = True
                    print(f"Trial {trial_num_py}: Improvement found! OldBest: {better_cost_val}_{better_penalty_val}")
                    
                    lkh_solver.set_better_cost(cost_after_lk)
                    lkh_solver.set_better_penalty(current_penalty_val)
                    print(f"New run best: Cost={lkh_solver.get_better_cost()}, Penalty={lkh_solver.get_better_penalty()}")
                    
                    print("Recording better tour (updates BestSuc pointers from current Suc, fills BetterTour array)...")
                    if not lkh_solver.RecordBetterTour(): # Uses current Suc links, affects BestSuc
                        print(f"Warning: RecordBetterTour failed in trial {trial_num_py}.")
                        # This is a critical failure if it happens, may lead to incorrect final tour

                    print("Adjusting candidate set...")
                    if not lkh_solver.AdjustCandidateSet(): # Uses current Suc links
                         print(f"Warning: AdjustCandidateSet failed in trial {trial_num_py}.")

                    print("Preparing for next kick...")
                    if not lkh_solver.PrepareKicking(): # Uses current Suc links
                        print(f"Warning: PrepareKicking failed in trial {trial_num_py}.")

                    # Hashing logic: LKH's StoreTour (called by LinKernighan if it finds an improvement 
                    # *within its own scope*) handles HashInsert. Python doesn't need to call HashInsert directly here.
                    if lkh_solver.is_hashing_used():
                        # LKH Hash is updated internally by StoreTour if LK found its own new best. Python doesn't need to re-insert.
                        # print(f"LKH Hash is now: {lkh_solver.get_lkh_hash()}") 
                        pass # No explicit HashInsert from Python needed here

                else:
                    print(f"Trial {trial_num_py}: No improvement over current run best ({better_cost_val}_{better_penalty_val}).")
            
            # --- End of Python-controlled trial loop ---
            print("Python LKH trial loop finished.")

            # Finalize the tour structure (Suc from BestSuc) and update LKH global Hash
            print("Finalizing tour structure (Suc from BestSuc) and LKH Hash...")
            lkh_solver.py_finalize_tour_from_best_suc()

            # Set LKH's global CurrentPenalty to the best penalty found in THIS run for RecordBestTour.
            final_run_penalty = lkh_solver.get_better_penalty()
            lkh_solver.set_current_penalty(final_run_penalty if final_run_penalty != sys.maxsize else 0)
            # Cost for RecordBestTour will be BetterCost of this run (already set if improvements were made)
            # or the initial large value if no tour was found/improved.
            
            # Record the best tour of this run against LKH's overall BestCost/BestPenalty
            print("Calling RecordBestTour to update LKH's global best if this run was better...")
            if not lkh_solver.RecordBestTour(): # This updates LKH's *global* BestCost, BestPenalty, BestTour
                print("Warning: RecordBestTour failed after trials loop.")

            # The cost of THIS specific Python-controlled run is in lkh_solver.get_better_cost()
            final_run_cost = lkh_solver.get_better_cost()
            if final_run_cost == sys.maxsize: # If no tour was ever accepted as "better"
                print("Warning: No tour was found or improved during this run. Cost remains at max_size.")
                # Potentially return an error or a very high value to indicate failure.
                # For now, we return it as is, but this indicates an issue.

            print(f"Best cost for THIS Python-controlled run: {final_run_cost}")
            
            # Get and print the overall best tour (from LKH's global BestTour array, updated by RecordBestTour)
            try:
                best_tour_nodes = lkh_solver.get_best_tour()
                print(f"LKH's overall best tour (first 10 nodes): {best_tour_nodes[:min(10, len(best_tour_nodes))]}")
                print(f"LKH's overall best cost: {lkh_solver.get_best_cost()}") # Compare with final_run_cost
            except Exception as e:
                print(f"Error getting LKH's best tour details: {e}")
            
            return final_run_cost
            
        except Exception as e:
            import traceback
            print(f"Error in Python solve_and_record_trajectory (granular control): {e}")
            traceback.print_exc()
            print("Consider falling back to direct C++ lkh_solver.solve_and_record_trajectory for robustness if issues persist.")
            raise # Re-raise the exception to make it clear the Python path failed
    
    def solve(self, max_trials=10, time_limit=3600):
        """
        Solve the TSP problem.
        This is a simple wrapper around solve_and_record_trajectory.
        
        Args:
            max_trials: Maximum number of trials
            time_limit: Time limit in seconds
            
        Returns:
            best_cost: The cost of the best tour found
        """
        return self.solve_and_record_trajectory(max_trials, time_limit)
    
    def get_tour(self):
        """
        Get the best tour found by the solver.
        
        Returns:
            tour: List of node IDs in the best tour
        """
        return lkh_solver.get_best_tour()
        
    def get_cost(self):
        """
        Get the cost of the best tour.
        
        Returns:
            cost: Cost of the best tour
        """
        return lkh_solver.get_best_cost()
        
    def get_dimension(self):
        """
        Get the dimension of the problem.
        
        Returns:
            dimension: Number of nodes in the problem
        """
        return lkh_solver.get_dimension()
        
def solve_tsp_advanced(param_file, problem_file, max_trials=10, seed=1, time_limit=3600):
    """
    Advanced Python implementation of the TSP solver.
    Uses a wrapper approach to ensure proper handling of shared variables.
    
    Args:
        param_file: Path to the parameter file
        problem_file: Path to the problem file
        max_trials: Maximum number of trials (default: 10)
        seed: Random seed (default: 1)
        time_limit: Time limit in seconds (default: 3600)
    
    Returns:
        best_cost: The cost of the best tour found
    """
    solver = LKHSolver(param_file, problem_file, seed)
    return solver.solve(max_trials, time_limit)

# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} parameter_file problem_file [max_trials] [seed] [time_limit]")
        sys.exit(1)
    
    param_file = sys.argv[1]
    problem_file = sys.argv[2]
    
    # param_file = "/home/kaiz/jpt-amz-meets-policy-driver/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.par"
    # problem_file = "/home/kaiz/jpt-amz-meets-policy-driver/data-evaluation/model_apply_outputs/TSPLIB_1/amz0000.ctsptw"
    
    max_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    time_limit = int(sys.argv[5]) if len(sys.argv) > 5 else 3600
    
    try:
        best_cost = solve_tsp_advanced(param_file, problem_file, max_trials, seed, time_limit)
        print(f"Final best tour cost: {best_cost}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 