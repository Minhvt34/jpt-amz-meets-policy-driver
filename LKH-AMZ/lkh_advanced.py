import ctypes
import sys
import time
import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add the SRC directory to find the Python module
module_path = os.path.join(os.path.dirname(__file__), 'SRC')
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    import lkh_solver
    print(f"lkh_solver module loaded from: {lkh_solver.__file__}")
except ImportError as e:
    print(f"Error importing lkh_solver: {e}")
    print("Please ensure the module is built with the class-based design for multiprocessing.")
    sys.exit(1)

class LKHSolver:
    """
    A Python wrapper for the LKHSolver C++ class that provides safe multiprocessing support.
    Each instance maintains its own isolated state, preventing conflicts between processes.
    """
    def __init__(self, param_file=None, problem_file=None, seed=1):
        # Create a new C++ LKHSolver instance
        self.solver = lkh_solver.LKHSolver()
        self.seed = seed
        self.initialized = False
        
        # Set files if provided
        if param_file:
            self.set_parameter_file(param_file)
        if problem_file:
            self.set_problem_file(problem_file)
        
    def set_parameter_file(self, param_file):
        """Set the parameter file path."""
        if not os.path.exists(param_file):
            raise FileNotFoundError(f"Parameter file '{param_file}' not found")
        self.solver.set_parameter_file(param_file)
        
    def set_problem_file(self, problem_file):
        """Set the problem file path."""
        if not os.path.exists(problem_file):
            raise FileNotFoundError(f"Problem file '{problem_file}' not found")
        self.solver.set_problem_file(problem_file)
        
    def set_tour_file(self, tour_file):
        """Set the tour output file path."""
        self.solver.set_tour_file(tour_file)
        
    def set_pi_file(self, pi_file):
        """Set the pi file path."""
        self.solver.set_pi_file(pi_file)
        
    def set_initial_tour_file(self, initial_tour_file):
        """Set the initial tour file path."""
        self.solver.set_initial_tour_file(initial_tour_file)
        
    def initialize(self):
        """Initialize the solver by reading parameters and problem."""
        try:
            print("Reading parameters...")
            if not self.solver.read_parameters():
                raise RuntimeError("Failed to read parameters")
            
            print("Reading problem...")
            if not self.solver.read_problem():
                raise RuntimeError("Failed to read problem")
            
            print("Allocating structures...")
            if not self.solver.allocate_structures():
                raise RuntimeError("Failed to allocate structures")
            
            self.initialized = True
            print(f"Solver initialized successfully. Problem dimension: {self.solver.get_dimension()}")
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
            
    def solve_with_trajectory(self, max_trials=10, time_limit=3600):
        """
        Solve using the C++ implementation with trajectory recording.
        This is the recommended method for production use.
        """
        if not self.initialized and not self.initialize():
            raise RuntimeError("Failed to initialize the solver")
        
        print(f"Starting C++ solver with max_trials={max_trials}, time_limit={time_limit}, seed={self.seed}")
        
        # Use the C++ high-level solver interface
        try:
            return self.solver.solve_with_trajectory(max_trials, time_limit)
        except Exception as e:
            print(f"Error in C++ solve_with_trajectory: {e}")
            raise
    
    def solve_python_controlled(self, max_trials=10, time_limit=3600):
        """
        Python-controlled solving using granular C++ method calls.
        This provides more control but is more complex.
        """
        if not self.initialized and not self.initialize():
            raise RuntimeError("Failed to initialize the solver")
        
        try:
            print(f"Initializing solver run with seed: {self.seed}")
            self.solver.initialize_run_globals(self.seed)

            print("Creating candidate set...")
            if not self.solver.create_candidate_set():
                raise RuntimeError("Failed to create candidate set")
            
            print("Initializing statistics...")
            self.solver.initialize_statistics()
            
            print("Validating solver state...")
            if not self.solver.validate_solver_state(True):
                raise RuntimeError("Solver state validation failed")
            
            # Reset node tour fields
            self.solver.reset_node_tour_fields()

            # Initialize run-specific bests
            self.solver.set_better_cost(sys.maxsize)
            self.solver.set_better_penalty(sys.maxsize)

            if max_trials > 0:
                if self.solver.is_hashing_used():
                    print("Initializing hash table...")
                    self.solver.hash_initialize()
            else:
                print("MaxTrials is 0. Using special path.")
                self.solver.set_trial_number(1)
                if not self.solver.choose_initial_tour():
                    raise RuntimeError("ChooseInitialTour failed in MaxTrials=0 case")
                
                current_p = self.solver.calculate_penalty()
                self.solver.set_current_penalty(current_p)
                self.solver.set_better_penalty(current_p)

            print("Preparing initial kicking strategy...")
            if not self.solver.prepare_kicking():
                raise RuntimeError("Failed to prepare initial kicking strategy")
            
            start_time = time.time()
            print(f"Starting solver trials: 1 to {max_trials}")
            
            for trial_num in range(1, max_trials + 1):
                # Time limit check
                if trial_num > 1 and (time.time() - start_time) >= time_limit:
                    print("*** Time limit exceeded ***")
                    break
                
                self.solver.set_trial_number(trial_num)
                print(f"-- Trial {trial_num}/{max_trials} --")

                # Choose FirstNode at random
                self.solver.select_random_first_node()

                print(f"Choosing initial tour for trial {trial_num}...")
                if not self.solver.choose_initial_tour():
                    print(f"Warning: ChooseInitialTour failed for trial {trial_num}")
                    continue
                
                print(f"Running Lin-Kernighan for trial {trial_num}...")
                cost_after_lk = self.solver.lin_kernighan()

                if cost_after_lk == sys.maxsize:
                    print(f"Error in LinKernighan algorithm for trial {trial_num}")
                    continue
                
                # Get current penalty after LK
                current_penalty_val = self.solver.calculate_penalty()
                self.solver.set_current_penalty(current_penalty_val)
                
                # Check for improvement
                better_penalty_val = self.solver.get_better_penalty()
                better_cost_val = self.solver.get_better_cost()

                print(f"Trial {trial_num}: Cost={cost_after_lk}, Penalty={current_penalty_val}")
                print(f"Comparing with: BetterCost={better_cost_val}, BetterPenalty={better_penalty_val}")

                improved = False
                if current_penalty_val < better_penalty_val or \
                   (current_penalty_val == better_penalty_val and cost_after_lk < better_cost_val):
                    improved = True
                    print(f"Trial {trial_num}: Improvement found!")
                    
                    self.solver.set_better_cost(cost_after_lk)
                    self.solver.set_better_penalty(current_penalty_val)
                    
                    print("Recording better tour...")
                    if not self.solver.record_better_tour():
                        print(f"Warning: RecordBetterTour failed in trial {trial_num}")

                    print("Adjusting candidate set...")
                    if not self.solver.adjust_candidate_set():
                         print(f"Warning: AdjustCandidateSet failed in trial {trial_num}")

                    print("Preparing for next kick...")
                    if not self.solver.prepare_kicking():
                        print(f"Warning: PrepareKicking failed in trial {trial_num}")

                    if self.solver.is_hashing_used():
                        self.solver.hash_initialize()
                        self.solver.hash_insert(self.solver.get_lkh_hash(), cost_after_lk)

                if not improved:
                    print(f"Trial {trial_num}: No improvement")
            
            print("Finalizing tour structure...")
            self.solver.finalize_tour_from_best_suc()

            # Set final penalty
            final_run_penalty = self.solver.get_better_penalty()
            if final_run_penalty != sys.maxsize:
                self.solver.set_current_penalty(final_run_penalty)
            
            print("Recording best tour...")
            if not self.solver.record_best_tour():
                print("Warning: RecordBestTour failed")

            final_run_cost = self.solver.get_better_cost()
            print(f"Best cost for this run: {final_run_cost}")
            
            # Get best tour details
            try:
                best_tour_nodes = self.solver.get_best_tour()
                print(f"Best tour (first 10 nodes): {best_tour_nodes[:min(10, len(best_tour_nodes))]}")
                print(f"Overall best cost: {self.solver.get_best_cost()}")
            except Exception as e:
                print(f"Error getting best tour details: {e}")
            
            return final_run_cost
            
        except Exception as e:
            import traceback
            print(f"Error in Python-controlled solve: {e}")
            traceback.print_exc()
            raise
    
    def solve(self, max_trials=10, time_limit=3600, use_python_control=False):
        """
        Solve the TSP problem.
        
        Args:
            max_trials: Maximum number of trials
            time_limit: Time limit in seconds
            use_python_control: If True, use Python-controlled solving (more verbose)
                              If False, use C++ high-level interface (recommended)
            
        Returns:
            best_cost: The cost of the best tour found
        """
        if use_python_control:
            return self.solve_python_controlled(max_trials, time_limit)
        else:
            return self.solve_with_trajectory(max_trials, time_limit)
    
    def get_tour(self):
        """Get the best tour found by the solver."""
        return self.solver.get_best_tour()
        
    def get_cost(self):
        """Get the cost of the best tour."""
        return self.solver.get_best_cost()
        
    def get_dimension(self):
        """Get the dimension of the problem."""
        return self.solver.get_dimension()
        
    def validate_state(self, fix_issues=True):
        """Validate the solver's internal state."""
        return self.solver.validate_solver_state(fix_issues)

def solve_tsp_instance(args):
    """
    Worker function for multiprocessing.
    Each process gets its own isolated LKHSolver instance.
    """
    param_file, problem_file, max_trials, seed, time_limit, instance_id = args
    
    try:
        print(f"Process {instance_id}: Starting with seed {seed}")
        
        # Create a new solver instance for this process
        solver = LKHSolver(param_file, problem_file, seed)
        
        # Solve the problem
        best_cost = solver.solve(max_trials, time_limit)
        
        # Get the tour
        best_tour = solver.get_tour()
        
        print(f"Process {instance_id}: Completed with cost {best_cost}")
        
        return {
            'instance_id': instance_id,
            'seed': seed,
            'best_cost': best_cost,
            'best_tour': best_tour,
            'dimension': solver.get_dimension()
        }
        
    except Exception as e:
        print(f"Process {instance_id}: Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return {
            'instance_id': instance_id,
            'seed': seed,
            'error': str(e),
            'best_cost': float('inf')
        }

def solve_tsp_multiprocessing(param_file, problem_file, max_trials=10, time_limit=3600, 
                            num_processes=None, seeds=None):
    """
    Solve TSP using multiple processes with different seeds.
    Each process gets its own isolated solver instance.
    
    Args:
        param_file: Path to the parameter file
        problem_file: Path to the problem file
        max_trials: Maximum number of trials per process
        time_limit: Time limit in seconds per process
        num_processes: Number of processes to use (default: CPU count)
        seeds: List of seeds to use (default: range(num_processes))
    
    Returns:
        List of results from all processes
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    if seeds is None:
        seeds = list(range(1, num_processes + 1))
    
    if len(seeds) != num_processes:
        raise ValueError(f"Number of seeds ({len(seeds)}) must match number of processes ({num_processes})")
    
    print(f"Starting multiprocessing with {num_processes} processes")
    print(f"Seeds: {seeds}")
    
    # Prepare arguments for each process
    process_args = [
        (param_file, problem_file, max_trials, seed, time_limit, i)
        for i, seed in enumerate(seeds)
    ]
    
    results = []
    
    # Use ProcessPoolExecutor for better resource management
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(solve_tsp_instance, args): args 
            for args in process_args
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Process {result['instance_id']} completed: Cost = {result.get('best_cost', 'ERROR')}")
            except Exception as e:
                print(f"Process {args[5]} generated an exception: {e}")
                results.append({
                    'instance_id': args[5],
                    'seed': args[3],
                    'error': str(e),
                    'best_cost': float('inf')
                })
    
    # Sort results by instance_id for consistent ordering
    results.sort(key=lambda x: x['instance_id'])
    
    return results

def solve_tsp_advanced(param_file, problem_file, max_trials=10, seed=1, time_limit=3600):
    """
    Advanced Python implementation of the TSP solver.
    Uses the new class-based wrapper for safe operation.
    
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
        print(f"Usage: {sys.argv[0]} parameter_file problem_file [max_trials] [seed] [time_limit] [--multiprocess] [--num_processes]")
        print("Examples:")
        print(f"  {sys.argv[0]} params.par problem.tsp")
        print(f"  {sys.argv[0]} params.par problem.tsp 10 1 3600")
        print(f"  {sys.argv[0]} params.par problem.tsp 10 1 3600 --multiprocess --num_processes 4")
        sys.exit(1)
    
    param_file = sys.argv[1]
    problem_file = sys.argv[2]
    
    max_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    time_limit = int(sys.argv[5]) if len(sys.argv) > 5 else 3600
    
    use_multiprocessing = '--multiprocess' in sys.argv
    
    try:
        if use_multiprocessing:
            # Get number of processes
            num_processes = multiprocessing.cpu_count()
            if '--num_processes' in sys.argv:
                idx = sys.argv.index('--num_processes')
                if idx + 1 < len(sys.argv):
                    num_processes = int(sys.argv[idx + 1])
            
            print(f"Running multiprocessing solver with {num_processes} processes")
            results = solve_tsp_multiprocessing(
                param_file, problem_file, max_trials, time_limit, num_processes
            )
            
            # Print summary
            print("\n=== MULTIPROCESSING RESULTS ===")
            best_result = min(results, key=lambda x: x.get('best_cost', float('inf')))
            
            for result in results:
                status = "ERROR" if 'error' in result else "OK"
                cost = result.get('best_cost', 'N/A')
                print(f"Process {result['instance_id']} (seed {result['seed']}): {cost} [{status}]")
            
            print(f"\nBest result: Cost = {best_result.get('best_cost', 'N/A')} (Process {best_result['instance_id']}, Seed {best_result['seed']})")
            
        else:
            # Single process
            print(f"Running single-process solver")
            best_cost = solve_tsp_advanced(param_file, problem_file, max_trials, seed, time_limit)
            print(f"Final best tour cost: {best_cost}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 