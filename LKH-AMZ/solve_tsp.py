#!/usr/bin/env python3
"""
Main script to solve TSP problems using the LKH-AMZ solver.
"""
import os
import sys
import argparse
import time
import traceback

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

# Import the advanced solver
from lkh_advanced import solve_tsp_advanced

def save_tour_to_file(filename, tour, cost):
    """
    Write the tour to a file in the LKH format.
    
    Args:
        filename: Path to the output file
        tour: List representing the tour
        cost: Cost of the tour
    """
    with open(filename, 'w') as f:
        f.write("NAME : Best tour found\n")
        f.write(f"COMMENT : Cost = {cost}\n")
        f.write(f"DIMENSION : {len(tour) - 1}\n")
        f.write("TOUR_SECTION\n")
        
        for node in tour:
            f.write(f"{node}\n")
        
        f.write("-1\n")
        f.write("EOF\n")

def main():
    parser = argparse.ArgumentParser(description='Solve TSP problems using LKH-AMZ')
    parser.add_argument('param_file', help='Path to the parameter file')
    parser.add_argument('problem_file', help='Path to the problem file')
    parser.add_argument('--advanced', action='store_true', help='Use the advanced solver with multiple trials')
    parser.add_argument('--max-trials', type=int, default=10, help='Maximum number of trials (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--time-limit', type=int, default=3600, help='Time limit in seconds (default: 3600)')
    parser.add_argument('--output', type=str, help='Output file for the tour')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.param_file):
        print(f"Error: Parameter file '{args.param_file}' does not exist")
        return 1
    
    if not os.path.exists(args.problem_file):
        print(f"Error: Problem file '{args.problem_file}' does not exist")
        return 1
    
    try:
        start_time = time.time()
        
        # Run the solver
        if args.advanced:
            print(f"Running advanced solver with {args.max_trials} trials...")
            best_cost = solve_tsp_advanced(
                args.param_file, 
                args.problem_file,
                max_trials=args.max_trials,
                seed=args.seed,
                time_limit=args.time_limit
            )
        else:
            print("Running basic solver...")
            best_cost = lkh_solver.solve_and_record_trajectory(args.param_file, args.problem_file)
        
        # Report results
        elapsed_time = time.time() - start_time
        print(f"\nSolving completed in {elapsed_time:.2f} seconds")
        print(f"Best tour cost: {best_cost}")
        
        # If output file is specified, save the tour
        if args.output:
            try:
                # Get the best tour
                best_tour = lkh_solver.get_best_tour()
                
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(args.output)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Write the tour to file
                save_tour_to_file(args.output, best_tour, best_cost)
                print(f"Best tour written to {args.output}")
                
                # Also print the first few nodes of the tour
                print(f"Tour (first 10 nodes): {best_tour[:min(10, len(best_tour))]}")
                
            except Exception as e:
                print(f"Warning: Could not save tour - {e}")
                if args.verbose:
                    traceback.print_exc()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 