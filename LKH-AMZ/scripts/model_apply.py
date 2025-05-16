import os
import shutil
import subprocess
from pathlib import Path
import datetime
import glob
import sys # Add sys for path modification
import json
import argparse # Added for command-line arguments

# --- Add LKH-AMZ directory to sys.path to find lkh_solver.so ---
# This assumes the script is in LKH-AMZ/scripts/
_script_dir_for_path = Path(__file__).resolve().parent
_lkh_amz_dir_for_path = _script_dir_for_path.parent
sys.path.insert(0, str(_lkh_amz_dir_for_path))
# Clean up temporary path variables if desired, though not strictly necessary
del _script_dir_for_path
del _lkh_amz_dir_for_path
# --- End path modification ---

# Attempt to import the LKH solver. 
# This requires lkh_solver.so to be in a place Python can find it (e.g., same directory, or site-packages)
# For development, we might need to adjust PYTHONPATH or ensure the script is run from a location
# where the .so file is discoverable (e.g., LKH-AMZ/ if lkh_solver.so is there)
try:
    import lkh_solver
except ImportError:
    print("Error: Could not import lkh_solver. Make sure lkh_solver.so is compiled and accessible.")
    print("You might need to run this script from the LKH-AMZ directory or adjust PYTHONPATH.")
    lkh_solver = None # Set to None so script can still be parsed if module not found initially

def run_subprocess(cmd, cwd=None, desc=None):
    command_str = ' '.join(map(str, cmd))
    if desc:
        print(f"Running {desc}: {command_str} (in {cwd or Path.cwd()})")
    else:
        print(f"Executing: {command_str} (in {cwd or Path.cwd()})")
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=cwd)
        if process.stdout:
            print(f"{desc or 'Subprocess'} stdout:\n{process.stdout}")
        if process.stderr:
            print(f"{desc or 'Subprocess'} stderr:\n{process.stderr}")
        return process.stdout.strip(), process.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running {''.join(cmd)}:")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: Command {cmd[0]} not found or script {cmd[1]} not executable.")
        raise

def main():
    parser = argparse.ArgumentParser(description="LKH Model Apply Script with configurable input/output directories.")
    parser.add_argument(
        "--input_dir", 
        type=Path,
        default=None, # Will be set relative to project_root_dir if None
        help="Directory for input files (new_route_data.json, etc.). Defaults to 'data/model_apply_inputs' under project root."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path,
        default=None, # Will be set relative to project_root_dir if None
        help="Directory for output files (proposed_sequences.json, TSPLIB files, etc.). Defaults to 'data/model_apply_outputs' under project root."
    )
    args = parser.parse_args()

    if not lkh_solver:
        return # Stop if lkh_solver couldn't be imported

    print(f"Script model_apply.py started at: {datetime.datetime.now()}")

    script_dir = Path(__file__).resolve().parent
    lkh_amz_dir = script_dir.parent
    project_root_dir = lkh_amz_dir.parent
    data_dir = project_root_dir / "data" # Base data directory

    # Set up input and output directories based on arguments or defaults
    model_apply_inputs_dir = args.input_dir if args.input_dir else data_dir / "model_apply_inputs"
    model_apply_outputs_dir = args.output_dir if args.output_dir else data_dir / "model_apply_outputs"

    print(f"Using input directory: {model_apply_inputs_dir.resolve()}")
    print(f"Using output directory: {model_apply_outputs_dir.resolve()}")

    # Path to model_build_outputs, assumed to be fixed relative to data_dir for now
    # This might need to become configurable if model_build also has evaluation modes
    model_build_outputs_dir = data_dir / "model_build_outputs" 

    # Output files and directories for this script (now relative to model_apply_outputs_dir)
    final_proposed_sequences_json = model_apply_outputs_dir / "proposed_sequences.json"
    runtime_json = model_apply_outputs_dir / "runtime.json"
    tsplib_dir_1 = model_apply_outputs_dir / "TSPLIB_1"
    tsplib_dir_2 = model_apply_outputs_dir / "TSPLIB_2"
    tsplib_dir_merged = model_apply_outputs_dir / "TSPLIB" # Output of merge (will contain copied .tour files)
    tours_dir_1 = model_apply_outputs_dir / "TOURS-TSPLIB_1"
    tours_dir_2 = model_apply_outputs_dir / "TOURS-TSPLIB_2"

    # Scripts from the project's scripts directory (fixed location)
    project_scripts_dir = project_root_dir / "scripts"
    manage_time_script = project_scripts_dir / "manage_running_time.py"
    build_tsplib_script = project_scripts_dir / "build_TSPLIB.py"
    tsplib2json_script = project_scripts_dir / "tsplib2json.py"

    # 1. Clean output directories
    print(f"Cleaning directory: {model_apply_outputs_dir}")
    if model_apply_outputs_dir.exists():
        shutil.rmtree(model_apply_outputs_dir)
    model_apply_outputs_dir.mkdir(parents=True, exist_ok=True)
    tsplib_dir_1.mkdir()
    tsplib_dir_2.mkdir()
    # tours_dir_1 and tours_dir_2 will be created by LKH if parameters are set, 
    # or we might need to create them if LKH outputs tours to current dir by default.

    # 2. Start time management
    print("Starting time management...")
    # Ensure input files for manage_running_time are resolved correctly
    new_route_data_for_time_mgt = model_apply_inputs_dir / "new_route_data.json"
    if not new_route_data_for_time_mgt.exists():
        print(f"Error: new_route_data.json not found at {new_route_data_for_time_mgt} for time management.")
        return # Critical input missing

    cmd_start_time = [
        "python3", str(manage_time_script),
        "-start_time", 
        "--time_file", str(runtime_json),
        "--r_file", str(new_route_data_for_time_mgt)
    ]
    run_subprocess(cmd_start_time, cwd=project_root_dir, desc="manage_running_time (start)")

    # --- ML INTEGRATION POINT: Load Model ---
    # model_json_path is from model_build_outputs, which is currently fixed
    model_json_path = model_build_outputs_dir / "model.json"
    historical_route_data_path = model_build_outputs_dir / "route_data.json" 
    print("Placeholder for ML model loading and pre-processing.")
    # --- END ML INTEGRATION POINT ---

    # Input files for build_TSPLIB, resolved from model_apply_inputs_dir
    input_new_route_data = model_apply_inputs_dir / "new_route_data.json"
    input_new_travel_times = model_apply_inputs_dir / "new_travel_times.json"
    input_new_package_data = model_apply_inputs_dir / "new_package_data.json"

    # Check existence of critical input files for build_TSPLIB
    for f_path in [input_new_route_data, input_new_travel_times, input_new_package_data, model_json_path, historical_route_data_path]:
        if not f_path.exists():
            print(f"Error: Critical input file {f_path} for build_TSPLIB not found.")
            return

    # 3. Build TSPLIB_1 instances
    print(f"Creating TSPLIB1 instances to {tsplib_dir_1}")
    cmd_build_tsplib1 = [
        "python3", str(build_tsplib_script),
        "--r_file", str(input_new_route_data),
        "--t_file", str(input_new_travel_times),
        "--p_file", str(input_new_package_data),
        "--z_file", str(model_json_path), # From model_build phase (fixed path)
        "--br_file", str(historical_route_data_path), # From model_build phase (fixed path)
        "-noPrune", "-noPruneFailed", "-zoneNeighborTrans", "-superNeighborTrans",
        str(tsplib_dir_1) # Output directory for TSPLIB_1 files
    ]
    run_subprocess(cmd_build_tsplib1, cwd=project_root_dir, desc="build_TSPLIB (1)")

    # 4. Build TSPLIB_2 instances
    print(f"Creating TSPLIB2 instances to {tsplib_dir_2}")
    cmd_build_tsplib2 = [
        "python3", str(build_tsplib_script),
        "--r_file", str(input_new_route_data),
        "--t_file", str(input_new_travel_times),
        "--p_file", str(input_new_package_data),
        "--z_file", str(model_json_path),
        "--br_file", str(historical_route_data_path),
        "-noPrune", "-noPruneFailed", "-zoneNeighborTrans", "-superNeighborTrans",
        "-superPred", "-trans",
        str(tsplib_dir_2) # Output directory for TSPLIB_2 files
    ]
    run_subprocess(cmd_build_tsplib2, cwd=project_root_dir, desc="build_TSPLIB (2)")

    # 5. Get time limits for LKH runs
    print("Determining LKH time limits...")
    cmd_get_time_limits = [
        "python3", str(manage_time_script),
        "--time_file", str(runtime_json),
        "--r_file", str(new_route_data_for_time_mgt) # Use the already checked path
    ]
    output, _ = run_subprocess(cmd_get_time_limits, cwd=project_root_dir, desc="manage_running_time (get_limits)")
    try:
        time_limit_1, time_limit_2 = map(int, output.split()[:2])
    except ValueError:
        print(f"Warning: Could not parse time limits from output: '{output}'. Using defaults.")
        time_limit_1, time_limit_2 = 27, 14 # Defaults from .sh script
    print(f"Running with LKH time limits: TSPLIB_1={time_limit_1}s, TSPLIB_2={time_limit_2}s")

    # 6. Copy C binaries needed for solve/merge to model_apply_outputs_dir (the main output dir for this run)
    # Source of binaries is model_build_outputs/bin (fixed path)
    binaries_to_copy = ["merge", "get_Length", "score", "LKH"]
    source_bin_dir = model_build_outputs_dir / "bin"
    print(f"Copying required C binaries from {source_bin_dir} to {model_apply_outputs_dir}")
    for bin_name in binaries_to_copy:
        source_file = source_bin_dir / bin_name
        dest_file = model_apply_outputs_dir / bin_name # Copy to root of current output dir
        if source_file.exists():
            shutil.copy(source_file, dest_file)
        else:
            print(f"Warning: Binary {bin_name} not found at {source_file} for copying.")
    
    # --- LKH Solving with Python Module --- (Original script used subprocess for LKH executable)
    lkh_problem_dirs_params = [
        (tsplib_dir_1, tours_dir_1, time_limit_1, "TSPLIB_1"),
        (tsplib_dir_2, tours_dir_2, time_limit_2, "TSPLIB_2")
    ]

    # Ensure LKH output tour directories exist
    tours_dir_1.mkdir(parents=True, exist_ok=True)
    tours_dir_2.mkdir(parents=True, exist_ok=True)

    # Call LKH for each sub-directory of problems
    print("Solving TSP instances using LKH executable via subprocess...")
    
    lkh_executable_path = model_apply_outputs_dir / "LKH" # LKH copied to current output_dir root
    if not lkh_executable_path.is_file():
        print(f"Error: LKH executable not found at {lkh_executable_path} after attempting to copy.")
        # return # Or raise an exception if this is critical

    for problem_dir, tour_output_dir, time_limit, desc_suffix in lkh_problem_dirs_params:
        print(f"Processing instances in {problem_dir} (output to {tour_output_dir}, time: {time_limit}s approx)")
        problem_dir_path = Path(problem_dir)
        tour_output_dir_path = Path(tour_output_dir)

        os.makedirs(tour_output_dir_path, exist_ok=True)

        tsp_files = glob.glob(os.path.join(problem_dir_path, "*.ctsptw"))
        if not tsp_files:
            print(f"Warning: No .ctsptw files found in {problem_dir_path}")
            continue

        common_par_settings = (
            "MAX_TRIALS = 10\n"
            "RUNS = 1\n"
            "SEED = 1\n"
            "TRACE_LEVEL = 1\n"
            "INITIAL_PERIOD = 100\n"
            "SUBGRADIENT = YES\n"
        )

        for tsp_file_path_str in tsp_files:
            tsp_file_path = Path(tsp_file_path_str)
            par_file_name = tsp_file_path.stem + ".par"
            par_file_path = problem_dir_path / par_file_name # Par file in same dir as problem
            tour_file_name = tsp_file_path.stem + ".tour"
            tour_file_full_path = tour_output_dir_path / tour_file_name

            par_content = (
                f"PROBLEM_FILE = {tsp_file_path.resolve()}\n"
                f"TOUR_FILE = {tour_file_full_path.resolve()}\n"
                f"TIME_LIMIT = {time_limit}\n"
                f"{common_par_settings}"
            )
            # --- UNCOMMENTED AND ADDED DIAGNOSTICS ---
            print(f"  LKH Pre-flight Check for: {tsp_file_path.name}")
            print(f"    LKH Executable: {lkh_executable_path.resolve()}")
            print(f"    Problem File: {tsp_file_path.resolve()}")
            print(f"    Parameter File to be written: {par_file_path.resolve()}")
            print(f"    Expected Tour File: {tour_file_full_path.resolve()}")
            print(f"    PAR File Content:\n{par_content}")

            try:
                with open(par_file_path, 'w') as f_par:
                    f_par.write(par_content)
                print(f"    Successfully generated PAR file: {par_file_path.name}")
            except IOError as e:
                print(f"    Error writing PAR file {par_file_path}: {e}")
                continue

            print(f"  Attempting to run LKH for: {tsp_file_path.name}")
            
            if not lkh_executable_path.is_file(): 
                print(f"    CRITICAL ERROR: LKH executable not found at {lkh_executable_path}. Skipping LKH run for this file.")
                continue

            try:
                cmd = [str(lkh_executable_path.resolve()), str(par_file_path.resolve())] 
                print(f"    Executing LKH command: {' '.join(cmd)} (CWD: {model_apply_outputs_dir.resolve()})")
                
                process = subprocess.run(
                    cmd,
                    cwd=model_apply_outputs_dir.resolve(), 
                    capture_output=True,
                    text=True,
                    check=False 
                )
                
                print(f"    LKH Return Code for {tsp_file_path.name}: {process.returncode}")
                if process.stdout:
                    print(f"    LKH STDOUT for {tsp_file_path.name}:\n{process.stdout}")
                if process.stderr:
                    print(f"    LKH STDERR for {tsp_file_path.name}:\n{process.stderr}")

                if process.returncode == 0:
                    print(f"    LKH completed successfully for {tsp_file_path.name}.")
                    if tour_file_full_path.is_file():
                        print(f"    SUCCESS: Tour file found at {tour_file_full_path}")
                    else:
                        print(f"    WARNING: LKH reported success but tour file {tour_file_full_path} was NOT found!")
                else:
                    print(f"    Error running LKH for {tsp_file_path.name}.")
            except FileNotFoundError: # Should be caught by pre-check, but good to have
                 print(f"    Error: LKH executable or PAR file not found when trying to run. LKH: {lkh_executable_path}, PAR: {par_file_path}")                 
            except Exception as e:
                print(f"    An unexpected error occurred while running LKH for {tsp_file_path.name}: {e}")
            # --- END UNCOMMENT AND DIAGNOSTICS ---

    # 7. Merge LKH solutions by copying .tour files
    print("Consolidating LKH tour files...")
    tsplib_dir_merged.mkdir(parents=True, exist_ok=True) # Target for copied .tour files

    tour_files_copied_count = 0
    for source_tour_dir in [tours_dir_1, tours_dir_2]:
        print(f"  Copying tour files from {source_tour_dir} to {tsplib_dir_merged}")
        if not source_tour_dir.is_dir():
            print(f"    Warning: Source tour directory {source_tour_dir} not found. Skipping.")
            continue
        
        for tour_file in glob.glob(os.path.join(source_tour_dir, "*.tour")):
            try:
                shutil.copy(tour_file, tsplib_dir_merged)
                tour_files_copied_count += 1
            except Exception as e:
                print(f"    Error copying tour file {tour_file} to {tsplib_dir_merged}: {e}")
    
    if tour_files_copied_count > 0:
        print(f"  Successfully copied {tour_files_copied_count} .tour files to {tsplib_dir_merged}.")
    else:
        print(f"  Warning: No .tour files were found in {tours_dir_1} or {tours_dir_2} to copy.")

    # 8. Convert merged TSPLIB tours to JSON output
    print(f"Converting merged tours to JSON: {final_proposed_sequences_json}")
    # tsp2amz.json is expected to be in TSPLIB_1 (created by build_TSPLIB.py in current output_dir)
    tsp2amz_json_path = tsplib_dir_1 / "tsp2amz.json"
    if not tsp2amz_json_path.exists():
        print(f"Error: tsp2amz.json not found at {tsp2amz_json_path}. Cannot convert tours to JSON.")
    else:
        cmd_tsplib2json = [
            "python3", str(tsplib2json_script),
            "--tour_dir", str(tsplib_dir_merged), 
            "--tsplib2amz_file", str(tsp2amz_json_path),
            "--out_json", str(final_proposed_sequences_json)
        ]
        run_subprocess(cmd_tsplib2json, cwd=project_root_dir, desc="tsplib2json conversion")

    print(f"Script model_apply.py finished at: {datetime.datetime.now()}")
    print(f"Proposed sequences should be in: {final_proposed_sequences_json}")

    # --- BEGIN DEBUG READ TSP2AMZ (END OF SCRIPT) ---
    # print("\\\\nDEBUG: Reading TSPLIB_1/tsp2amz.json at the very end of model_apply.py:")
    # tsp2amz_path_debug_end = tsplib_dir_1 / \"tsp2amz.json\"
    # if tsp2amz_path_debug_end.exists():
    #     try:
    #         with open(tsp2amz_path_debug_end, 'r') as f_debug_end:
    #             data_debug_end = json.load(f_debug_end)
    #             if isinstance(data_debug_end, dict) and data_debug_end:
    #                 first_key_end = next(iter(data_debug_end))
    #                 print(f\"  Content of first key ('{first_key_end}') in {tsp2amz_path_debug_end}:\")
    #                 print(f\"  {json.dumps(data_debug_end[first_key_end], indent=2)}\")
    #             else:
    #                 print(f\"  {tsp2amz_path_debug_end} is empty or not a dictionary.\")
    #     except Exception as e_debug_end:
    #         print(f\"  Error reading or parsing {tsp2amz_path_debug_end}: {e_debug_end}\")
    # else:
    #     print(f\"  {tsp2amz_path_debug_end} does not exist at this point.\")
    # print(\"--- END DEBUG READ TSP2AMZ (END OF SCRIPT) ---\\\\n\")
    # --- END DEBUG ---

if __name__ == "__main__":
    main() 