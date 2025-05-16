import argparse
import datetime
import json
import shutil
from pathlib import Path
import subprocess
import re # Added for parsing C scorer output

def load_json_file(file_path, description):
    """Helper function to load a JSON file."""
    print(f"Loading {description} from: {file_path}")
    if not file_path.exists():
        print(f"Error: {description} file not found at {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {description}.")
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None

def save_json_file(data, file_path, description):
    """Helper function to save data to a JSON file."""
    print(f"Saving {description} to: {file_path}")
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved {description}.")
    except Exception as e:
        print(f"An unexpected error occurred while saving {file_path}: {e}")
        return None

def generate_tour_files_from_json(sequences_data, tsp2amz_data, output_dir, sequence_key_name):
    """
    Generates .tour files from JSON sequence data.
    - sequences_data: Loaded JSON data for proposed or actual sequences.
    - tsp2amz_data: Loaded JSON data from tsp2amz.json for ID mapping.
    - output_dir: Directory to save the .tour files.
    - sequence_key_name: Key in sequences_data (e.g., "proposed_sequence", "actual_sequence").
    Returns the count of .tour files successfully generated.
    """
    routes_dict_to_iterate = None
    if isinstance(sequences_data, dict):
        if "sequences" in sequences_data and isinstance(sequences_data["sequences"], dict):
            routes_dict_to_iterate = sequences_data["sequences"]
        else:
            # If "sequences" key is not there, or not a dict, assume the main dict is the routes
            routes_dict_to_iterate = sequences_data 
            print(f"Info: Using the root of the JSON data for routes in {output_dir} (no 'sequences' sub-key found or it was not a dict).")
    
    if not routes_dict_to_iterate: # Check if we have a valid dictionary of routes
        print(f"Warning: No valid route dictionary found in data for {output_dir}. Cannot generate tour files.")
        return 0

    if not tsp2amz_data:
        print(f"Warning: tsp2amz_data is missing. Cannot generate tour files for {output_dir}.")
        return 0

    generated_files_count = 0
    # Ensure routes_dict_to_iterate is not empty before iterating
    if not routes_dict_to_iterate:
         print(f"Warning: Route dictionary for {output_dir} is empty. No tour files will be generated.")
         return 0

    # --- BEGIN ADDED DEBUG PRINTS ---
    print(f"Debug: RouteIDs in sequence data for {output_dir} (first 5): {list(routes_dict_to_iterate.keys())[:5]}")
    print(f"Debug: Total RouteIDs in sequence data for {output_dir}: {len(routes_dict_to_iterate.keys())}")
    print(f"Debug: RouteIDs in tsp2amz_data (first 5): {list(tsp2amz_data.keys())[:5]}")
    print(f"Debug: Total RouteIDs in tsp2amz_data: {len(tsp2amz_data.keys())}")

    sequence_keys = set(routes_dict_to_iterate.keys())
    tsp2amz_keys = set(tsp2amz_data.keys())

    keys_in_sequence_not_in_tsp2amz = sequence_keys - tsp2amz_keys
    keys_in_tsp2amz_not_in_sequence = tsp2amz_keys - sequence_keys

    if keys_in_sequence_not_in_tsp2amz:
        print(f"Debug: Keys in sequence data for {output_dir} BUT NOT in tsp2amz_data (first 5): {list(keys_in_sequence_not_in_tsp2amz)[:5]}")
        print(f"Debug: Total count of such keys: {len(keys_in_sequence_not_in_tsp2amz)}")
    if keys_in_tsp2amz_not_in_sequence:
        print(f"Debug: Keys in tsp2amz_data BUT NOT in sequence data for {output_dir} (first 5): {list(keys_in_tsp2amz_not_in_sequence)[:5]}")
        print(f"Debug: Total count of such keys: {len(keys_in_tsp2amz_not_in_sequence)}")
    # --- END ADDED DEBUG PRINTS ---

    for route_id, sequence_info in routes_dict_to_iterate.items():
        if not isinstance(sequence_info, dict): # Add check if sequence_info is a dict
            print(f"Warning: RouteID {route_id} data is not a dictionary. Skipping.")
            continue

        if route_id not in tsp2amz_data:
            print(f"Warning: RouteID {route_id} not found in tsp2amz mapping. Skipping tour file generation.")
            continue

        route_tsp_info = tsp2amz_data[route_id]
        node_map = route_tsp_info.get("node_map")
        depot_node_id = route_tsp_info.get("depot_node_id")
        dimension = route_tsp_info.get("dimension")

        if node_map is None or depot_node_id is None or dimension is None:
            print(f"Warning: Incomplete TSP metadata for RouteID {route_id} in the provided tsp2amz mapping. Skipping tour file generation for this route.")
            if route_tsp_info: # Should exist if we got here based on current logic
                 print(f"  Available metadata keys for this RouteID: {list(route_tsp_info.keys())}")
            else: # Defensive, should not happen if route_id was in tsp2amz_data
                 print(f"  Error: route_tsp_info is unexpectedly None for RouteID {route_id}")

            missing_keys_list = []
            if node_map is None: missing_keys_list.append("'node_map'")
            if depot_node_id is None: missing_keys_list.append("'depot_node_id'")
            if dimension is None: missing_keys_list.append("'dimension'")
            print(f"  Missing required keys: {', '.join(missing_keys_list)}")
            continue

        stop_id_sequence = sequence_info.get(sequence_key_name)
        if not stop_id_sequence:
            print(f"Warning: No '{sequence_key_name}' found for RouteID {route_id}. Skipping tour file generation.")
            continue

        # Build the numerical sequence of nodes for the TOUR_SECTION
        # This sequence should be what LKH would output: depot, customer1, customer2, ...
        final_tour_nodes = [depot_node_id] # Start with the depot
        valid_sequence = True
        for stop_id in stop_id_sequence: # These are customer stop IDs
            numerical_node_id = node_map.get(stop_id)
            if numerical_node_id is None:
                print(f"Warning: StopID '{stop_id}' in RouteID '{route_id}' not found in node_map. Skipping this route.")
                valid_sequence = False
                break
            if numerical_node_id != depot_node_id: # Add only if it's not the depot itself
                final_tour_nodes.append(numerical_node_id)
        
        if not valid_sequence:
            continue
        
        if len(final_tour_nodes) != dimension:
            print(f"Warning: RouteID {route_id} - Number of nodes in generated tour ({len(final_tour_nodes)}) does not match problem dimension ({dimension}). Skipping tour file generation for this route.")
            continue
        
        # The DIMENSION in the .tour file should match the dimension from tsp2amz_data,
        # which build_TSPLIB.py calculates as 1 (depot) + number of unique packages.
        # The final_tour_nodes list represents the sequence of *all* nodes visited, starting with depot.
        # The number of items in final_tour_nodes should align with `dimension` if all stops are unique and distinct from depot.
        # If dimension represents unique nodes, and final_tour_nodes can have repeats (though not typical for LKH output for simple TSP),
        # this needs care. For now, assume `dimension` from `tsp2amz_data` is correct for the .tour file header.

        if generated_files_count == 0: # Print only for the first tour file being generated in this call
            print(f"Debug Route: {route_id}")
            print(f"  Dimension from tsp2amz: {dimension}")
            print(f"  Final tour nodes (count: {len(final_tour_nodes)}): {final_tour_nodes[:20]}..." # Print first 20 nodes
                  f"{final_tour_nodes[-5:] if len(final_tour_nodes) > 20 else ''}") # and last 5

        tour_content = []
        tour_content.append(f"NAME : {route_id}.tour")
        tour_content.append(f"TYPE : TOUR")
        tour_content.append(f"DIMENSION : {dimension}")
        tour_content.append("TOUR_SECTION")
        for node_id_in_seq in final_tour_nodes:
            tour_content.append(str(node_id_in_seq))
        tour_content.append("-1")
        tour_content.append("EOF")

        tour_file_path = output_dir / f"{route_id}.tour"
        try:
            with open(tour_file_path, 'w') as f:
                f.write("\n".join(tour_content) + "\n")
            generated_files_count += 1
        except IOError as e:
            print(f"Error writing tour file {tour_file_path}: {e}")
            
    print(f"Generated {generated_files_count} .tour files in {output_dir}")
    return generated_files_count

def run_c_scorer(score_binary_path, scoring_workspace_path):
    """
    Runs the C score binary.
    - score_binary_path: Path to the compiled C 'score' executable.
    - scoring_workspace_path: Path to the workspace directory. CWD will be set to this.
      This directory should contain proposed .tour files and an ACTUAL_TOURS subdirectory.
    Returns (stdout, stderr) from the C scorer process.
    """
    # The C score binary expects the directory of proposed tours as its argument.
    # It will look for "./ACTUAL_TOURS/" relative to its CWD.
    # So, if CWD is scoring_workspace_path, the argument should be "."
    cmd_for_scorer = [str(score_binary_path.resolve()), "."] 
    
    print(f"Running C scorer: {' '.join(cmd_for_scorer)} in CWD: {scoring_workspace_path.resolve()}")
    try:
        process = subprocess.run(
            cmd_for_scorer,
            cwd=scoring_workspace_path.resolve(),
            capture_output=True,
            text=True,
            check=False # We check returncode manually
        )
        
        # Always print stdout/stderr for transparency during execution
        print(f"C scorer STDOUT:\n{process.stdout}")
        if process.stderr:
            print(f"C scorer STDERR:\n{process.stderr}")
        
        if process.returncode != 0:
            print(f"Warning: C scorer exited with code {process.returncode}.")
            # stderr content is already printed above.

        return process.stdout, process.stderr
    except FileNotFoundError:
        print(f"Error: C scorer binary not found at {score_binary_path} when trying to run.")
        return None, f"C scorer binary not found at {score_binary_path}."
    except Exception as e:
        print(f"An unexpected error occurred while running C scorer: {e}")
        return None, str(e)

def parse_c_score_output(stdout_str, stderr_str):
    """
    Parses the stdout from the C scorer to extract scores.
    Returns a dictionary containing parsed scores and any stderr.
    """
    scores = {}
    # Example line: "Score RouteID_S_AAABBBCCC_DDD_EEE_20230101_FFFF_GGG: 0.00000"
    # Or "No score RouteID_..."
    score_pattern = re.compile(r"^Score\s+(\S+):\s*([0-9.-]+)", re.MULTILINE)
    no_score_pattern = re.compile(r"^No score\s+(\S+)", re.MULTILINE) # From score.c

    if stdout_str:
        for match in score_pattern.finditer(stdout_str):
            route_id = match.group(1)
            try:
                score_val = float(match.group(2))
                scores[route_id] = score_val
                print(f"Parsed score for {route_id}: {score_val}")
            except ValueError:
                print(f"Warning: Could not parse score value '{match.group(2)}' for {route_id}. Storing as string.")
                scores[route_id] = match.group(2) # Store as string if not float

        for match in no_score_pattern.finditer(stdout_str):
            route_id = match.group(1)
            if route_id not in scores: # Don't overwrite if already scored (e.g. if it had a score then "No score")
                scores[route_id] = "No score"
            print(f"Parsed 'No score' for {route_id}")
    
    final_result = {"route_scores": scores}
    if stderr_str and stderr_str.strip(): # Add non-empty stderr
        final_result["stderr_from_c_scorer"] = stderr_str.strip()
    
    if not scores and not (stderr_str and stderr_str.strip()):
        final_result["summary"] = "No scores parsed and no stderr from C scorer."
    elif not scores:
        final_result["summary"] = "No scores successfully parsed from C scorer output."
        
    return final_result

def main():
    parser = argparse.ArgumentParser(description="Apply C scoring algorithm to proposed route sequences.")
    parser.add_argument(
        "snapshot_name",
        nargs='?',
        default=None,
        help="Optional snapshot name for this scoring run (e.g., for versioning outputs)."
    )
    parser.add_argument(
        "--input_dir_apply_outputs",
        type=Path,
        default=None, # Will be set relative to project_root_dir / data-evaluation if None
        help="Directory containing outputs from model_apply.py (proposed_sequences.json, TSPLIB_1/tsp2amz.json). Defaults to 'data-evaluation/model_apply_outputs'"
    )
    parser.add_argument(
        "--input_dir_score_eval_data",
        type=Path,
        default=None, # Will be set relative to project_root_dir / data-evaluation if None
        help="Directory containing evaluation data (eval_actual_sequences.json). Defaults to 'data-evaluation/model_score_inputs'"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None, # Will be set relative to project_root_dir / data-evaluation if None
        help="Directory for saving the final score file. Defaults to 'data-evaluation/model_score_outputs'"
    )
    args = parser.parse_args()

    print(f"Script model_score.py started at: {datetime.datetime.now()}")
    if args.snapshot_name:
        print(f"Snapshot name: {args.snapshot_name}")

    script_dir = Path(__file__).resolve().parent
    lkh_amz_dir = script_dir.parent
    project_root_dir = lkh_amz_dir.parent

    # Base directories for evaluation-related data, configurable via args
    data_evaluation_base_dir = project_root_dir / "data-evaluation"

    apply_outputs_dir = args.input_dir_apply_outputs if args.input_dir_apply_outputs else data_evaluation_base_dir / "model_apply_outputs"
    score_eval_inputs_dir = args.input_dir_score_eval_data if args.input_dir_score_eval_data else data_evaluation_base_dir / "model_score_inputs"
    score_main_output_dir = args.output_dir if args.output_dir else data_evaluation_base_dir / "model_score_outputs"
    
    print(f"Reading model_apply.py outputs from: {apply_outputs_dir.resolve()}")
    print(f"Reading score evaluation inputs (actuals) from: {score_eval_inputs_dir.resolve()}")
    print(f"Saving score outputs to: {score_main_output_dir.resolve()}")

    # Workspace for C scorer inputs, relative to this script's main output directory
    scoring_workspace = score_main_output_dir / "c_score_workspace"
    actual_tours_dir_in_workspace = scoring_workspace / "ACTUAL_TOURS"

    # Input JSON files, paths constructed from new directory arguments
    proposed_sequences_file = apply_outputs_dir / "proposed_sequences.json"
    actual_sequences_file = score_eval_inputs_dir / "eval_actual_sequences.json"
    tsp2amz_json_file = apply_outputs_dir / "TSPLIB_1" / "tsp2amz.json" 

    output_file_name = f"model_performance_score_{args.snapshot_name}.json" if args.snapshot_name else "model_performance_score.json"
    score_output_file = score_main_output_dir / output_file_name

    # 1. Clean and create output & workspace directories
    print(f"Preparing output directory: {score_main_output_dir}")
    if score_main_output_dir.exists() and args.snapshot_name is None:
        print(f"Cleaning main score output directory: {score_main_output_dir}")
        shutil.rmtree(score_main_output_dir)
    score_main_output_dir.mkdir(parents=True, exist_ok=True)
    
    if scoring_workspace.exists():
        shutil.rmtree(scoring_workspace)
    scoring_workspace.mkdir(parents=True, exist_ok=True)
    actual_tours_dir_in_workspace.mkdir(parents=True, exist_ok=True)

    # 2. Prepare C score binary
    # Source of score binary is assumed to be from standard model_build_outputs
    model_build_outputs_dir = project_root_dir / "data" / "model_build_outputs" 
    score_binary_source_path = model_build_outputs_dir / "bin" / "score" 
    # Destination for the C score binary is within this script's main output directory
    score_binary_dest_dir = score_main_output_dir / "bin"
    score_binary_dest_dir.mkdir(parents=True, exist_ok=True)
    copied_score_binary_path = score_binary_dest_dir / "score"

    if not score_binary_source_path.exists():
        print(f"Error: C score binary not found at expected location: {score_binary_source_path}")
        print("Please ensure the 'score' binary is in 'data/model_build_outputs/bin/'. It is typically copied there by model_build.py.")
        return
    
    print(f"Copying C score binary from {score_binary_source_path} to {copied_score_binary_path}")
    shutil.copy(score_binary_source_path, copied_score_binary_path)
    copied_score_binary_path.chmod(0o755) # Make it executable

    # 3. Load necessary JSON input files
    print("Loading input JSON files...")
    proposed_sequences_data = load_json_file(proposed_sequences_file, "proposed sequences (from model_apply run)")
    actual_sequences_data = load_json_file(actual_sequences_file, "actual sequences (evaluation ground truth)")
    tsp2amz_data = load_json_file(tsp2amz_json_file, "tsp2amz mapping (from model_apply run)")

    abort_scoring = False
    error_messages = []
    if not proposed_sequences_data:
        error_messages.append(f"  - Loaded data for proposed sequences ({proposed_sequences_file.resolve()}) is empty or invalid.")
        abort_scoring = True
    if not actual_sequences_data:
        error_messages.append(f"  - Loaded data for actual sequences ({actual_sequences_file.resolve()}) is empty or invalid.")
        abort_scoring = True
    if not tsp2amz_data:
        error_messages.append(f"  - Loaded data for tsp2amz mapping ({tsp2amz_json_file.resolve()}) is empty or invalid.")
        abort_scoring = True
    
    if abort_scoring:
        print("Critical input JSON data is missing or empty. Aborting scoring. Details:")
        for msg in error_messages:
            print(msg)
        return

    # Create a lookup map for tsp2amz_data by actual RouteID
    tsp2amz_by_route_id = {}
    if tsp2amz_data: # Ensure it's not None
        for tsp_problem_key, tsp_info in tsp2amz_data.items(): # tsp_problem_key is e.g. "amz0000.ctsptw"
            if isinstance(tsp_info, dict) and "route_id" in tsp_info:
                actual_route_id = tsp_info["route_id"]
                tsp2amz_by_route_id[actual_route_id] = tsp_info
            else:
                print(f"Warning: Skipping entry in tsp2amz_data due to unexpected format or missing 'route_id': {tsp_problem_key}")
    
    # Copy .tsp files to the scoring workspace
    print("Copying .tsp problem files to scoring workspace...")
    copied_tsp_files = 0
    if tsp2amz_by_route_id: 
        for actual_route_id, tsp_info_entry in tsp2amz_by_route_id.items(): 
            if isinstance(tsp_info_entry, dict) and "tsp_file_path" in tsp_info_entry and "problem_filename" in tsp_info_entry:
                # tsp_file_path in tsp2amz.json is relative to the directory of tsp2amz.json itself.
                tsp_problem_base_dir = tsp2amz_json_file.parent 
                source_tsp_relative_path_from_json = Path(tsp_info_entry["tsp_file_path"])
                source_tsp_full_path = tsp_problem_base_dir / source_tsp_relative_path_from_json
                
                # Name the destination TSP file based on the actual_route_id, like the .tour files
                original_tsp_filename = Path(tsp_info_entry["problem_filename"])
                original_extension = "".join(original_tsp_filename.suffixes) # e.g., .ctsptw.tsp or .tsp
                destination_tsp_name = f"{actual_route_id}{original_extension}"
                destination_tsp_path = scoring_workspace / destination_tsp_name
                
                if source_tsp_full_path.exists():
                    try:
                        shutil.copy(source_tsp_full_path, destination_tsp_path)
                        copied_tsp_files += 1
                    except Exception as e:
                        print(f"Warning: Could not copy TSP file {source_tsp_full_path} to {destination_tsp_path}: {e}")
                else:
                    print(f"Warning: TSP file not found at {source_tsp_full_path} for route corresponding to {actual_route_id}. Skipping copy.")
            else:
                available_keys = list(tsp_info_entry.keys()) if isinstance(tsp_info_entry, dict) else "Not a dict"
                print(f"Warning: Incomplete tsp_info for route {actual_route_id} (available keys: {available_keys}) when trying to copy .tsp file. Skipping.")
        print(f"Copied {copied_tsp_files} .tsp files to {scoring_workspace.resolve()}")
    else:
        print("Warning: tsp2amz_by_route_id mapping is empty. Cannot copy .tsp files.")

    # 4. Generate .tour files for the C scorer
    print("Generating .tour files for proposed sequences...")
    num_proposed_tours = generate_tour_files_from_json(
        proposed_sequences_data, tsp2amz_by_route_id, scoring_workspace, "proposed"
    )
    print("Generating .tour files for actual sequences...")
    num_actual_tours = generate_tour_files_from_json(
        actual_sequences_data, tsp2amz_by_route_id, actual_tours_dir_in_workspace, "actual_sequence"
    )

    if num_proposed_tours == 0 and num_actual_tours == 0: # Check if any tours were generated at all
        print("Warning: No .tour files (neither proposed nor actual) were generated. C scorer will likely find no data to score.")
        # Allow to proceed, C scorer will output "No score"
    elif num_proposed_tours == 0:
        print("Warning: No proposed tour files were generated. C scorer might not produce results for proposed tours.")
    elif num_actual_tours == 0:
        print("Warning: No actual tour files were generated for ACTUAL_TOURS. C scorer might not produce results for actual tours.")

    # 5. Run the C scorer
    print("Running C scoring executable...")
    stdout_str, stderr_str = run_c_scorer(copied_score_binary_path, scoring_workspace)
    
    # 6. Parse C scorer output
    final_scores = {}
    if stdout_str is not None or (stderr_str is not None and stderr_str.strip()):
        print("Parsing C scorer output...")
        final_scores = parse_c_score_output(stdout_str if stdout_str else "", stderr_str if stderr_str else "")
    else:
        print("No output (stdout/stderr) received from C scorer to parse.")
        final_scores = {
            "summary": "No output from C scorer.",
            "route_scores": {},
            "stderr_from_c_scorer": stderr_str if stderr_str else "Not available"
        }
        
    # 7. Save the final scores
    print("Saving final scores...")
    save_json_file(final_scores, score_output_file, "C performance score")
        
    print(f"Script model_score.py finished at: {datetime.datetime.now()}")
    print(f"Scoring results, using the C scorer, should be in: {score_output_file}")
    print(f"Temporary C scorer workspace (with .tour files) is at: {scoring_workspace}")

if __name__ == "__main__":
    main() 