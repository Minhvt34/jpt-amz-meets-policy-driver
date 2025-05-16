import os
import shutil
import subprocess
from pathlib import Path
import datetime

def main():
    print(f"Script started at: {datetime.datetime.now()}")

    # Define base directory assuming this script is in LKH-AMZ/scripts/
    # So, LKH-AMZ is the parent of the script's directory.
    script_dir = Path(__file__).resolve().parent
    lkh_amz_dir = script_dir.parent
    project_root_dir = lkh_amz_dir.parent # Parent of LKH-AMZ
    
    data_dir = project_root_dir / "data" # Data directory is sibling to LKH-AMZ
    model_build_inputs_dir = data_dir / "model_build_inputs"
    model_build_outputs_dir = data_dir / "model_build_outputs"
    
    out_file_model_json = model_build_outputs_dir / "model.json"

    # 1. Clean model_build_outputs directory
    print(f"Cleaning directory: {model_build_outputs_dir}")
    if model_build_outputs_dir.exists():
        shutil.rmtree(model_build_outputs_dir)
    model_build_outputs_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy compiled C binaries
    # Assuming these are in lkh_amz_dir (root of LKH-AMZ checkout) after a general 'make'
    # The original .sh script copied from $BASE_DIR/LKH-AMZ/ which implies they are in the LKH-AMZ sub-folder of where script is.
    # If this script is in LKH-AMZ/scripts, then lkh_amz_dir is correct.
    binaries_output_dir = model_build_outputs_dir / "bin"
    binaries_output_dir.mkdir(parents=True, exist_ok=True)
    
    source_binaries_dir = lkh_amz_dir # This is where 'LKH', 'solve' etc. are created by root Makefile
                                      # and the SRC Makefile creates LKH in ../ (i.e., lkh_amz_dir)
    
    binaries_to_copy = ["get_Length", "score", "merge", "solve", "LKH"]
    print(f"Copying C binaries from {source_binaries_dir} to {binaries_output_dir}")
    for bin_name in binaries_to_copy:
        source_file = source_binaries_dir / bin_name
        if source_file.exists():
            shutil.copy(source_file, binaries_output_dir / bin_name)
            print(f"  Copied {bin_name}")
        else:
            print(f"  Warning: Binary {bin_name} not found at {source_file}")

    # 3. Copy route_data for zone fixup
    print(f"Copying route_data for zone fixup.")
    input_route_data = model_build_inputs_dir / "route_data.json"
    output_route_data = model_build_outputs_dir / "route_data.json"
    if input_route_data.exists():
        shutil.copy(input_route_data, output_route_data)
        print(f"  Copied {input_route_data} to {output_route_data}")
    else:
        print(f"  Warning: Input route_data.json not found at {input_route_data}")
        
    # 4. Build model by running analyze_zone_id_order_main.py
    # Script is in ../scripts/ relative to LKH-AMZ directory
    analyze_script_path = project_root_dir / "scripts" / "analyze_zone_id_order_main.py"
    input_actual_sequences = model_build_inputs_dir / "actual_sequences.json"

    if not analyze_script_path.exists():
        print(f"Error: analyze_zone_id_order_main.py not found at {analyze_script_path}")
        return

    if not input_route_data.exists():
        print(f"Error: Cannot run analyze script, input route_data.json not found: {input_route_data}")
        return

    if not input_actual_sequences.exists():
         print(f"Error: Cannot run analyze script, input actual_sequences.json not found: {input_actual_sequences}")
         return

    print("Building model using analyze_zone_id_order_main.py...")
    cmd = [
        "python3", str(analyze_script_path),
        "--r_file", str(input_route_data),
        "--s_file", str(input_actual_sequences),
        "--out_file", str(out_file_model_json)
    ]
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=lkh_amz_dir)
        print("Analyze script stdout:")
        print(process.stdout)
        print("Analyze script stderr:")
        print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running analyze_zone_id_order_main.py:")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
    except FileNotFoundError:
        print(f"Error: python interpreter not found or script {analyze_script_path} not executable.")


    # --- ML INTEGRATION POINT ---
    # Here, you would typically:
    # 1. Load historical data (e.g., route_data.json, actual_sequences.json).
    # 2. Train your ML model using this data.
    # 3. Save the trained ML model and any associated preprocessing objects (scalers, encoders)
    #    to the model_build_outputs_dir.
    #    Example: joblib.dump(model, model_build_outputs_dir / "my_ml_model.pkl")
    # The output of this ML process might be the 'model.json' itself, or supplement it,
    # or produce parameters that LKH will use.
    print("Placeholder for ML model training.")
    # --- END ML INTEGRATION POINT ---

    print(f"Script finished at: {datetime.datetime.now()}")

if __name__ == "__main__":
    main() 