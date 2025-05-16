Create a Docker image and run model_build.sh

```bash
docker build -t jpt-amz-model .
```

"model-score" phase:
> model-score
> model-score [snapshot-name]
> Apply the scoring algorithm using data/model_apply_outputs/proposed_sequences.json created during the model-apply phase. The scoring algorithm compares your proposed route sequences against the actual sequences for the same set of stops...

This implies there's a dedicated scoring mechanism or script. The "optional fast internal JPT scoring" mentioned in the model_apply.sh logs (which we couldn't activate easily because the script cleans its output directory) might be a more basic or integrated check.
To proceed with scoring using the data-evaluation files:

1. Locate the Scoring Script/Mechanism:
We need to find out how the "scoring algorithm" is invoked.
LKH-AMZ/JPT/score.c and LKH-AMZ/JPT/JPTscore.c: These are C source files for scoring. The model_build.sh script compiles programs in LKH-AMZ and copies a binary named score to data/model_build_outputs/bin/. This score binary is then used by the ./merge and ./solve shell scripts.
The score.c main function looks for an ACTUAL_TOURS directory and compares tours from there with tours in a results directory (e.g., TOURS-TSPLIB). It then prints an average score.
JPTscore.c seems to contain the core scoring logic (JPTscore function) that calculates dev * erp.
./merge script's behavior:
The ./merge script (which exists in LKH-AMZ/merge and also in the output directories like data_results/merge and data_evaluation_results/merge because model_apply.sh copies it there from data/model_build_outputs/bin/) has this logic:
This means the compiled score program is indeed the one that would perform the internal scoring if ACTUAL_TOURS is present when ./merge is run.
scripts/exract_build_and_apply.py (misspelled as exract?): This script seems to be a utility for sampling and splitting a larger dataset into model_build_inputs, model_apply_inputs, and model_score_inputs. It also creates new_invalid_sequence_scores.json. This confirms that eval_invalid_sequence_scores.json is related to the scoring process.

Path to Local Evaluation:
Given this, here's how we can try to get a local score using your data-evaluation files:
The core challenge remains that model_apply.sh cleans the output directory (/app/data/model_apply_outputs/) at its start. This means any ACTUAL_TOURS directory we create before running model_apply.sh gets wiped.
However, the compiled score binary and the merge script are copied by model_apply.sh into the /app/data/model_apply_outputs/ directory (from /app/data/model_build_outputs/bin/) before they are executed.
This gives us an opportunity:
Run model_apply.sh. This will generate proposed_sequences.json and also place the merge, score, get_Length, etc., binaries into /app/data/model_apply_outputs/.

After model_apply.sh completes, but before exiting the interactive Docker session, we can then manually create ACTUAL_TOURS and invoke the ./score program directly.
Revised Plan for Local Scoring (within one interactive Docker session):
Start Docker Interactively (ensure your local data-evaluation/model_apply_inputs files are already renamed to new_route_data.json, etc.):

```bash
    docker run -it --rm \
        -v "$(pwd)/data-evaluation/model_apply_inputs:/app/data/model_apply_inputs:ro" \
        -v "$(pwd)/data-evaluation/model_score_inputs:/app/data/model_score_inputs:ro" \
        -v "$(pwd)/data_evaluation_results:/app/data/model_apply_outputs" \
        jpt-amz-model /bin/bash
```
		
Inside the Container (at /app# prompt):
Run model_apply.sh to generate the proposed tours and copy the necessary binaries:

```bash
./model_apply.sh
```

(This will produce proposed_sequences.json in /app/data/model_apply_outputs/ and also the merge, score binaries etc. in that same directory).
Navigate to the output directory where the binaries and tour data now reside:

```bash
 cd /app/data/model_apply_outputs
```
 
Create the ACTUAL_TOURS directory here:

```bash
mkdir ACTUAL_TOURS
```

Use route2tour.py to populate ACTUAL_TOURS. Note the path changes because we are now in /app/data/model_apply_outputs/:

```bash
    ../../src/scripts/route2tour.py --r_file ../model_apply_inputs/new_route_data.json --s_file ../model_score_inputs/eval_actual_sequences.json --out_dir ACTUAL_TOURS
```

(Verify path: ../../src/scripts/ from /app/data/model_apply_outputs/ should correctly point to /app/src/scripts/)
Now, directly invoke the compiled score program. The ./merge script calls it as ./score $result where $result is TSPLIB (the final merged tour directory name). So, you would run:

```bash
./score TSPLIB
```

This should look for TOURS-TSPLIB/ (containing your model's proposed tours) and ACTUAL_TOURS/ (containing the ground truth tours) and print a score.
Examine the output. Then you can exit the container.
This approach allows model_apply.sh to do its full run, including creating proposed_sequences.json and placing the necessary binaries. Then, we augment its output directory with the actual tours and run the scoring.
The eval_invalid_sequence_scores.json might be used by the score program implicitly if it encounters certain issues, or it might be for a higher-level scoring script provided by the competition. The C code for score.c doesn't show an obvious direct use of a JSON file for invalid scores, but it's possible the JPTscore function or other utilities handle it.