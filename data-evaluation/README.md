This structure can be found at [Amazon Routing Challenge](https://github.com/MIT-CAVE/rc-cli/tree/main/templates)

1. Explore the datasets available for the [2021 Amazon Last Mile Routing Research Challenge](https://registry.opendata.aws/amazon-last-mile-challenges/)
```sh
aws s3 ls --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/
```

2. Select the desired dataset (**evaluation** or **training**) and copy it to your `data` directory in the `rc-cli` installation path
   - Evaluation dataset:
    ```
    aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-evaluation/ ~/.rc-cli/data/
    ```
   - Training dataset:
    ```
    aws s3 sync --no-sign-request s3://amazon-last-mile-challenges/almrrc2021/almrrc2021-data-training/ ~/.rc-cli/data/
    ```

## Project Structures
Regardless of the programming language(s) or libraries you use for your application, the following directories and files must be present:

```
├── data
│   ├── model_build_inputs
│   │   ├── actual_sequences.json
│   │   ├── invalid_sequence_scores.json
│   │   ├── package_data.json
│   │   ├── route_data.json
│   │   └── travel_times.json
│   ├── model_build_outputs
│   ├── model_apply_inputs
│   │   ├── new_package_data.json
│   │   ├── new_route_data.json
│   │   └── new_travel_times.json
│   ├── model_apply_outputs
│   │   └── proposed_sequences.json
│   ├── model_score_inputs
│   │   ├── new_actual_sequences.json
│   │   └── new_invalid_sequence_scores.json
│   ├── model_score_outputs
│   │   └── scores.json
│   └── model_score_timings
│       ├── model_apply_time.json
│       └── model_build_time.json
├── data-evaluation
│   ├── model_apply_inputs
│   │   ├── new_package_data.json
│   │   ├── new_route_data.json
│   │   ├── new_travel_time.json
│   ├── model_apply_outputs
│   ├── model_score_inputs
│   │   ├── eval_actual_sequences.json
│   │   ├── eval_invalid_sequence_scores.json
│   ├── model_score_outputs
├── LKH-AMZ
├── diagram
├── scripts
│   └── <source-code-file(s)-or-dir(s)>
├── Dockerfile
├── model_build.sh
└── model_apply.sh
```
