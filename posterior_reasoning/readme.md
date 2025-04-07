# Posterior Reasoning 

This folder contains the the code and the final results after applying posterior reasoning, used to reproduce the results in **Table 5** of our paper.



## Important Contents

- **`run.sh`**: The script to run posterior reasoning
- **`results.json`**: A JSON file containing the complete posterior reasoning results of our paper.
- **`eval.py`**: A script to evaluate and display the results from `results.json`.


## Instruction of the tool

This script `run.sh` runs a multi-step pipeline that processes the entire posterior reasoning.

### Prerequisites

1. **Docker Setup**  
   Pull and run the provided Docker container for the environment setup:
   ```bash
   docker pull dnxie/resym:latest
   ```
   ```
   docker run -it --rm --memory=100G --name resym \
     -v /absolute/path/to/resym:/home/ReSym \
     -v /absolute/path/to/data/folder:/home/data \
     -v /absolute/path/to/prediction_result/folder:/home/results \
     -v /absolute/path/to/posterior_results/folder:/home/posterior_results \
     dnxie/resym:latest
   ```
   
2. **Directory Requirements:**

- `/home/ReSym` – the ReSym repo downloaded from github
- `/home/data` – if using our Zenodo dataset, point to the root of `ReSym_rawdata` or a similar structure (see `../process_data/readme.md` for details and requirements)
- `/home/results` - if using our artifacts, this is the `ReSym_data` folder. The folder must contain:
    - `vardecoder_pred.jsonl`: prediction results from VarDecoder
    - `fielddecoder_pred.jsonl`: prediction results from FieldDecoder
- `/home/posterior_results` – any folder path for saving intermediate and final results


    **If you want to test using the provided sample data for a quick test:**
    ```bash
    docker run -it --rm --memory=100G --name resym \
        -v /absolute/path/to/resym:/home/ReSym \
        -v /absolute/path/to/resym/sample_data:/home/data \
        -v /absolute/path/to/resym/sampled_results:/home/results \
        -v /absolute/path/to/your/preferred/posterior_results:/home/posterior_results \
        dnxie/resym:latest
    ```

3. **Conda Environment**  
   Activate the provided Conda environment for running the script:
   ```bash
   conda activate binary
   ```


### Usage

Inside the docker (Docker setup see `../readme.md`)

```
cd /home/ReSym/posterior_reasoning
```

```
bash run.sh [--test] [--clean]
```

#### Options
- `--test`: Enables **test mode**. No evaluation will be performed, and ground truth is not required. Use this mode if you are applying the tool to real-world data where the ground truth is not available. If the test mode is disabled, the script will automaticaly collect the ground truth and conduct evaluation at the end.
- `--clean`: Cleans up all intermediate files and folders after the pipeline finishes. Use this only if you no longer need intermediate artifacts. You shouldn't need to look into those. The pipeline is designed to be used as a black box. 


#### Important Requirements

Before running this script, you must prepare the data correctly:
- Make sure the model prediction results (`vardecoder_pred.jsonl` and `fielddecoder_pred.jsonl`) are in folder `/home/results`.
- Data will be read from the directory: `/home/data`
- Make sure to run the data pre-processing step beforehand from the `../process_data` directory using:
```
bash process_data.sh /home/data --field --reason (--test)
```

- The flags `--field` and `--reason` are **required** for posterior reasoning.
- If you used `--test` during data processing, you must also use `--test` when running this script. Otherwise there will be errors due to the missing of the ground truth.
- **Do not** use the `--clean` flag in the data processing step, as this will delete essential files needed for posterior reasoning.

### Output

The final output JSON will be located at:
```
/home/posterior_results/results.json
```

Intermediate logs and data will be stored in subfolders within `/home/posterior_results` unless `--clean` is used.

### Evalution

If you are not in the test mode, the evalution will be automatically done with `run.sh` and the results should be printed in the command line. 

However, if you wish to run the evalution again, simply run 
```bash
python eval.py results.json
```

If you are in test mode, no evaluation can be done.

## `results.json` Structure

The `results.json` file contains a dictionary where each key-value pair represents the evaluation of a particular variable in a binary. Here's the structure of an example entry:

```json
"mwarning**KadNode**2e8d7c4309cfa01389d7ccb4986397b29bf23e508049cabee256ea6ff17590b0**sub_413229**v2": {
    "pred": {
        "type": "man_viewer_info_list*",
        "offsets": {
            "0": {
                "size": 8,
                "name": "next",
                "type": "man_viewer_info_list*"
            },
            "8": {
                "size": 8,
                "name": "info",
                "type": "char*"
            }
        }
    },
    "gt": {
        "type": "peer*",
        "offsets": {
            "0": {
                "size": 8,
                "name": "next",
                "type": "peer*"
            },
            "8": {
                "size": 8,
                "name": "addr_str",
                "type": "char*"
            }
        }
    }
}
```

### Key Structure

Each key in the JSON file is formatted using the `**` separator and has the following components: author name, project name, binary name, function name, and variable name.


### Value Structure

Each value contains two main components:
- **`pred`** (predicted layout): The predicted layout of the variable after applying posterior reasoning. It includes:
  - **type**: The predicted type of the variable (i.e., struct type).
  - **offsets**: A dictionary where each key represents an offset, and the value contains:
    - **size**: The predicted size of the field.
    - **name**: The predicted name of the field.
    - **type**: The predicted type of the field.
    
- **`gt`** (ground truth layout): The ground truth layout of the variable, structured similarly to `pred`.

### Instructions for Evaluation

To evaluate the results and reproduce the findings presented in Table 5 of the paper:

```bash
python eval.py results.json
```
