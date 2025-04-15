# ReSym Artifact

This repository provides artifacts for the paper [**"ReSym: Harnessing LLMs to Recover Variable and Data Structure Symbols from Stripped Binaries"**](https://www.cs.purdue.edu/homes/lintan/publications/resym-ccs24.pdf) (CCS 2024).

🏆 **ACM SIGSAC Distinguished Paper Award Winner**


## Instruction

This document explains the configuration file used to control the full pipeline of ReSym: from data processing to model training, inference, evaluation for `VarDecoder` and `FieldDecoder`, and the posterior reasoning.


### 1. Prerequisites

1.1 **Docker Setup**  
   Pull and run the provided Docker container for the environment setup:
   ```bash
   docker pull dnxie/resym:cuda
   ```
   ```
   docker run -it --rm --memory=100G --name resym \
     -v /absolute/path/to/resym:/home/ReSym \
     -v /absolute/path/to/data/folder:/home/data \
     -v /absolute/path/to/results/folder:/home/results \
     dnxie/resym:cuda
   ```
  

1.2 **Directory Requirements:**

- `/home/ReSym` – the ReSym source code pulled from github
- `/home/data` – if using our dataset on [Zenodo](https://zenodo.org/records/13923982) (`ReSym_rawdata`), point to the root of `ReSym_rawdata` or a similar structure (see `./process_data/readme.md` for details and requirements)
- `/home/results` - Your preferred folder for storing all results

Note that if you wish you use existing spliting for training and testing, put this file under `/home/results/training_data/split.json`


1.3 **Set your hugging face account**

Your Hugging Face access token must be exported as:
  ```bash
  export HF_TOKEN=your_token_here
```

1.4 **Data**  
We provide two binary files in the `ReSym/sample_data` folder as an example. Our full dataset from [Zenodo](https://zenodo.org/records/13923982) (`ReSym_rawdata`) is with the same structure. If you wish to prepare data yourself, please make sure your data folder should contain two subfolders:
- `bin`: Contains **non-stripped binaries** with debug information. (Note that `bin` folder is not required for test mode, see below for more details.)
- `decompiled`: Contains decompiled code from **fully stripped binaries**.

To use our sample data:
```bash
docker run -it --rm --memory=100G --name resym \
     -v /absolute/path/to/resym:/home/ReSym \
     -v /absolute/path/to/resym/sample_data:/home/data \
     -v /absolute/path/to/results/folder:/home/results \
     dnxie/resym:cuda
```



### 2. Start the pipeline

ReSym is designed to function as a **press-button end-to-end pipeline**. Once properly configured, you can run the entire workflow — from data processing to training, inference, and evaluation — using a single command:

```bash
cd /home/ReSym
```
```bash
bash run_resym.sh
```

This script will:
1. Load and validate your configuration
2. Process raw and decompiled data
3. Train or load models as specified
4. Run inference
5. Optionally perform evaluation and posterior reasoning


⚠️ **However**, before running this script, **please carefully read the next section** to understand and configure the config file. Misconfigurations may lead to skipped stages, unintended overwrites, or incomplete results.


### 3. Config File

The config file (named `./config`) is a simple Bash-style key-value list that is **sourced** into the main script (`run_resym.sh`). It controls all modes, parameters, paths, and flags used in the pipeline.

#### 3.1 Mode Flags

| Variable       | Type    | Description                                                                                                                                                                                                                                                                                           | Default |
|----------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| `test_mode`    | boolean | If `true`, enables test mode. Ground truth will not be collected and evaluation is skipped. Use this mode when applying ReSym to real-world data where ground truth is not available. It will only use the decompiled code from `/home/data/decompiled` for fully-stripped binaries.                  | `true` |
| `field`        | boolean | Considers variable clusters and field access expressions. If `false`, ReSym will only consider singleton variables.                                                                                                                                                                                   | `true`  |
| `reason`       | boolean | Enables posterior reasoning (requires `field=true`).                                                                                                                                                                                                                                                   | `true`  |
| `clean`        | boolean | If `true`, cleans up all intermediate results after processing.                                                                                                                                                                                                                                        | `true`  |

---

#### 3.2 Training Configuration

| Variable                     | Type    | Description                                                                                                 | Default                        |
|------------------------------|---------|-------------------------------------------------------------------------------------------------------------|--------------------------------|
| `TRAIN_ON`                   | boolean | If `true`, trains VarDecoder and FieldDecoder (if `field=true`). If `false`, uses existing checkpoints, ignore all the settings in this section.     | `false`                         |
| `EPOCH`                      | int     | Number of training epochs.                                                     | `3`                            |
| `BATCH_SIZE`                 | int     | Per-device batch size during training.                                                                      | `32`                            |
| `LR`                         | float   | Learning rate for both models.                                                                              | `0.001`                        |
| `TRAIN_SPLIT`                | float   | Proportion of binaries used for training (1 - value = test split). Set to 0 if want to use all data for testing.                                         | `0.5`                          |
| `MODEL_NAME`                | string  | HF model ID or path used to initialize the model. The training scripts is designed for starcoder family. If changed to other models, the scripts may crash.                                                            | `"bigcode/starcoderbase-3b"`   |
| `vardecoder_max_token_train`| int     | Max input token length for VarDecoder training.                                                             | `4096`                         |
| `fielddecoder_max_token_train`| int   | Max input token length for FieldDecoder training.                                                           | `4096`                         |
|
| `LOG_STEPS`                  | int     | Interval (in steps) for logging training progress.                                                          | `10`                           |
| `BF16`                       | boolean | If `true`, enables bfloat16 (bf16) precision during training. Note that the setting may need to be changed based on the GPU. BF16 (BFloat16) is primarily supported by NVIDIA Ampere and later GPU architectures                                               | `true`                         |


#### 3.3 Inference Configuration

| Variable                     | Type    | Description                                                                 | Default                          |
|------------------------------|---------|-----------------------------------------------------------------------------|----------------------------------|
| `num_beams`                 | int     | Number of beams for beam search decoding during inference.                 | `4`                              |
| `vardecoder_ckpt`           | string  | Path to VarDecoder checkpoint (used only when `TRAIN_ON=false`).           | `"/home/models/vardecoder"`      |
| `fielddecoder_ckpt`         | string  | Path to FieldDecoder checkpoint (used only when `TRAIN_ON=false`).         | `"/home/models/fielddecoder"`    |
| `vardecoder_max_token_inf`  | int     | Max input token length for VarDecoder inference. Must be larger than 1024.                           | `8192`                           |
| `fielddecoder_max_token_inf`| int     | Max input token length for FieldDecoder inference. Must be larger than 1024.                         | `8192`                           |
                    




####  3.4 Hardware & Environment


| Variable        | Type    | Description                                                         | Default          |
|------------------|---------|---------------------------------------------------------------------|------------------|
| `VISIBLE_GPUS`   | string  | Comma-separated GPU indices to expose to the job.                   | `"0,1,2,3"`      |
| `MAX_PROC`   | int  | Maximum number of processes used when pre-processing data.                   | `20`      |



---

####  3.5 Configuration Caveats

To make the pipeline more user-friendly and robust, the `run_resym.sh` script includes several **automatic checks and overwrite rules** to prevent inconsistent settings. Below are key caveats to be aware of when configuring the system:

**3.5.1. `test_mode=true` Implications**

When `test_mode` is enabled or `TRAIN_ON` is disabled:
- `EPOCH`, `BATCH_SIZE`, `LR`, `TRAIN_SPLIT`, and `model_name` are **ignored**.
- `TRAIN_SPLIT` is forcibly set to `0.0`.
- Evaluation steps are **skipped** entirely.
- Used for applying ReSym to **fully-stripped binaries** where ground truth labels are not available.

**💡 Tip:** You should also set `TRAIN_ON=false` when `test_mode=true`, though the script does enforce this automatically.

---

**3.5.2 `TRAIN_ON=false` Behavior**

If `TRAIN_ON=false`, the script:
- **Skips training** and loads checkpoints from `vardecoder_ckpt` and `fielddecoder_ckpt`.
- Automatically resets `model_name` to `"bigcode/starcoderbase-3b"` unless otherwise specified.
- Does **not** update checkpoint paths during runtime.

---

**3.5.3 `reason=true` Forces `field=true`**

If `reason=true` but `field=false`, the script prints a warning and **automatically enables `field=true`**.

This ensures posterior reasoning logic has the required field data structures.



### 4. Results

After running the pipeline via `run_resym.sh`, all intermediate and final output files will be stored under the `/home/results/` directory.

#### VarDecoder Results

All training/test splits and prediction results are located here: `/home/results/training_data`. Specifically, you can find the recovered  **variable names and types** in `/home/results/training_data/vardecoder_pred.jsonl`.

This file contains prediction entries for each test decompiled file, including:
- The input prompt used in the `prompt` entry of each line
- The predicted variable **name** and **type** in the `pred` entry of each line
- Inference time per example (for profiling/debugging)
- Some other metadata information like binary name, function id, etc.

#### Posterior Reasoning Results

Results for **user-defined data structures** — focusing on struct layout recovery — are available in: `/home/results/posterior_reasoning_results/results.json`.

This file summarizes recovered information about each structure, including:
- data structure name
- List of fields
- Field names and types
- Field sizes and offsets
- Structural layout and inferred organization

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

**Key structure**: Each key in the JSON file is formatted using the `**` separator and has the following components: author name, project name, binary name, function name, and variable name.


**Value Structure**: Each value contains two main components:
- **`pred`** (predicted layout): The predicted layout of the variable after applying posterior reasoning. It includes:
  - **type**: The predicted type of the variable (i.e., struct type).
  - **offsets**: A dictionary where each key represents an offset, and the value contains:
    - **size**: The predicted size of the field.
    - **name**: The predicted name of the field.
    - **type**: The predicted type of the field.
    
- **`gt`** (ground truth layout): The ground truth layout of the variable, structured similarly to `pred`. Note that ground truth is not available in the `test` mode.




## ReSym Overview

The ReSym tool consists of three main components:
#### **1. Data Processing** (`./process_data`)

Converts raw binary and decompiled code into a format suitable for model training or inference. Details of the usage can be found in (`./process_data/readme.md`)

#### **2. Model Inference / Training** (`./training_src` and Zenodo)

- Use this stage to train the model or run inference on your processed data.
- If you prefer to skip training, we provide pre-trained checkpoints available on Zenodo.


#### **3. Posterior Reasoning** (`./posterior_reasoning`)

Aggregates model predictions across functions and modules to recover the layout of user-defined data structures. Details of the usage can be found in (`./posterior_reasoning/readme.md`)

If your goal is only to recover variable names and types, the posterior reasoning stage is not required. It is specifically designed for recovering complex structure layouts.



## Provided Data and Resources

- **Our complete dataset: Binary files and decompiled code**: Available on [Zenodo](https://zenodo.org/records/13923982) (`ReSym_rawdata`). This includes raw binary files and corresponding decompiled code we used in this project:
     - `bin/`: Contains raw **non-stripped binary files** with debugging information.
     - `decompiled/`: Decompiled code from **fully stripped** binaries.
     - `metadata.json`: Metadata for the binaries, including project information.
     - **Note**: You can generate annotations using the provided scripts in this repository.
- **Training, testing, and prediction data on our dataset**: Available on [Zenodo](https://zenodo.org/records/13923982) (`ReSym_data`). This includes: training data, testing data, and prediction results for FieldDecoder and VarDecoder. 
- **Model checkpoints**: Fine-tuned VarDecoder and FieldDecoder model checkpoints are available on [Zenodo](https://zenodo.org/records/13923982). **They are also integrated into the docker container we provide.**




## Citing us
```
@inproceedings{10.1145/3658644.3670340,
author = {Xie, Danning and Zhang, Zhuo and Jiang, Nan and Xu, Xiangzhe and Tan, Lin and Zhang, Xiangyu},
title = {ReSym: Harnessing LLMs to Recover Variable and Data Structure Symbols from Stripped Binaries},
year = {2024},
isbn = {9798400706363},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3658644.3670340},
doi = {10.1145/3658644.3670340},
abstract = {Decompilation aims to recover a binary executable to the source code form and hence has a wide range of applications in cyber security, such as malware analysis and legacy code hardening. A prominent challenge is to recover variable symbols, including both primitive and complex types such as user-defined data structures, along with their symbol information such as names and types. Existing efforts focus on solving parts of the problem, e.g., recovering only types (without names) or only local variables (without user-defined structures). In this paper, we propose ReSym, a novel hybrid technique that combines Large Language Models (LLMs) and program analysis to recover both names and types for local variables and user-defined data structures. Our method encompasses fine-tuning two LLMs to handle local variables and structures, respectively. To overcome the token limitations inherent in current LLMs, we devise a novel Prolog-based algorithm to aggregate and cross-check results from multiple LLM queries, suppressing uncertainty and hallucinations. Our experiments show that ReSym is effective in recovering variable information and user-defined data structures, substantially outperforming the state-of-the-art methods.},
booktitle = {Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security},
pages = {4554–4568},
numpages = {15},
keywords = {large language models, program analysis, reverse engineering},
location = {Salt Lake City, UT, USA},
series = {CCS '24}
}
```
