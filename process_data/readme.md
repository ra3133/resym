# ReSym Push-Button Data Processing Script

This repository provides code to generate training labels for **VarDecoder** (labels for singleton variable names and types) and **FieldDecoder** (labels for field access expressions) from binary and decompiled code.

## Prerequisites

1. **Docker Setup**  
   Pull and run the provided Docker container for the environment setup:
   ```bash
   docker pull dnxie/resym:latest
   ```
   ```
   docker run -it --rm --memory=100G --name resym \
      -v /absolute/path/to/resym:/home/ReSym \
      -v /absolute/path/to/your/data/folder:/home/data \
      dnxie/resym:latest
   ```
   

2. **Data**  
   We provide two binary files in the `ReSym/sample_data` folder as an example. Your data folder should contain two subfolders:
   - `bin`: Contains **non-stripped binaries** with debug information. (Note that `bin` folder is not required for test mode, see below for more details.)
   - `decompiled`: Contains decompiled code from **fully stripped binaries**.
  

   To use our sample data:
   ```
   docker run -it --rm --memory=100G --name resym \
      -v /absolute/path/to/resym:/home/ReSym \
      -v /absolute/path/to/resym/sample_data:/home/data \
      dnxie/resym:latest
   ```
3. **Conda Environment**  
   Activate the provided Conda environment for running the script:
   ```bash
   conda activate binary
   ```
 

## Usage

To process the data, run the following command:

```
cd /home/ReSym/process_data/
```

```bash
bash process_data.sh /home/data [--clean] [--field] [--reason] [--test]
```

- **Required Parameter**: `/home/data` is the path to your data folder (must include `bin` and `decompiled` directories).
- **Optional Flags**:
  - `--clean`: Cleans up all intermediate results after processing.
  - `--field`: Considers variable clusters and field access expressions to generate training data for both **VarDecoder** and **FieldDecoder**.
  - `--reason`: Collects necessary data for posterior reasoning. If you plan to run posterior reasoning, use this flag.
  - `--test`: Enables test mode. Use this mode when you are applying ReSym to real-world data where the ground truth is not avaialble. It will only use the decomplied code from `/home/data/decompiled` from fully-stripped binaries and generate data **without ground truth**. The generated data can be used to the models for inference.

### Example Command

```bash
bash process_data.sh /home/data --field --clean --reason
```

- **With `--field`**: Generates training data for both:
   - **VarDecoder**: with variable clusters (Section 3.2.1). The generated data will include variable clusters, which are labeled as `"-", "-"`.
   - **FieldDecoder**
- **Without `--field`**: Generates only the training data for **VarDecoder**, excluding variable clusters.

### Example Output

An example log for the above command looks like this:

```bash
bash process_data.sh /home/data/ --field --clean
Extracting both stack variables and field access information.
Clean flag is set. Will clean intermediate results after processing.
mkdir: created directory '/home/data/decompiled_files/'
mkdir: created directory '/home/data/decompiled_vars'
mkdir: created directory '/home/data/debuginfo_subprograms'
mkdir: created directory '/home/data/align'
mkdir: created directory '/home/data/train_var'
mkdir: created directory '/home/data/logs'
mkdir: created directory '/home/data/commands'
mkdir: created directory '/home/data/field_access/'
mkdir: created directory '/home/data/train_field'
=== Progress: 0/2 ===
=== Progress: 1/2 ===
Cleaning intermediate results.
Data processing finished. The results can be found in /home/data/train_var and /home/data/train_field.
```

## Customization

- **Parallel Processing**: The script processes up to 20 binary files in parallel by default (`MAX_PROC=20`). You can modify this value directly in the `process_data.sh` script.
- **Decompiled Code Format**: We recommend following our decompiled code format for easier integration. If using a different format, modify the code in `parse_decompiled.py` accordingly.

## Output

The output is stored in:
- `train_var`: Contains training data for **VarDecoder**.
- `train_field`: Contains training data for **FieldDecoder**.

Note that they are not necessarily all for training. They are the data format we used for the models. One can choose to use them as training or testing data.

### Example Output for VarDecoder

```json
{
    "code": "__int64 __fastcall sub_40171B(__int64 a1)\n{...}\n",
    "prompt": "In the following decompiled C program, what are the original name, data type, data size and tag of variables `a1`?\n```\n__int64 __fastcall sub_40171B(__int64 a1)\n{...}\n```",
    "output": "a1: uctxt, ucontext_t*",
    "funname": "read_mpx_status_sig",
    "label": {
        "a1": [
            "uctxt",
            "ucontext_t*"
        ]
    }
}
```

- **Prompt and Output**: These are the inputs and labels used for training the **VarDecoder** model.
- **Custom Use**: The raw code and labels are also provided for customization in other tasks.

