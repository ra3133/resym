# Artifact

## Tool Overview

The tool consists of three main components:
#### **1. Data Processing** (`./process_data`)

Converts raw binary and decompiled code into a format suitable for model training or inference. Details of the usage can be found in (`./process_data/readme.md`)

#### **2. Model Inference / Training** (`./training_src` and Zenodo)

- Use this stage to train the model or run inference on your processed data.
- If you prefer to skip training, we provide pre-trained checkpoints available on Zenodo.


#### **3. Posterior Reasoning** (`./posterior_reasoning`)

Aggregates model predictions across functions and modules to recover the layout of user-defined data structures. Details of the usage can be found in (`./posterior_reasoning/readme.md`)

If your goal is only to recover variable names and types, the posterior reasoning stage is not required. It is specifically designed for recovering complex structure layouts.



## Provided Data and Resources

- **Data preparation script**: Located in the `process_data` folder. It generates training data with ground truth symbol information. The script is push-button, and usage instructions are provided in the folder.
- **Binary files and decompiled code**: Available on [Zenodo](https://zenodo.org/records/13923982) (`ReSym_rawdata`). This includes raw binary files and corresponding decompiled code we used in this project:
     - `bin/`: Contains raw **non-stripped binary files** with debugging information.
     - `decompiled/`: Decompiled code from **fully stripped** binaries.
     - `metadata.json`: Metadata for the binaries, including project information.
     - **Note**: You can generate annotations using the provided scripts in this repository.
- **Training/inference scripts**: Found in the `training_src` folder for VarDecoder and FieldDecoder models.
- **Training, testing, and prediction data**: Available on [Zenodo](https://zenodo.org/records/13923982) (`ReSym_data`). This includes: training data, testing data, and prediction results for FieldDecoder and VarDecoder. 
- **Model checkpoints**: Fine-tuned VarDecoder and FieldDecoder model checkpoints are available on [Zenodo](https://zenodo.org/records/13923982).
- **Final results**: Posterior reasoning results for recovering user-defined data structures in folder `posterior_reasoning`. The details and instructions can be found in the folder.



