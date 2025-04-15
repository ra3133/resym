#!/bin/bash

# === Load Configuration ===
CONFIG_FILE="config"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file $CONFIG_FILE not found."
    exit 1
fi

source "$CONFIG_FILE"

# === Validate Config Logic ===
if [ "$test_mode" = true ]; then
    echo "[WARNING] Test mode is enabled. Overwrite TRAIN_ON to false. EPOCH, BATCH_SIZE, LR, TRAIN_SPLIT, and model_name will be ignored."
    TRAIN_SPLIT=0
    TRAIN_ON=false
fi 

if [ "$TRAIN_ON" = true ]; then
    echo "[INFO] Training mode is enabled. vardecoder_ckpt and fielddecoder_ckpt will be ignored."
    vardecoder_ckpt=""  # clear it
    fielddecoder_ckpt=""
else
    vardecoder_ckpt="/home/models/vardecoder"
    fielddecoder_ckpt="/home/models/fielddecoder"
    model_name="bigcode/starcoderbase-3b"
    TRAIN_SPLIT=0
    echo "[INFO] Training mode is disabled. Use the existing checkpoints. Overwrite vardecoder_ckpt, fielddecoder_ckpt, and model_name to default value."
    echo "[INFO] Training mode is disabled. Overwrite TRAIN_SPLIT to 0. Use all data for testing."
fi

if [ "$reason" = true ] && [ "$field" != true ]; then
    echo "[WARNING] 'reason=true' requires 'field=true'. Overriding 'field' to true."
    field=true
fi

# === Compute Test Split ===
TEST_SPLIT=$(awk "BEGIN {print 1 - $TRAIN_SPLIT}")


# === Paths ===
data_root="/home/data"
result_folder="/home/results"
code_root="/home/ReSym"
model_folder="/home/models"

train_data_folder="$result_folder/training_data"
posterior_results_folder="$result_folder/posterior_reasoning_results"
retrain_model_folder="$result_folder/models"

mkdir -p "$train_data_folder"
mkdir -p "$posterior_results_folder"

# === Optional flags ===
TEST_FLAG=""
if [ "$test_mode" = true ]; then
    TEST_FLAG="--test"
fi

FIELD_FLAG=""
if [ "$field" = true ]; then
    FIELD_FLAG="--field"
fi

REASON_FLAG=""
if [ "$reason" = true ]; then
    REASON_FLAG="--reason"
fi

CLEAN_FLAG=""
if [ "$clean" = true ]; then
    CLEAN_FLAG="--clean"
fi

BF16_FLAG=""
if [ "$BF16" = true ]; then
    BF16_FLAG="--bf16"
fi


# === Utility Functions ===

find_latest_valid_checkpoint() {
    local checkpoint_root=$1
    local latest_ckpt=""
    for ckpt in $(ls -d "$checkpoint_root"/checkpoint-* 2>/dev/null | sort -V); do
        if [ -f "$ckpt/config.json" ] && [ -f "$ckpt/generation_config.json" ] && \
        { [ -f "$ckpt/pytorch_model.bin" ] || [ -f "$ckpt/model.safetensors" ]; }; then
            latest_ckpt="$ckpt"
        fi
    done

    if [ -n "$latest_ckpt" ]; then
        echo "$latest_ckpt"
    else
        echo "[WARNING] No valid checkpoint found in $checkpoint_root. Falling back to base dir." >&2
        echo "$checkpoint_root"
    fi
}

# === Activate Conda Env ===
eval "$(conda shell.bash hook)"
conda activate binary

# === Step 1: Process Data ===
cd "$code_root/process_data" || exit 1
echo "[INFO] Processing data..."
bash process_data.sh "$data_root" $FIELD_FLAG $REASON_FLAG $TEST_FLAG $CLEAN_FLAG --max_proc $MAX_PROC


# === Step 2: Generate JSONL Files ===
cd "$code_root/process_data" || exit 1

echo "[INFO] Generating JSONL for vardecoder..."
python gen_jsonl.py \
    --input_folder "$data_root/train_var" \
    --decompiled_folder "$data_root/decompiled" \
    --output_folder "$train_data_folder" \
    --train "$TRAIN_SPLIT" \
    --test "$TEST_SPLIT" \
    --model vardecoder

if [ "$field" = true ]; then
    echo "[INFO] Generating JSONL for fielddecoder..."
    python gen_jsonl.py \
        --input_folder "$data_root/train_field" \
        --decompiled_folder "$data_root/decompiled" \
        --output_folder "$train_data_folder" \
        --train "$TRAIN_SPLIT" \
        --test "$TEST_SPLIT" \
        --model fielddecoder
fi

# === Step 3: Run Training or Inference ===
cd "$code_root/training_src" || exit 1

export WANDB_MODE=disabled

if [ "$TRAIN_ON" = true ]; then
    echo "[INFO] Running training mode..."
    NUM_GPUS=$(echo "$VISIBLE_GPUS" | awk -F',' '{print NF}')
    # Define checkpoint paths for retraining
    var_ckpt_dir="$retrain_model_folder/vardecoder"
    field_ckpt_dir="$retrain_model_folder/fielddecoder"
    # Train VarDecoder
    CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS torchrun --nproc-per-node=$NUM_GPUS vardecoder_train.py \
        "$train_data_folder/vardecoder_train.jsonl" \
        "$var_ckpt_dir" \
        --model_name "$MODEL_NAME" \
        --max_token "$vardecoder_max_token_train" \
        --lr "$LR" \
        --epoch "$EPOCH" \
        --batch_size "$BATCH_SIZE" \
        $BF16_FLAG \
        --log_steps "$LOG_STEPS"
    # Update ckpt paths to newly trained ones
    vardecoder_ckpt=$(find_latest_valid_checkpoint "$var_ckpt_dir")

    if [ "$field" = true ]; then
        # Train FieldDecoder
        CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS torchrun --nproc-per-node=$NUM_GPUS fielddecoder_train.py \
            "$train_data_folder/fielddecoder_train.jsonl" \
            "$field_ckpt_dir" \
            --model_name "$MODEL_NAME" \
            --max_token "$fielddecoder_max_token_train" \
            --lr "$LR" \
            --epoch "$EPOCH" \
            --batch_size "$BATCH_SIZE" \
            $BF16_FLAG \
            --log_steps "$LOG_STEPS"
        fielddecoder_ckpt=$(find_latest_valid_checkpoint "$field_ckpt_dir")
    fi
fi


echo "[INFO] Running inference..."
cd "$code_root/training_src" || exit 1
CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS python vardecoder_inf.py \
    "$train_data_folder/vardecoder_test.jsonl" \
    "$train_data_folder/vardecoder_pred.jsonl" \
    "$vardecoder_ckpt" \
    --model_name "$MODEL_NAME" \
    --max_token "$vardecoder_max_token_inf" \
    --num_beams "$num_beams"

if [ "$field" = true ]; then
    cd "$code_root/training_src" || exit 1
    CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS python fielddecoder_inf.py \
        "$train_data_folder/fielddecoder_test.jsonl" \
        "$train_data_folder/fielddecoder_pred.jsonl" \
        "$fielddecoder_ckpt" \
        --model_name "$MODEL_NAME" \
        --max_token "$fielddecoder_max_token_inf" \
        --num_beams "$num_beams"
fi

# === Step 4: Run Evaluation ===
cd "$code_root/training_src" || exit 1
if [ "$test_mode" != true ]; then
    echo "[INFO] Running evaluation for VarDecoder..."
    python eval_vardecoder.py "$train_data_folder/vardecoder_pred.jsonl"

    if [ "$field" = true ]; then
        echo "[INFO] Running evaluation for FieldDecoder..."
        python eval_fielddecoder.py "$train_data_folder/fielddecoder_pred.jsonl"
    fi
fi


if [ "$reason" = true ]; then
    # === Step 5: Run Posterior Reasoning ===
    cd "$code_root/posterior_reasoning" || exit 1
    echo "[INFO] Running posterior reasoning..."
    bash run.sh $posterior_results_folder $TEST_FLAG $CLEAN_FLAG
fi
