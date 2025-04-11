#!/bin/bash

# ---------- Flag handling ----------
test_flag=""
clean_flag=""

# Parse flags and positional arguments
result_folder=$1
shift  # Shift to parse remaining arguments


for arg in "$@"; do
    if [[ "$arg" == "--test" ]]; then
        test_flag="--test"
        echo "[INFO] Test mode enabled. No ground truth is provided. No evaluation will be conduted."
    elif [[ "$arg" == "--clean" ]]; then
        clean_flag="--clean"
        echo "[INFO] Clean-up mode enabled. Will clean up all the intermediate files at the end."
    fi
done

# ---------- Utilities ----------


create_dir() {
    target_dir=$1

    # Check if the destination directory exists, create it if necessary
    if [ ! -d "$target_dir" ]; then
        mkdir -vp "$target_dir"
        if [ $? -ne 0 ]; then
            echo "Failed to create destination directory '$target_dir'." >&2
            exit 1
        fi
    fi
}


create_and_clean_dir() {
    target_dir=$1
    create_dir $target_dir
    # Check if the directory contains any files before trying to remove them
    if [ "$(ls -A "$target_dir")" ]; then
        rm -r "$target_dir"/*
        if [ $? -ne 0 ]; then
            echo "Warning: Could not remove some files in '$target_dir'."
        fi
    # else
    #     echo "The directory '$target_dir' is empty."
    fi
}


remove_folder_if_exists() {
    local folder="$1"

    if [[ -d "$folder" ]]; then
        echo "Removing folder: $folder"
        rm -r "$folder"
    fi
}

# ---------- Directories ----------
# result_folder="/home/posterior_results"
data_root="/home/data"

prep_folder="$result_folder/prep"
equiv_vars_folder="$result_folder/equiv_vars"
group_data_folder="$result_folder/group_data"
layout_eval_folder="$result_folder/layout_eval"
final_folder="$result_folder/final"

group_log="$result_folder/group_log"
vote_log="$result_folder/log"
result_json="$result_folder/results.json"


# ---------- Step execution ----------
create_and_clean_dir "$prep_folder"
python prep.py /home/results/training_data/vardecoder_pred.jsonl /home/results/training_data/fielddecoder_pred.jsonl  "$data_root" "$prep_folder" ${test_flag:+--test}

create_and_clean_dir "$equiv_vars_folder"
python callgraph.py "$prep_folder" "$equiv_vars_folder" ${test_flag:+--test}

create_and_clean_dir "$group_data_folder"
python group_info.py "$equiv_vars_folder" "$prep_folder" "$data_root" "$group_data_folder" ${test_flag:+--test} > "$group_log"

create_and_clean_dir "$layout_eval_folder"
python vote_offset.py "$equiv_vars_folder/" "$prep_folder" "$group_data_folder" "$layout_eval_folder" ${test_flag:+--test} > "$vote_log"

create_and_clean_dir "$final_folder"
python vote_type.py "$layout_eval_folder" "$final_folder"

python dump_result.py "$prep_folder" "$final_folder" "$data_root" --out "$result_json" ${test_flag:+--test}


if [[ -z "$test_flag" ]]; then
    echo "=============== Evaluation Results: ==============="
    python eval.py "$result_json"
else
    echo "[INFO] Skipping evalution in test mode."
fi

# ---------- Clean-up ----------
if [[ -n "$clean_flag" ]]; then
    echo "[INFO] Cleaning intermediate folders..."
    remove_folder_if_exists "$prep_folder"
    remove_folder_if_exists "$equiv_vars_folder"
    remove_folder_if_exists "$group_data_folder"
    remove_folder_if_exists "$layout_eval_folder"
    remove_folder_if_exists "$final_folder"
    rm $group_log
    rm $vote_log
    echo "[INFO] Cleanup complete."
fi


echo "The results can be found in $result_json"
