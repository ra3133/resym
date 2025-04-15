#!/bin/bash

MAX_PROC=20
check_interval=1


# Ensure at least one argument is provided (the required parameter)
if [ -z "$1" ]; then
    echo "Error: Missing required parameter."
    echo "Usage: $0 <required_param> [--field] [--clean] [--reason] [--test] [--max_proc <int>]"
    exit 1
fi

source_dir=$1    
shift

field_flag=""
clean_flag=""
reason_flag=""
test_flag=""

# === Parse flags and optional arguments ===
while [[ $# -gt 0 ]]; do
    case "$1" in
        --field)
            field_flag="--field"
            echo "Extracting both stack variables and field access information."
            shift
            ;;
        --reason)
            reason_flag="--reason"
            echo "Extract information for posterior reasoning. Will keep necessary intermediate results for posterior reasoning even if --clean flag is on."
            shift
            ;;
        --clean)
            clean_flag="--clean"
            echo "Clean flag is set. Will clean intermediate results after processing."
            shift
            ;;
        --test)
            test_flag="--test"
            echo "Test flag is set. Enter testing mode. Will only analyze the decompiled code and will not analyze the (unstripped) binary file."
            shift
            ;;
        --max_proc)
            if [[ -n "$2" && "$2" != --* ]]; then
                MAX_PROC="$2"
                echo "Max processes set to $MAX_PROC"
                shift 2
            else
                echo "[ERROR] --max_proc flag requires a numeric argument." >&2
                exit 1
            fi
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done


# Show default message if field flag not set
if [[ -z "$field_flag" ]]; then
    echo "Extracting stack variables only."
fi

# Check for conflict: reason without field
if [[ -n "$reason_flag" && -z "$field_flag" ]]; then
    echo "[Warning] Posterior reasoning cannot be proceeded without field access information. Overwrite --field to true" >&2
    field_flag="--field"
    # exit 1
fi


check_dir_exist() {
    target_dir=$1
    # Check if the source directory exists
    if [ ! -d "$target_dir" ]; then
        echo "Directory '$target_dir' does not exist." >&2
        exit 1
    fi
}

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

check_file_exists() {
    target_file=$1

    if [ ! -f "$target_file" ]; then
        echo "Required file '$target_file' does not exist." >&2
        exit 1
    fi
}

create_and_clean_dir() {
    target_dir=$1
    create_dir "$target_dir"

    if [ "$(ls -A "$target_dir")" ]; then
        find "$target_dir" -mindepth 1 -delete
        if [ $? -ne 0 ]; then
            echo "Warning: Could not remove some files in '$target_dir'."
        fi
    fi
}


remove_folder_if_exists() {
    local folder="$1"

    if [[ -d "$folder" ]]; then
        echo "Removing folder: $folder"
        rm -r "$folder"
    # else
    #     echo "Folder does not exist: $folder"
    fi
}


bin_dir="$source_dir/bin"
decompiled_dir="$source_dir/decompiled"
# meta_data_fpath="$source_dir/metadata.json"
check_dir_exist $bin_dir
check_dir_exist $decompiled_dir
# check_file_exists $meta_data_fpath

decompiled_files_dir="$source_dir/decompiled_files/"
decompiled_vars_dir="$source_dir/decompiled_vars"
debuginfo_subprograms_dir="$source_dir/debuginfo_subprograms"
align_var="$source_dir/align"
train_var="$source_dir/train_var"
logs_dir="$source_dir/logs"
commands_dir="$source_dir/commands"
field_access_dir="$source_dir/field_access/"
align_field="$source_dir/align_field"
train_field="$source_dir/train_field"
callsite_dir="$source_dir/callsite/"
dataflow_dir="$source_dir/dataflow/"

create_and_clean_dir $decompiled_files_dir
create_and_clean_dir $decompiled_vars_dir
create_and_clean_dir $logs_dir

if [ -z "$test" ]; then  # if test flag is NOT set
    create_and_clean_dir $debuginfo_subprograms_dir
    create_and_clean_dir $align_var
    create_and_clean_dir $train_var
fi


if [ -n "$field_flag" ]; then 
    create_and_clean_dir $commands_dir
    create_and_clean_dir $field_access_dir
    if [ -z "$test" ]; then  # if test flag is NOT set
        create_and_clean_dir $align_field
        create_and_clean_dir $train_field
    fi
fi

if [ -n "$reason_flag" ]; then
    create_and_clean_dir $callsite_dir
    create_and_clean_dir $dataflow_dir
fi


process_file() {
    local FILE=$1

    binname=$(basename "$FILE")
    
    # Check if the corresponding decompiled file exists, if not, skip processing
    if [ ! -f "$decompiled_dir/$binname.decompiled" ]; then
        return
    fi
    

    python prep_decompiled.py "$decompiled_dir/$binname.decompiled" $decompiled_files_dir $decompiled_vars_dir >> "$logs_dir/parse_decompiled_errors"

    python parse_dwarf.py $bin_dir"/$binname" --save_dir=$debuginfo_subprograms_dir


    if [ -n "$field_flag" ]; then
        python init_align.py $decompiled_vars_dir $debuginfo_subprograms_dir $decompiled_files_dir $align_var $train_var --bin $binname >> "$source_dir/logs/align_errors"


        echo "#!/bin/bash" > "$commands_dir/$binname"_command.sh
        python gen_command.py "$decompiled_files_dir" "$source_dir" --bin "$binname" ${reason_flag:+--reason} >> "$commands_dir/${binname}_command.sh"
        bash "$commands_dir/$binname"_command.sh >> "$logs_dir/clang_errors" 2>&1

        python align_field.py $align_var $field_access_dir $align_field  $train_field --bin $binname >> $logs_dir/align_field_errors
    else
        python init_align.py $decompiled_vars_dir $debuginfo_subprograms_dir $decompiled_files_dir $align_var  $train_var --bin $binname --ignore_complex >> "$source_dir/logs/align_errors"
    fi

    echo "$binname" >> "$donefiles"
}


process_file_test_only(){
    local FILE=$1
    binname=$(basename "$FILE")
    
    # Check if the corresponding decompiled file exists, if not, skip processing
    if [ ! -f "$decompiled_dir/$binname.decompiled" ]; then
        return
    fi
    
    python prep_decompiled.py "$decompiled_dir/$binname.decompiled" $decompiled_files_dir $decompiled_vars_dir >> "$logs_dir/parse_decompiled_errors"

    if [ -n "$field_flag" ]; then
        echo "#!/bin/bash" > "$commands_dir/$binname"_command.sh
        python gen_command.py "$decompiled_files_dir" "$source_dir" --bin "$binname" ${reason_flag:+--reason} >> "$commands_dir/${binname}_command.sh"
        bash "$commands_dir/$binname"_command.sh >> "$logs_dir/clang_errors" 2>&1
        python gen_train_field_test_mode.py $decompiled_files_dir $field_access_dir $train_field  --bin $binname
    fi

    python gen_train_var_test_mode.py $decompiled_files_dir $decompiled_vars_dir $train_var  --bin $binname

    echo "$binname" >> "$donefiles"
}

# Create an array of all files in the bin directory
files=($bin_dir/*)
jobs_cntr=0
jobs_to_run_num=${#files[@]}
donefiles=$source_dir/"completed_files"
rm -f $donefiles
touch $donefiles

# Main processing loop with multithreading logic
while ((jobs_cntr < jobs_to_run_num)); do
    # Get the current number of running jobs
    cur_jobs_num=$(wc -l < <(jobs -r))

    # Get the current file to process
    current_file="${files[jobs_cntr]}"

    # Check if the file has already been processed
    if ! grep -q -F "$current_file" "$donefiles"; then
        if ((cur_jobs_num < MAX_PROC)); then
            echo "=== Progress: $jobs_cntr/$jobs_to_run_num ==="
            if [[ -n "$test_flag" ]]; then
                process_file_test_only "$current_file" &
            else 
                process_file "$current_file" &
            fi
            ((jobs_cntr++))
        else
            sleep "$check_interval"
        fi
    else
        # File already processed, so just increase the counter
        ((jobs_cntr++))
    fi
done

# Wait for all background jobs to complete
wait


if [ -n "$clean_flag" ]; then
    echo "Cleaning intermediate results."
    remove_folder_if_exists $decompiled_files_dir
    remove_folder_if_exists $debuginfo_subprograms_dir
    remove_folder_if_exists $align_var
    remove_folder_if_exists $logs_dir
    remove_folder_if_exists $commands_dir
    remove_folder_if_exists $align_field
    rm $donefiles
    if [ -z "$reason_flag" ]; then
        remove_folder_if_exists $decompiled_vars_dir
        remove_folder_if_exists $field_access_dir
        remove_folder_if_exists $callsite_dir
        remove_folder_if_exists $dataflow_dir
    fi
fi


if [[ -n "$test_flag" ]]; then
    echo "Data processing finished. The results can be found in $train_var and $train_field."
elif [[ -n "$field_flag" ]]; then
    echo "Data processing finished. The results can be found in $train_var and $train_field."
else
    echo "Data processing finished. The results can be found in $train_var"
fi
