import os
import json
import argparse
import random
from utils import *

def extract_binary_name(filename, suffix=".decompiled"):
    return filename.replace(suffix, "") if filename.endswith(suffix) else filename

def ensure_split(split_path, binaries, train_ratio, test_ratio):
    if os.path.exists(split_path):
        print(f"[INFO] Loading existing split from {split_path}")
        return read_json(split_path)

    print("[INFO] Creating new train/test split based on binaries from decompiled folder")
    random.shuffle(binaries)
    total = len(binaries)
    train_count = int(total * train_ratio)
    test_count = int(total * test_ratio)
    split = {
        "train": binaries[:train_count],
        "test": binaries[train_count:train_count + test_count]
    }

    dump_json(split_path, split)
    print(f"[INFO] Saved split to {split_path}")
    return split

def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    split_path = os.path.join(args.output_folder, "split.json")

    # Step 1: Get binary names from decompiled files
    decompiled_files = get_file_list(args.decompiled_folder)
    binaries = [extract_binary_name(f) for f in decompiled_files if f.endswith('.decompiled')]

    # Step 2: Generate or load split
    split = ensure_split(split_path, binaries, args.train, args.test)

    # Step 3: Prepare output files
    train_output_path = os.path.join(args.output_folder, f"{args.model}_train.jsonl")
    test_output_path = os.path.join(args.output_folder, f"{args.model}_test.jsonl")
    train_out = open(train_output_path, 'w')
    test_out = open(test_output_path, 'w')

    # Step 4: Process input JSONs
    input_files = get_file_list(args.input_folder)
    for json_file in input_files:
        if not json_file.endswith(".json"):
            continue

        bin_name = json_file.split('-')[0]
        if bin_name not in binaries:
            continue  # Skip if no corresponding decompiled file exists

        json_path = os.path.join(args.input_folder, json_file)

        json_data = read_json(json_path)
        line = json.dumps(json_data)

        if bin_name in split["train"]:
            train_out.write(line + "\n")
        elif bin_name in split["test"]:
            test_out.write(line + "\n")

    train_out.close()
    test_out.close()
    print(f"[DONE] Wrote:\n  → {train_output_path}\n  → {test_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Folder containing JSON files")
    parser.add_argument("--decompiled_folder", required=True, help="Folder containing .decompiled files")
    parser.add_argument("--output_folder", required=True, help="Folder to write output jsonl files")
    parser.add_argument("--train", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--test", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--model", required=True, help="Model name (e.g., fielddecoder)")

    args = parser.parse_args()
    main(args)