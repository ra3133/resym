import argparse
import os
from utils import *
from typing import Dict, List
from tqdm import tqdm

clang_commands = ['field_access']

clang_commands_for_reasoning = ['callsite', 'dataflow']

def main(src_dir, save_dir, target_bin, reason=False):
    target_commands = clang_commands
    if reason:
        target_commands += clang_commands_for_reasoning

    for f in get_file_list(src_dir):
        if not f.endswith(".c"):
            continue
        if target_bin and not f.startswith(target_bin):
            continue
        
        for c in target_commands:
            target = f.replace('.c', '')
            command = f"/home/ReSym/clang-parser/build/{c} {os.path.join(src_dir, f)} {os.path.join(save_dir, c, target+'.json')}"
            print(command)
    
        
        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--bin', required=False, default=None)
    parser.add_argument('--reason', action='store_true')

    args = parser.parse_args()

    main(args.src_dir, args.save_dir, args.bin, args.reason)



