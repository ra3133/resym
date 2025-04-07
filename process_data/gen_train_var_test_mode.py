import argparse
import os
from utils import *
from tqdm import tqdm
from typing import List, Dict
from align_stack import gen_prompt

def main(decompiled_files_dir, decompiled_vars_dir, save_dir, target_bin):
    for f in tqdm(get_file_list(decompiled_vars_dir), disable=(target_bin)):
        if not f.endswith('.json'):
            continue

        if target_bin and not f.startswith(target_bin):
            continue

        
        decompiled_vars_path = os.path.join(decompiled_files_dir, f.replace('_var.json', '.c'))
        if not os.path.exists(decompiled_vars_path):
            print(f"Cannot find code source file {decompiled_vars_path}")
        
        code = read_file(decompiled_vars_path, readlines=True)
        code = ''.join(code[1:]).strip()

        variable_data = read_json(os.path.join(decompiled_vars_dir, f))
        vars = [a['name'] for a in variable_data['argument']]
        for var in variable_data['variable']:
            if var['rbp_offset_dec'] is None:
                continue
            vars.append(var['name'])
        
        prompt = gen_prompt(code, vars)
        data = {
            'code': code,
            'prompt': prompt,
        }
        dump_json(os.path.join(save_dir, f), data)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('decompiled_files_dir')
    parser.add_argument('decompiled_vars_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--bin', required=False, default=None)
    # parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()
    
    main(args.decompiled_files_dir, args.decompiled_vars_dir, args.save_dir, args.bin)