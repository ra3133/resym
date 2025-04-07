import argparse
import os
from utils import *
from tqdm import tqdm
from typing import List, Dict
from gen_train_field import gen_prompt

def main(decompiled_files_dir, field_access_dir, save_dir, target_bin):
    for f in tqdm(get_file_list(field_access_dir), disable=(target_bin)):
        if not f.endswith('.json'):
            continue

        if target_bin and not f.startswith(target_bin):
            continue


        decompiled_file_path = os.path.join(decompiled_files_dir, f.replace('.json', '.c'))
        if not os.path.exists(decompiled_file_path):
            print(f"Cannot find code source file {decompiled_file_path}")
        
        code = read_file(decompiled_file_path, readlines=True)
        code = ''.join(code[1:]).strip()

        field_access_data = read_json(os.path.join(field_access_dir, f))
        expressions = []
        for access in field_access_data:
            if access['expr'] in expressions:
                continue
            expressions.append(access['expr'])
        
        prompt = gen_prompt(code, expressions)
        data = {
            'code': code,
            'prompt': prompt,
        }
        dump_json(os.path.join(save_dir, f), data)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('decompiled_files_dir')
    parser.add_argument('field_access_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--bin', required=False, default=None)
    # parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()
    
    main(args.decompiled_files_dir, args.field_access_dir, args.save_dir, args.bin)