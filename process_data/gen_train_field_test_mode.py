import argparse
import os
from utils import *
from tqdm import tqdm
from typing import List, Dict
from gen_train_field import gen_prompt

def main(decompiled_files_dir, field_access_dir, save_dir, target_bin):
    # metadata = read_json(metadata_fpath)
    for f in tqdm(get_file_list(field_access_dir), disable=(target_bin)):
        if not f.endswith('.json'):
            continue

        if target_bin and not f.startswith(target_bin):
            continue

        binname, fun_id = f.replace('.json', '').split('-')
        decompiled_file_path = os.path.join(decompiled_files_dir, f"{binname}-{fun_id}.c")
        if not os.path.exists(decompiled_file_path):
            print(f"Cannot find code source file {decompiled_file_path}")
        
        code = read_file(decompiled_file_path, readlines=True)
        code = ''.join(code[1:]).strip()

        # proj = find_proj(metadata, binname)
        # if proj is None:
        #     print(f"Error: cannot find project for binary {binname} in the metadata")
        #     continue

        field_access_data = read_json(os.path.join(field_access_dir, f))
        expressions = []
        for access in field_access_data:
            if access['expr'] in expressions:
                continue
            expressions.append(access['expr'])
        
        if len(expressions)==0:
            continue 
        
        prompt, first_token = gen_prompt(code, expressions)
        data = {
            'code': code,
            'prompt': prompt,
            'first_token': first_token,
            'bin': binname,
            'fun_id': fun_id,
            # 'proj': proj
        }
        dump_json(os.path.join(save_dir, f), data)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('decompiled_files_dir')
    parser.add_argument('field_access_dir')
    # parser.add_argument('metadata_fpath')
    parser.add_argument('save_dir')
    parser.add_argument('--bin', required=False, default=None)
    # parser.add_argument('--test_mode', action='store_true')
    args = parser.parse_args()
    
    main(args.decompiled_files_dir, args.field_access_dir, args.save_dir, args.bin)