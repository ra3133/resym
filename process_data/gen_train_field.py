import argparse
import os
from utils import *
from tqdm import tqdm
from typing import List, Dict

def sort_heap_data(align_heap_data: List[Dict]) -> List[Dict]:
    return sorted(align_heap_data, key=lambda x: (x['varName'], x['offset']))


def rm_dup_from_align_heap_data(align_heap_data: List[Dict]) -> List[Dict]:
    seen = set()
    unique_data = []

    for item in align_heap_data:
        # Create a tuple based on varName and offset
        key = (item['varName'], item['offset'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)

    return unique_data

def process_align_heap_data(align_heap_data):
    ground_truth = []
    visited_expr = set()
    align_heap_data = rm_dup_from_align_heap_data(align_heap_data)
    sorted_align_heap_data = sort_heap_data(align_heap_data)
    for hdata in sorted_align_heap_data:
        expr = hdata['expr']
        if expr in visited_expr:
            continue
        visited_expr.add(expr)
        assert 'aligned' in hdata
        curr = [expr]
        curr.append(hdata['aligned']['varName'])
        curr.append(hdata['aligned']['type'])
        curr.append(hdata['aligned']['fieldName'])
        curr.append(hdata['aligned']['fieldType'])
        ground_truth.append(curr)
    return ground_truth

def gen_data_point(align_heap_data, binname, fun_id):
    code = align_heap_data['code']
    
    ground_truth = process_align_heap_data(align_heap_data['aligned'])

    expressions = [v[0] for v in ground_truth]
    if len(expressions) == 0:
        return None
    prompt, first_token = gen_prompt(code, expressions)

    output = []
    for expr, varname, vartype, fname, ftype in ground_truth:
        output.append(f"{expr}: {varname}, {vartype} -> {fname}, {ftype}")

    output = '\n'.join(output)
    
    return {
        'code': code.strip(),
        'prompt': prompt,
        'output': output,
        'label': ground_truth,
        'funname': align_heap_data['funname'],
        'first_token': first_token,
        'bin': binname,
        'fun_id': fun_id,
        # 'proj': proj
    }
    
def gen_prompt (code:str, expressions:List[str]):
    prompt = 'What are the variable name and type for the following memory accesses: '
    prompt += ', '.join([e for e in expressions]) + '?\n'
    prompt += f'```\n{code.strip()}\n```' 
    prompt += f'{expressions[0]}:'
    return prompt, expressions[0]


def gen_fielddecoder_data(fname, align_heap_data, binname, fun_id, save_dir):
        
    save_fpath = os.path.join(save_dir, fname + '.json')

    save_data = gen_data_point (align_heap_data, binname, fun_id)

    if save_data is not None:
        dump_json(save_fpath, save_data)
