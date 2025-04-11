import argparse
from typing import List, Dict, Union, Set
from utils import *
from vote_utils import get_ida_type, get_ida_size, get_process_file_path
import re
from tqdm import tqdm
from collections import defaultdict

root_data_folder = ""
PREDICT='predict'


def get_funname(symbol):
    # from "fun_addr": "0x4a0ac6" to sub_4A0AC6
    assert re.match(r'0x[\d\w]+', symbol)
    return f'sub_{symbol[2:].upper()}'

def parse_pred(inference, ground_truth, model_type:str, skip_arr:bool=False) -> (Union[Dict, None], Dict, List):
    def _parse_heap_pred():
        vars_gt = {}
        vars_pred = {}
        skip_expr = set()

        if ground_truth:
            try:
                for var in ground_truth.strip().split('\n'):
                    org, new = var.split(': ')
                    c1, c2 = new.split('->')
                    tmp = c1.strip().split(', ') + c2.strip().split(', ')
                    if tmp[2].strip() == '-' and skip_arr:
                        skip_expr.add(org)
                        continue
                    vars_gt[org] = tmp
                    assert len(vars_gt[org]) == 4
            except:
                vars_gt = {}

        try:
            for var in inference.strip().split('\n'):
                org, new = var.split(':')
                if org in skip_expr:
                    continue
                c1, c2 = new.strip().split('->')
                tmp = c1.strip().split(',') + c2.strip().split(',')
                vars_pred[org] = [t.strip() for t in tmp]
                assert len(vars_pred[org]) == 4
        except:
            vars_pred = {}
        
        return vars_pred, vars_gt, []


    def _parse_stack_pred():
        vars_gt = {}
        gt_var_order = []   # variable order
        vars_pred = {}
        
        if ground_truth:
            gt_mismatch = False
            for var in ground_truth.strip().split('\n'):
                varname, labels = var.split(': ')
                vars_gt[varname] = labels.split(', ')
                gt_var_order.append(varname)
                if len(vars_gt[varname]) != 2:
                    gt_mismatch = True
                    break
            if gt_mismatch:
                vars_gt = {}

        try:
            for var in inference.strip().split('\n'):
                varname, labels = var.split(':')
                tmp = labels.strip().split(', ')
                vars_pred[varname.strip()] = [t.strip() for t in tmp]
                assert len(vars_pred[varname]) == 2
        except:
            vars_pred = {}

        return vars_pred, vars_gt, gt_var_order



    if model_type=='heap':
        return _parse_heap_pred()
    elif model_type == 'stack':
        return _parse_stack_pred()


def collect_calls(merged_inference_data):
    def _get_train_fun_id(fundata):
        if 'stack' in fundata:
            return fundata['stack']['bin'], fundata['stack']['fun_id']
        elif 'heap' in fundata:
            return fundata['heap']['bin'], fundata['heap']['fun_id']
        else:
            assert False, False

    for proj in tqdm(merged_inference_data, desc='collect calls'):
        for fun, fundata in merged_inference_data[proj].items():
            binname, funid = _get_train_fun_id(fundata)
            callsite_fpath = get_process_file_path(root_data_folder, "callsite", binname, funid)
            if not callsite_fpath:
                continue
            callsite_data = read_json(callsite_fpath)
            
            for call in callsite_data:
                callee = call['funName']

                merged_inference_data[proj][fun]['callee'].append({'fun': callee, 'args':call['args']})

                if callee not in merged_inference_data[proj]:
                    continue
                merged_inference_data[proj][callee]['caller'].append({'fun': fun, 'args':call['args']})
    return merged_inference_data

def process_dataflow(dataflow_fpath: str) -> Dict[str, Set]:
    # return {var1 : [variables equals to var1 ], ...}
    ret = {}
    if not dataflow_fpath:
        return ret
    
    dataflow_data = read_json(dataflow_fpath)
    for assignment in dataflow_data:
        lhs, rhs = assignment['lhsVarName'], assignment['rhsVarName']
        if lhs in ret:
            ret[lhs].add(rhs)
        elif rhs in ret:
            ret[rhs].add(lhs)
        else:
            ret[lhs] = set()
            ret[lhs].add(rhs)

    # convert to list
    for var in ret:
        ret[var] = list(ret[var])
        
    return ret

def get_arg_var_info(binname, fun_id) -> (List, List):
    decompiled_vars_fpath = get_process_file_path(root_data_folder, "decompiled_vars", binname, fun_id)
    if decompiled_vars_fpath is None: 
        return None, None
    decompiled_vars_data = read_json(decompiled_vars_fpath)
    args = [a['name'] for a in decompiled_vars_data['argument']]
    vars = [v['name'] for v in decompiled_vars_data['variable']]
    return args, vars 

def get_ida_type_helper(binname, fun_id, varnames) -> Dict[str, str]:
    ret = {}
    for var in varnames:
        decompiled_vars_fpath = get_process_file_path(root_data_folder, "decompiled_vars", binname, fun_id)
        ida_type = get_ida_type(decompiled_vars_fpath, var)
        ret[var] = ida_type
    return ret

def get_ida_size_helper(binname, fun_id, varnames, ida_type_config) -> Dict[str, str]:
    ret = {}
    for var in varnames:
        decompiled_vars_fpath = get_process_file_path(root_data_folder, "decompiled_vars", binname, fun_id)
        ida_size = get_ida_size(decompiled_vars_fpath, var, ida_type_config)
        ret[var] = ida_size
    return ret

def merge_stack_heap_inference(stack_fpath, heap_fpath, stack_fun_by_bin, heap_fun_by_bin, test_mode:bool):
    ida_type_config = read_json('./config/ida_types.json')
    save_data = {}
    # iter 1, stack inference, init all "proj" and "fun" info
    with open(stack_fpath, 'r') as fp:
        fp_lines = fp.readlines()
        for bin, indices in tqdm(stack_fun_by_bin.items(), desc='iterate stack'):
            save_data[bin] = {}
            for i in indices:
                line = json.loads(fp_lines[i])
                if test_mode:
                    # in testing mode, no ground truth to be collected
                    vars_pred, vars_gt, gt_var_order = parse_pred(line[PREDICT], "", model_type='stack')
                else:
                    vars_pred, vars_gt, gt_var_order = parse_pred(line[PREDICT], line['output'], model_type='stack')
                # get dataflow info
                dataflow_fpath = get_process_file_path(root_data_folder, "dataflow", line['bin'], line['fun_id'])
                dataflow_data = process_dataflow(dataflow_fpath)
                args, vars = get_arg_var_info(line['bin'], line['fun_id'])
                if args is None:
                    continue
                save_data[bin]["sub_"+line['fun_id']] = {
                    'stack': {
                        'index': i,
                        'inference': vars_pred, 
                        'ground_truth': vars_gt, 
                        'order': gt_var_order,
                        'bin': line['bin'],
                        'fun_id': line['fun_id'],
                        'cluster_var': line.get('cluster_var', {})
                        },
                    'caller': [],
                    'callee': [],
                    'dataflow': dataflow_data,
                    'argument': args,
                    'variable': vars
                    }
                
    # iter 2, heap inference, merge info into stack info
    with open(heap_fpath, 'r') as fp:
        fp_lines = fp.readlines()
        for bin, indices in tqdm(heap_fun_by_bin.items(),desc='iterate heap'):
            if bin not in save_data:
                save_data[bin] = {}

            for i in indices:
                line = json.loads(fp_lines[i])
                if test_mode:   
                    # in testing mode, no ground truth to be collected
                    vars_pred, vars_gt, _ = parse_pred(line[PREDICT], "", model_type='heap', skip_arr=True)
                else:
                    vars_pred, vars_gt, _ = parse_pred(line[PREDICT], line['output'], model_type='heap', skip_arr=True)
                filed_access_fpath = get_process_file_path(root_data_folder, "field_access", line['bin'], line['fun_id'])
                if not filed_access_fpath:
                    continue
                
                filed_access_data = read_json(filed_access_fpath)
                parsed = {}
                seen_access = set()
                for access in filed_access_data:
                    key = (access['varName'], access['offset'])
                    if key in seen_access:
                        continue
                    seen_access.add(key)
                    calculated_offset = max(1, int(access['lhsPointeeSize'])) * int(access['offset'])
                    parsed[access['expr']] = {
                        'calculated_offset': calculated_offset,
                        'varName':access['varName'],
                        'exprPointeeSize': access['exprPointeeSize']
                        }
                if "sub_"+line['fun_id'] not in save_data[bin]:
                    args, vars = get_arg_var_info(line['bin'], line['fun_id'])
                    if args is None:
                        continue
                    dataflow_fpath = get_process_file_path(root_data_folder, "dataflow", line['bin'], line['fun_id'])
                    dataflow_data = process_dataflow(dataflow_fpath)
                    save_data[bin]["sub_"+line['fun_id']] = {
                        'caller': [], 
                        'callee': [], 
                        'dataflow': dataflow_data, 
                        'argument': args, 
                        'variable': vars}
                save_data[bin]["sub_"+line['fun_id']]['heap'] = {
                    'index': i,
                    'inference': vars_pred, 
                    'ground_truth': vars_gt, 
                    'parsed': parsed,
                    'bin': line['bin'],
                    'fun_id': line['fun_id']
                    }
    
    for bin, bindata in tqdm(save_data.items(), desc='get relevant info'):
        for fun, fundata in bindata.items():
            all_vars = set()
            fpath = ''
            if 'stack' in fundata:
                binname = fundata['stack']['bin']
                fun_id = fundata['stack']['fun_id']
            elif 'heap' in fundata:
                binname = fundata['heap']['bin']
                fun_id = fundata['heap']['fun_id']
            else:
                assert False, fun
            all_vars.update(fundata['variable'])
            all_vars.update(fundata['argument'])

            save_data[bin][fun]['ida_type'] = get_ida_type_helper(binname, fun_id, all_vars)
            save_data[bin][fun]['ida_size'] = get_ida_size_helper(binname, fun_id, all_vars, ida_type_config)
    return save_data
               
def main(stack_fpath, heap_fpath, tmp_root_data_folder, save_dir, test_mode):
    
    def _get_fun_by_bin(fpath):
        
        fun_by_bin:Dict[str:List[int]] = defaultdict(list)
        selected_bin = set()
        skipped_bin = set()
        with open(fpath, 'r') as fp:
            for i, line in enumerate(fp.readlines()):
                line = json.loads(line)
                # author, proj = line['proj'].split('/')
                binname = line['bin'] 
                if not os.path.exists(os.path.join(root_data_folder, 'bin', binname)):
                    skipped_bin.add(binname)
                    continue
                # if binname not in [b.strip() for b in read_file('/home/data/completed_files')]:
                #     continue
                selected_bin.add(binname)
                funid = line['fun_id']
                sep = '**'
                # key = f'{author}{sep}{proj}{sep}{binname}'
                key = f'{binname}'
                fun_by_bin[key].append(i)

        return fun_by_bin, selected_bin, skipped_bin

    global root_data_folder
    root_data_folder = tmp_root_data_folder

    init_folder(save_dir)
    stack_fun_by_bin, selected_bin, skipped_bin = _get_fun_by_bin(stack_fpath)
    heap_fun_by_bin, _, _ = _get_fun_by_bin(heap_fpath)
    print(f"Selected {len(selected_bin)} bins. ")
    print(f"Skipped {len(skipped_bin)} bins because they are not in {os.path.join(root_data_folder, 'bin')}")
    merged_data = merge_stack_heap_inference(stack_fpath, heap_fpath, stack_fun_by_bin, heap_fun_by_bin, test_mode)
    merged_data_with_calls = collect_calls(merged_data)
    for bin in merged_data_with_calls:
        dump_json(os.path.join(save_dir, bin+".json"), merged_data_with_calls[bin])




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stack_fpath', help='the file path of inference results (predict.jsonl)')
    parser.add_argument('heap_fpath', help='the file path of inference results (predict.jsonl)')
    parser.add_argument('root_data_folder', help='The folder of all the intermediate results of data pre-processing')
    parser.add_argument('save_dir')
    parser.add_argument("--test", action="store_true", help="Enable test mode")
    args = parser.parse_args()
    main(args.stack_fpath, args.heap_fpath, args.root_data_folder, args.save_dir, args.test)

