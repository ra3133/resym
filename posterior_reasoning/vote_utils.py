import re
import json
import pprint
from typing import List, Dict, Set, Union
import statistics
import os


def parse_funvar(funvar):
    fun, var = funvar.split('---')
    if var.startswith('&'):
        var = var[1:]
    return fun, var

class ParseError(Exception):
    def __init__(self, msg):
        self.msg = msg

def get_fun_clusters(funstackdata:Dict, order:List)-> List[List]:
    last_not_dash = None
    curr_cluster = []
    cluster = []
    for var in order:
        if var not in funstackdata:
            continue
        if funstackdata[var] == ['-', '-']:
            if not curr_cluster and last_not_dash!=None:
                curr_cluster.append(last_not_dash)
            curr_cluster.append(var)
        else:
            last_not_dash = var
            if curr_cluster:
                cluster.append(curr_cluster)
            curr_cluster = []

    return cluster

    
def get_gt_layout(align_init_fpath, var, ida_type_config, consider_arr=True) -> (Dict, Dict, bool, bool, str):
    def _process_field_type(field_data):
        field_type = field_data['field_attr']['type_name']
        if field_data['field_attr']['array_dims']:
            # when field is an array
            for dim in field_data['field_attr']['array_dims']:
                field_type += f"[{dim}]"

        return field_type

    # return stack_offsets, heap_offsets, whether it is pointer
    def _parse_heap_layout(vardata):
        ret_layout = {}
        if 'aligned' in vardata and  'point_to_struct_fileds' in vardata['aligned']['Attr']['type_attr']:
            curr_size = 0
            for field in vardata['aligned']['Attr']['type_attr']['point_to_struct_fileds']:
                field_size = field['field_attr']['total_size']
                if field_size is None:
                    # print(var, align_init_fpath)
                    break
                ret_layout[curr_size] = {
                    'size': field_size, 
                    'name': field['field_name'],
                    'type': _process_field_type(field)
                    }

                curr_size += field_size
        return ret_layout
    
    def _parse_stack_layout(varname, vardata, idx):
        ret_layout = {}

        if 'aligned' in vardata:
            type_attr = vardata['aligned']['Attr']['type_attr']
            # if it is struct
            if type_attr['is_struct'] and len(type_attr['struct_fields'])>0:
                curr_size = 0
                for field in type_attr['struct_fields']:
                    filed_size = field['field_attr']['total_size']
                    if filed_size is None:
                        break
                    ret_layout[curr_size] = {
                        'size': filed_size,
                        'name': field['field_name'],
                        'type': _process_field_type(field)
                    }
                    curr_size += filed_size
            else:
                if consider_arr and type_attr['array_dims'] and type_attr['base_size']:
                    # only consider the first dimension
                    for i in range(type_attr['array_dims'][0]):
                        ret_layout[i+1] = {
                            'size': type_attr['base_size'],
                            'name': '-',
                            'type': '-'
                            }
                elif type_attr['total_size'] is not None:
                    # print(var, 'reach here')
                    ret_layout[0] = {
                        'size': type_attr['total_size'],
                        'name': vardata['aligned']['Attr']['DW_AT_name'],
                        'type': type_attr['type_name']
                        }

            # else, stack variable, its own size
            if not ret_layout:
                # 1. infer with ida size:
                infer_size = -1
                idatype = vardata['type']
                array_size = int(vardata.get('array_size', 1)) if vardata.get('array_size', 1) else 1
                if '*' in idatype:
                    infer_size = 8
                elif idatype in ida_type_config:
                    infer_size = ida_type_config[idatype] * array_size

                if infer_size>0:
                    ret_layout[0] = {
                        'size': infer_size,
                        'name': '-',
                        'type': '-'
                        }
                    return ret_layout

            
            # 2. use address 
            if not ret_layout and idx>=0 and idx+1 < len(align_init_data['variable']):
                next = align_init_data['variable'][idx+1]
                if align_init_data['variable'][idx]['rbp_offset_dec'] is not None and next['rbp_offset_dec'] is not None:
                    ret_layout[0] = {
                        'size': next['rbp_offset_dec'] - align_init_data['variable'][idx]['rbp_offset_dec'],
                        'name': '-',
                        'type': '-'
                        }
            
        return ret_layout
    
    def _is_gt_ptr(vardata)-> bool:
        if 'aligned' in vardata:
            return vardata['aligned']['Attr']['type_attr']['is_pointer']
        return False

    def _is_gt_arr(vardata)->bool:
        if 'aligned' in vardata:
            return vardata['aligned']['Attr']['type_attr']['is_array']
        return False

    def _get_gt_type(vardata) -> str:
        if 'aligned' in vardata:
            return vardata['aligned']['Attr']['type_attr']['type_name']
        return False

    
    # return stack offsets and heap offsets
    stack_layout = {}
    heap_layout = {}
    is_gt_ptr = False
    align_init_data = read_json(align_init_fpath)
    for argdata in align_init_data['argument']:
        if argdata['name'] == var:
            heap_layout = _parse_heap_layout(argdata)
            stack_layout = _parse_stack_layout(var, argdata, -1)
            is_gt_ptr = _is_gt_ptr(argdata)
            is_gt_arr = _is_gt_arr(argdata)
            gt_type = _get_gt_type(argdata)
            break

    for i, vardata in enumerate(align_init_data['variable']):
        if vardata['name'] == var:
            heap_layout = _parse_heap_layout(vardata)
            stack_layout = _parse_stack_layout(var, vardata, i)
            is_gt_ptr = _is_gt_ptr(vardata)
            is_gt_arr = _is_gt_arr(vardata)
            gt_type = _get_gt_type(vardata)
            break
        
    return stack_layout, heap_layout, is_gt_ptr, is_gt_arr, gt_type


def get_gt_type(align_init_fpath, var) -> Union[str, None]:
    def _get_type(vardata):
        if 'aligned' in vardata:
            return vardata['aligned']['Attr']['DW_AT_type']
        return None

    align_init_data = read_json(align_init_fpath)
    for argdata in align_init_data['argument']:
        if argdata['name'] == var:
            return _get_type(argdata)

    for i, vardata in enumerate(align_init_data['variable']):
        if vardata['name'] == var:
            return _get_type(vardata)
        
    return None


def is_pred_ptr(pred_type: str) -> bool:
    return '*' in pred_type

def is_relevant_stmt(stmt, var, pass_addr):
    # pass_addr is more strict
    # pass_addr=False is super set of pass_addr=True
    if pass_addr:
        pattern = r'[^A-Za-z0-9_]&{}[^A-Za-z0-9_]'.format(re.escape(var))
    else: 
        pattern = r'[^A-Za-z0-9_]{}[^A-Za-z0-9_]'.format(re.escape(var))
        

    if re.search(pattern, stmt):
        return True
    else:
        return False

def is_direct_pass(pass_param_exp):
    if '+' in pass_param_exp or '<<' in pass_param_exp or '>>' in pass_param_exp or '++' in pass_param_exp or '^' in pass_param_exp:
        return False
    if '[' in pass_param_exp.replace('[0]', '**'):
        return False
    return True

def extract_function_name_from_declaration(declaration):
    # Match the function name using regular expressions
    match = re.search(r'(\w+)\s*\(', declaration)
    if not match:
        return None
    return match.group(1)

def preprocess_statement(statement):
    # Remove assignment pattern
    statement = re.sub(r'^\s*\w+\s*=\s*', '', statement)
    
    # Remove trailing semicolon
    statement = re.sub(r';\s*$', '', statement)
    
    if re.search(r'^\s*(if|while)\s*\(', statement):
        # Remove control structures
        statement = re.sub(r'^\s*if\s*\(', '(', statement)
        statement = re.sub(r'^\s*while\s*\(', '(', statement)
        
        # If statement is wrapped with brackets, remove them
        if statement.startswith('(') and statement.endswith(')'):
            statement = statement[1:-1].strip()
    if not statement.endswith(')'):
        statement += ')'
    return statement.strip()

def extract_function_and_args(statement):
    # statement should be preprocessed 
    
    # Match function name and its arguments
    match = re.search(r'(\w+)\s*\((.*)\)', statement, re.DOTALL)
    if not match:
        return None, []

    fun_name = match.group(1)
    args_str = match.group(2)
    
    if len(fun_name)<=2 or fun_name=='return':
        return None, []
    # Split the arguments using a comma, while handling commas inside parentheses
    level = 0
    args = []
    arg_start = 0
    for i, char in enumerate(args_str):
        if char == '(':
            level += 1
        elif char == ')':
            level -= 1
        elif char == ',' and level == 0:
            args.append(args_str[arg_start:i].strip())
            arg_start = i + 1
    args.append(args_str[arg_start:].strip())

    return fun_name, args

def get_pred_size(pred_type:str, type_config:Dict) -> int:
    pred_type = pred_type.replace('const ', '').strip()
    if '*' in pred_type:
        return 8
    if pred_type in type_config:
        return int(type_config[pred_type][2])
    
    return -1 

def get_var_ida_size_from_offset(align_init_data, varname) -> int:
    # it has to be a variable, not argument
    this_idx = -1
    for i, vardata in enumerate(align_init_data['variable']):
        if vardata['name'] == varname:
            this_idx = i

    # assert this_idx >= 0, f'cannot find {varname}'
    if this_idx < 0:
        return -1 

    if this_idx+1 == len(align_init_data['variable']):
        # no next, nothing we can do
        return -1

    if align_init_data['variable'][this_idx]['rbp_offset_dec'] is None:
        # variable is register, no offset
        return -1
    
    if align_init_data['variable'][this_idx+1]['rbp_offset_dec'] is None:
        # not likely
        return -1
    
    return align_init_data['variable'][this_idx+1]['rbp_offset_dec'] - align_init_data['variable'][this_idx]['rbp_offset_dec']

def get_ida_size(decompiled_vars_fpath, varname, ida_type_config) -> int:
    def _get_size_from_type(tt):
        tt = tt.replace('const', '').strip()
        if '*' in tt:
            return 8
        elif tt in ida_type_config:
            return ida_type_config[tt]
        else:
            return -1

    decompiled_vars_data = read_json(decompiled_vars_fpath)
    
    try:
    # if True:
        for arg in decompiled_vars_data['argument']:
            if arg['name'] == varname:
                return _get_size_from_type(arg['type'])
        for var in decompiled_vars_data['variable']:
            if var['name'] == varname:
                # try use offset first
                ida_size_from_offset = get_var_ida_size_from_offset(decompiled_vars_data, varname)
                # print("ida_size_from_offset", ida_size_from_offset)
                if ida_size_from_offset > 0:
                    return ida_size_from_offset
                else:
                    if var['array_size'] is not None:
                        return max(-1, int(var['array_size']) * _get_size_from_type(var['type'])) 
                    else:
                        return _get_size_from_type(var['type'])
    except:
        return -1

def get_ida_type(decompiled_vars_fpath, varname) -> (str, bool):
    try:
        decompiled_vars_data = read_json(decompiled_vars_fpath)
        for var in decompiled_vars_data['argument'] + decompiled_vars_data['variable']:
            if var['name'] == varname:
                if var.get('array_size', None) is not None:
                    return f"{var['type']}[{var['array_size']}]", True
                else:
                    return var['type'], False
       
    except:
        # no `type` (e.g., variable `...`)
        return None, False
    # return None, False  # shouldn't happen

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)

def pprint_dict(d):
    pprint.pprint(d)

def print_stat(data_list: List):
    min_val = min(data_list)
    max_val = max(data_list)
    median = statistics.median(data_list)
    mean = statistics.mean(data_list)
    stdev = statistics.stdev(data_list)
    variance = statistics.variance(data_list)
    print('Len: {}, Min: {}, Max: {}, Median: {}, Average: {}, stdev: {}, Variance: {}'.format(\
        len(data_list), min_val, max_val, median, mean, stdev, variance))


def get_process_file_path(root_data_folder, folder_name, bin_name, fun_id):
    if folder_name == 'decompiled_vars':
        ret_path = os.path.join(root_data_folder, folder_name, f"{bin_name}-{fun_id}_var.json")
    else:
        ret_path = os.path.join(root_data_folder, folder_name, f"{bin_name}-{fun_id}.json")
    if not os.path.exists(ret_path):
        return None
    return ret_path
