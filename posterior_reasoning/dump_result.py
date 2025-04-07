import argparse
from typing import List, Dict, Union, Set, Any
from tqdm import tqdm
from utils import *
from vote_utils import *
import os
from collections import defaultdict



def strip_type(type_str):
    if type_str is None:
        return ""
    type_str = type_str.replace('const ', '')
    type_str = type_str.replace('struct ', '')
    type_str = type_str.replace('*', '')
    return type_str.strip()


class Results():
    def __init__(self):
        self.pred = defaultdict(dict)
        self.gt = defaultdict(dict)

    def update_pred(self, key, offsets:Dict, structname:str):
        offsets = {int(k): int(v) for k, v in offsets.items()}
        self.pred [key] = {
            'offset': offsets,
            'annotation':{
                'structname':structname,
                'filedname': {},
                'filetype': {}
            }
        }

    def update_gt(self, key, offsets:Dict, structname:str):
        # print('Update gt ' ,key)
        offsets = {int(k): int(v) for k, v in offsets.items()}
        self.gt [key] = {
            'offset': offsets,
            'annotation':{
                'structname':structname,
                'filedname': {},
                'filetype': {}
            }
        }

    def update_pred_field(self, key, offset, name, type):
        offset = int(offset)
        assert key in self.pred
        self.pred[key]['annotation']['filedname'][offset] = name
        self.pred[key]['annotation']['filetype'][offset] = type


    def update_gt_field(self, key, offset, name, type):
        offset = int(offset)
        assert key in self.gt
        self.gt[key]['annotation']['filedname'][offset] = name
        self.gt[key]['annotation']['filetype'][offset] = type
    


    def dump_results(self, save_fpath, test_mode):
        save = {}
        for key in self.pred:
            save[key] = {
                'pred': {'type': self.pred[key]['annotation']['structname'], 'offsets': {}}
                }
            for off in self.pred[key]['offset']:
                save[key]['pred']['offsets'][off] = {
                    'size': self.pred[key]['offset'][off], 
                    'name': self.pred[key]['annotation']['filedname'][off],
                    'type': self.pred[key]['annotation']['filetype'][off]
                    }
            
            if not test_mode:
                assert key in self.gt, key + "is missing"
                save[key]['gt'] = {'type': self.gt[key]['annotation']['structname'], 'offsets': {}}
                save[key] = {
                    'pred': {'type': self.pred[key]['annotation']['structname'], 'offsets': {}}, 
                    'gt': {'type': self.gt[key]['annotation']['structname'], 'offsets': {}}
                    }
                # print(self.pred[key])
                for off in self.pred[key]['offset']:
                    save[key]['pred']['offsets'][off] = {
                        'size': self.pred[key]['offset'][off], 
                        'name': self.pred[key]['annotation']['filedname'][off],
                        'type': self.pred[key]['annotation']['filetype'][off]
                        }
                for off in self.gt[key]['offset']:
                    save[key]['gt']['offsets'][off] = {
                        'size': self.gt[key]['offset'][off], 
                        'name': self.gt[key]['annotation']['filedname'][off],
                        'type': self.gt[key]['annotation']['filetype'][off]
                        }
        dump_json(save_fpath, save)


# get_fpath_from_train_prefix = '/local2/xie342/shared/dataset'
def eval(prep_folder, eval_folder, data_root_folder, out_fpath, test_mode:bool, skip_arr=True):

    def _get_fun_id(fundata):
        if 'stack' in fundata:
            return fundata['stack']['bin'], fundata['stack']['fun_id']
        elif 'heap' in fundata:
            return fundata['heap']['bin'], fundata['heap']['fun_id']
        else:
            assert False, False

    def _eval_stack(fun, fundata):
        bin, fun_id = _get_fun_id(fundata)
        array_clusters = fundata['stack']['cluster_var'].get('array', [])
        arr_heads = [c[0] for c in array_clusters]


        cluster_pred = get_fun_clusters(fundata['stack']['inference'], fundata['stack']['order'])
        cluster_pred = {c[0]: c for c in cluster_pred}  # {head : cluster (list)}

        if not test_mode:
            cluster_gt = get_fun_clusters(fundata['stack']['ground_truth'], fundata['stack']['order'])
            cluster_gt = {c[0]: c for c in cluster_gt if c not in arr_heads and skip_arr}
        
        tmp_seen_var = set()

        
        for head_pred in cluster_pred:

            key = f'{filename}**{fun}**{head_pred}'
            if key in seen_var:
                continue
            tmp_seen_var.add(key) 

            if skip_arr and head_pred in arr_heads:
                continue
            
            pred_layout = {}
            curr_offset = 0
            
            for v in cluster_pred[head_pred]:
                pred_layout[curr_offset] = fundata['ida_size'][v]
                curr_offset += fundata['ida_size'][v]
            pred_type = fundata['stack']['inference'][head_pred][1]
            our_results.update_pred(key, pred_layout, pred_type)
            for offset in pred_layout:
                our_results.update_pred_field(key, offset, "-", "-")

            if not test_mode:
                align_fpath = get_process_file_path(data_root_folder, "align", bin, fun_id)
                if not os.path.exists(align_fpath):
                    assert False, align_fpath
                stack_layout, _, _, is_gt_arr, _ = get_gt_layout(align_fpath, head_pred, ida_type_config)
                gt_type = get_gt_type(align_fpath, head_pred)
                if skip_arr and is_gt_arr:
                    continue

                if head_pred not in cluster_gt:
                    our_results.update_gt(key, {k:v['size'] for k, v in stack_layout.items()}, fundata['stack']['ground_truth'][head_pred][1])
                    for offset in stack_layout:
                        our_results.update_gt_field(key, offset, stack_layout[offset]['name'], stack_layout[offset]['type'])

        if test_mode:
            return
        
        for head_gt in cluster_gt:
            # if head_gt in cluster_pred:
            #     continue

            if skip_arr and head_gt in arr_heads:
                continue

            key = f'{filename}**{fun}**{head_gt}'
            if key in seen_var:
                continue
            tmp_seen_var.add(key)

            
            align_fpath = get_process_file_path(data_root_folder, "align", bin, fun_id)
            if not os.path.exists(align_fpath):
                assert False, align_fpath
            stack_layout, _, _, is_gt_arr, _ = get_gt_layout(align_fpath, head_gt, ida_type_config)
            if is_gt_arr:
                continue

            pred_layout = {}

            try:
                pred_size = get_pred_size(fundata['stack']['inference'][head_gt][1], type_config)
            except:
                pred_size = fundata['ida_size'][head_gt]

            if pred_size > 0:
                pred_layout[0] = pred_size
            else:
                pred_layout[0] = fundata['ida_size'][head_gt]

            our_results.update_gt(key, {k:v['size'] for k, v in stack_layout.items()}, fundata['stack']['ground_truth'][head_gt][1])
            if head_gt not in cluster_pred:
                our_results.update_pred(key, pred_layout, fundata['stack']['inference'].get(head_gt, ['-', '-'])[1])  # may not exist?
                pred = fundata['stack']['inference'].get(head_gt, ['-', '-'])
                our_results.update_pred_field(key, 0, pred[0], pred[1])



        # update name evaluator
        for head_gt in cluster_gt:
            key = f'{filename}**{fun}**{head_gt}'
            
            if key in seen_var:
                # print(f"seen {key}")
                continue
            seen_var.add(key) 
            # print(key)

            if skip_arr and head_gt in arr_heads:
                continue

            align_fpath = get_process_file_path(data_root_folder, "align", bin, fun_id)
            if not os.path.exists(align_fpath):
                assert False, align_fpath
            stack_layout, _, _, is_gt_arr, _ = get_gt_layout(align_fpath, head_gt, ida_type_config)
            if skip_arr and is_gt_arr:
                continue

            for offset in stack_layout:
                our_results.update_gt_field(key, offset, stack_layout[offset]['name'], stack_layout[offset]['type'])


    def _eval_heap(fun, fundata):
        # ----------- 1. get pred offsets for all vars ------------
        heap_pred_offsets = defaultdict(dict) # {var:{offset:size}}
        var2oneExpr = {}  # {var: one of the expr}
        for expr, expr_data in fundata['heap']['parsed'].items():
            if expr not in fundata['heap']['inference']:
                continue
            var = expr_data['varName']
            offset = expr_data['calculated_offset']
            ida_size = expr_data['exprPointeeSize']
            # print(os.path.join(prep_folder, projfile), fun)
            var2oneExpr[var] = expr

            # print(fundata['heap']['inference'][expr])
            pred_size = get_pred_size(fundata['heap']['inference'][expr][3], type_config)
            size = pred_size if pred_size>0 else ida_size

            heap_pred_offsets[var][int(offset)] = {
                'size': int(size), 
                'name': fundata['heap']['inference'][expr][2],
                'type': fundata['heap']['inference'][expr][3],
                }

        # ------- 2. for each var, get gt and compare ---------
        for var in heap_pred_offsets:
            key = f'{filename}**{fun}**{var}'
            
            if key in seen_var:
                # print(f"seen {key}")
                continue
            seen_var.add(key)

            pred_offsets = {k:v['size'] for k, v in heap_pred_offsets[var].items()}
            pred_type_name = fundata['heap']['inference'][var2oneExpr[var]][1]
            our_results.update_pred(key, pred_offsets, pred_type_name)

            for offset in heap_pred_offsets[var]:
                pred_name = heap_pred_offsets[var][offset]['name']
                pred_type = heap_pred_offsets[var][offset]['type']
                our_results.update_pred_field(key, offset, pred_name, pred_type)




            if not test_mode:
                bin, fun_id = _get_fun_id(fundata)
                align_fpath = get_process_file_path(data_root_folder, "align", bin, fun_id)
                if not os.path.exists(align_fpath):
                    assert False, align_fpath

                _, heap_layout, is_gt_ptr, _, _ = get_gt_layout(align_fpath, var, ida_type_config)
                if skip_arr and is_gt_ptr and not heap_layout:
                    # array
                    continue
                
                gt_offsets = {k:v['size'] for k, v in heap_layout.items()}

                # update name eval
                gt_type_name = fundata['heap']['ground_truth'][var2oneExpr[var]][1]
                our_results.update_gt(key, gt_offsets, gt_type_name)

                for offset in heap_layout:
                    if offset in heap_pred_offsets[var]:
                        pred_name = heap_pred_offsets[var][offset]['name']
                        pred_type = heap_pred_offsets[var][offset]['type']

                    else:
                        pred_name = "-"
                        pred_type = "-"
                    
                    our_results.update_gt_field(key, offset, heap_layout[offset]['name'], heap_layout[offset]['type'])

            
            
    def _eval_group(group_data):
        final_type = group_data['final']['voted_type']['stack_type']
        final_field = group_data['final']['voted_type']['field']
        final_layout = group_data['final']['voted_offsets']
        assert len(final_layout) == len(final_field)

        for funvar, funvardata in group_data['votes'].items():
            fun, var = parse_funvar(funvar)
            # author, proj, binname = binfolder.split('**')

            fun, var = parse_funvar(funvar) 
            key = f'{binfolder}**{fun}**{var}'
            if key in seen_var:
                continue
            seen_var.add(key)


            if not test_mode:
                # funaddr = fun.replace('sub_', '')
                bin, fun_id = funvardata['bin'], funvardata['fun_id']
                align_init_fpath = get_process_file_path(data_root_folder, "align", bin, fun_id) 
                if not os.path.exists(align_init_fpath):
                    # not possible .. 
                    print(f"Cannot find the ground truth file: {align_init_fpath}")
                    continue

                gt_type = get_gt_type(align_init_fpath, var)
                stack_layout, heap_layout, is_gt_ptr, is_gt_arr, _ = get_gt_layout(align_init_fpath, var, ida_type_config, consider_arr=False)
                if skip_arr and is_gt_arr:
                    continue

                stack_offsets = {k : v['size'] for k, v in stack_layout.items()}
                heap_offsets = {k : v['size'] for k, v in heap_layout.items()}

            if 'stack' in funvardata and not is_pred_ptr(funvardata['stack'][1]):

                if not test_mode:
                    if skip_arr and is_gt_arr:
                        continue
                    if not stack_offsets:
                        continue
                    our_results.update_gt(key, stack_offsets, gt_type)

                    for offset in stack_layout:
                        if str(offset) in final_field:
                            pred_name = final_field[str(offset)]['name']
                            pred_type = final_field[str(offset)]['type']
                            
                        else:
                            pred_name = '-'
                            pred_type = '-'

                        our_results.update_gt_field(key, offset, stack_layout[offset]['name'], stack_layout[offset]['type'])


                our_results.update_pred(key, final_layout, final_type)

                for offset in final_field:
                    pred_name = final_field[offset]['name']
                    pred_type = final_field[offset]['type']
                    our_results.update_pred_field(key, offset, pred_name, pred_type)

            else:
                
                if not test_mode:
                    if skip_arr and is_gt_ptr and not heap_layout:
                        continue
                        # assert False, f"{funvar}, {os.path.join(eval_folder, binfolder, group_f)}"
                    
                    our_results.update_gt(key, heap_offsets, gt_type)

                    for offset in heap_layout:
                        
                        if str(offset) in final_field:
                            # print('in')
                            pred_name = final_field[str(offset)]['name']
                            pred_type = final_field[str(offset)]['type']
                            # our_results.update_pred_field(key, offset, pred_name, pred_type)
                        else:
                            pred_name = '-'
                            pred_type = '-'

                        our_results.update_gt_field(key, offset, heap_layout[offset]['name'], heap_layout[offset]['type'])

                our_results.update_pred(key, final_layout, final_type)

                for offset in final_field:
                    pred_name = final_field[offset]['name']
                    pred_type = final_field[offset]['type']
                    our_results.update_pred_field(key, offset, pred_name, pred_type)



    ida_type_config = read_json('./config/ida_types.json')
    type_config = read_json('./config/base_types.json')
    our_results = Results()

    seen_var = set()
    # 1. iterate groups
    for binfolder in tqdm(os.listdir(eval_folder), desc='Eval group'):
        for group_f in get_file_list(os.path.join(eval_folder, binfolder)):
            group_data = read_json(os.path.join(eval_folder, binfolder, group_f))
            _eval_group(group_data)

    # 2. iterate the rest of the prediction
    for projfile in tqdm(get_file_list(prep_folder), desc='The rest of the prediction'):
        if not projfile.endswith('.json'):
            continue
        proj_data = read_json(os.path.join(prep_folder, projfile))
        filename = projfile.replace('.json', '')
        for fun, fundata in proj_data.items():
            if 'stack' in fundata:
                _eval_stack(fun, fundata)
            if 'heap' in fundata:
                _eval_heap(fun, fundata)

    if out_fpath:
        our_results.dump_results(out_fpath, test_mode)
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('prep_folder')
    parser.add_argument('eval_folder')
    parser.add_argument('data_root_folder')
    parser.add_argument("--out", help="out put csv file path", default="")
    parser.add_argument("--test", action="store_true", help="Enable test mode")
    args = parser.parse_args()

    eval(args.prep_folder, args.eval_folder, args.data_root_folder, args.out, args.test)

