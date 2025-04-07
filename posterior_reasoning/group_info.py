import argparse
from typing import List, Dict, Union, Set
from tqdm import tqdm
from utils import *
from vote_utils import *
from collections import defaultdict 
import shutil


def get_ground_truth(data_root, bin, fun_id, var, pred_ptr, ida_type_config):

    ground_truth_data = {}
    align_fpath = get_process_file_path(data_root, "align", bin, fun_id)
    stack_layout, heap_layout, is_gt_ptr, _, gt_type = get_gt_layout(align_fpath, var.strip('&'), ida_type_config, consider_arr = True)

    if not pred_ptr:
        return stack_layout, gt_type
    else:
        return heap_layout, gt_type


def get_fun_clusters(funaddr:str, funstackdata:Dict, order:List)-> List[List]:
    last_not_dash = None
    curr_cluster = []
    cluster = []
    for var in order:
        # print(var)
        if var not in funstackdata:
            print(f"No inference of {funaddr}---{var} is found.")
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
    if curr_cluster:
        cluster.append(curr_cluster)
    return cluster


def get_vote(fundata, var, funaddr, type_config, test_mode):
    def _relevant_cluster(all_cluster, var):
        for cluster in all_cluster:
            if var == cluster[0]:  # only when var is the head
                return cluster
        return None

    if 'stack' in fundata:
        cluster_pred = get_fun_clusters(funaddr, fundata['stack']['inference'], fundata['stack']['order'])
        cluster_gt = []
        if not test_mode:
            cluster_gt = get_fun_clusters(funaddr, fundata['stack']['ground_truth'], fundata['stack']['order'])
    else:
        cluster_pred, cluster_gt = [], []
        

    if var.startswith('&'):
        var = var.strip('&')
    vote_data = {}

    # stack prediction
    if 'stack' in fundata and fundata['stack']['inference'] and var in fundata['stack']['inference']:
        stack_offsets:List[List[int]] = []   # list of <offset, size> pair

        vote_data['ida_type'] = fundata['ida_type'][var]
        vote_data['ida_size'] = fundata['ida_size'][var]

        vote_data['stack'] = fundata['stack']['inference'][var]
        if not test_mode:
            vote_data['stack_gt'] = fundata['stack']['ground_truth'][var]

        pred_size = get_pred_size(fundata['stack']['inference'][var][1], type_config)
        vote_data['pred_size'] = pred_size

        vote_data['bin'] = fundata['stack']['bin']
        vote_data['fun_id'] = fundata['stack']['fun_id']


        var_cluster_pred = _relevant_cluster(cluster_pred, var)
        
        var_cluster_gt = []
        if not test_mode:
            var_cluster_gt = _relevant_cluster(cluster_gt, var)
        if var_cluster_pred:
            vote_data['cluster'] = [[v, fundata['ida_type'][v][0], fundata['ida_size'][v]] for v in var_cluster_pred]
            curr_offset = 0
            for c in vote_data['cluster']:
                stack_offsets.append([curr_offset, c[2]])
                curr_offset += c[2]
        else:
            if pred_size > 0:
                stack_offsets.append([0, pred_size])
            else:
                stack_offsets.append([0, vote_data['ida_size']])
        if var_cluster_gt:
            vote_data['cluster_gt'] = [[v, fundata['ida_type'][v][0], fundata['ida_size'][v]] for v in var_cluster_gt]

        vote_data['stack_offsets'] = stack_offsets
    # heap prediction
    heap_data = {}
    # fundata['heap']['inference'] could be None if mismatch
    if 'heap' in fundata and fundata['heap']['inference']:
        heap_offsets:List[List[int]] = []   # list of <offset, size> pair
        for heap_expr in fundata['heap']['parsed']:
            if heap_expr not in fundata['heap']['inference']:
                # some of the expr were removed due to duplicated (var, offset) pairs
                continue
            if var == fundata['heap']['parsed'][heap_expr]['varName']:
                offset = int(fundata['heap']['parsed'][heap_expr]['calculated_offset'])
                # print(fundata['heap'])
                heap_data[offset] = {}
                heap_data[offset]['pred'] = fundata['heap']['inference'][heap_expr]
                heap_data[offset]['ida_size'] = fundata['heap']['parsed'][heap_expr]['exprPointeeSize']
                heap_data[offset]['pred_size'] = get_pred_size(fundata['heap']['inference'][heap_expr][3], type_config)
                # heap_data[offset]['pred_size'] = 
                if heap_data[offset]['pred_size'] > 0:
                    heap_offsets.append([offset, heap_data[offset]['pred_size']])
                else:
                    heap_offsets.append([offset, heap_data[offset]['ida_size']])

    if heap_data:
        vote_data['heap'] = heap_data
        vote_data['heap_offsets'] = heap_offsets 
        vote_data['bin'] = fundata['heap']['bin']
        vote_data['fun_id'] = fundata['heap']['fun_id']

    return vote_data

def process_votes(all_vote_data) -> (Dict, int, int):
    def _inc_cnt(d, key):
        if key not in d:
            d[key] = 0
        d[key] += 1 
        return d


    processed_layout = {}
    for funvar, vote in all_vote_data.items():
        fun, var = funvar.split('---')
        if 'stack' in vote:
            if 'cluster' in vote and var.replace('&', '') == vote['cluster'][0][0] :
                
                curr_offset = 0
                processed_layout[curr_offset] = defaultdict(dict)
                _inc_cnt(processed_layout[curr_offset], vote['ida_size'])

                curr_offset = vote['ida_size']


                for field in vote['cluster']:
                    if curr_offset not in processed_layout:
                        processed_layout[curr_offset] = defaultdict(dict)
                    _inc_cnt(processed_layout[curr_offset], field[2])
                    curr_offset += field[2]
        if 'heap' in vote:
            for offset, heap_pred in vote['heap'].items():
                if offset not in processed_layout:
                    processed_layout[offset] = defaultdict(dict)
                _inc_cnt(processed_layout[offset], heap_pred['ida_size'])

    return processed_layout 
        
def main(equiv_vars_dir, prep_dir, data_root,  save_dir, test_mode):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    init_folder(save_dir)

    ida_type_config = read_json('./config/ida_types.json')
    type_config = read_json('./config/base_types.json')
    # for each project
    for proj_equiv_var_fname in tqdm(get_file_list(equiv_vars_dir)):
        projname = proj_equiv_var_fname.replace('.json',  '')
        print(f"---------------------{projname}---------------------")
        
        try:
            proj_prep_data = read_json(os.path.join(prep_dir, proj_equiv_var_fname))
            proj_equiv_vars = read_json(os.path.join(equiv_vars_dir, proj_equiv_var_fname))
        except:
            print(f"Prep and equiv vars file not found, Skip.")
            continue
        if not proj_equiv_vars:  # empty list
            continue 
        # print(os.path.join(save_dir, projname))
        init_folder(os.path.join(save_dir, projname))
    
        # for each groups of equivalent variables
        for represent, equiv_var_list in proj_equiv_vars.items():
            group_data = {}

            # 1. get the ground truth if not in test mode
            if not test_mode:
                use_idx = 0
                while use_idx < len(equiv_var_list):
                    if use_idx == len(equiv_var_list):
                        print(f"Fail to get ground truth for cluster {represent}")
                        group_data['ground_truth'] = {}
                        break
                    first_funvar = equiv_var_list[use_idx]
                    funname, var = first_funvar.split('---')
                    if funname not in proj_prep_data:
                        # some functions are only in callgraph (being called) but not in the training/testing set
                        print(f"{funname} not in {os.path.join(prep_dir, proj_equiv_var_fname)}")
                        use_idx += 1
                        continue
                    # if var.strip('&') not in proj_prep_data[funname]['stack']['inference']:
                    #     use_idx += 1
                    #     continue

                    if 'stack' in proj_prep_data[funname]:
                        bin, funid = proj_prep_data[funname]['stack']['bin'], proj_prep_data[funname]['stack']['fun_id']
                    else:
                        bin, funid = proj_prep_data[funname]['heap']['bin'], proj_prep_data[funname]['heap']['fun_id']
                    
                    try:
                        pred_ptr = 'stack' not in proj_prep_data[funname] or is_pred_ptr(proj_prep_data[funname]['stack']['inference'][var.strip('&')][1])
                    except:
                        use_idx += 1
                        continue
                    ground_truth, gt_type = get_ground_truth(data_root, bin, funid, var, pred_ptr, ida_type_config)
                    if not ground_truth:
                        # no ground truth. reason: the var has no aligned info
                        use_idx += 1
                        continue
                    else:
                        group_data['ground_truth'] = {
                            'layout': ground_truth, 
                            'type': gt_type, 
                            'source': first_funvar,
                        }
                        break


                if use_idx == len(equiv_var_list):
                    # none of the var in the list has ground truth
                    continue 

            # 2. get info for each vote
            group_data['votes'] = {}
            num_vote = 0 
            for funvar in equiv_var_list:
                # print(funvar)
                funaddr, var = funvar.split('---')
                if funaddr not in proj_prep_data:
                    continue 
                
                # print(funaddr)
                vote_data = get_vote(proj_prep_data[funaddr], var, funaddr, type_config, test_mode)
                if vote_data:  # if not empty
                    group_data['votes'][funvar] = vote_data
                    num_vote += 1

            if num_vote > 1:
                group_data['num_vote'] = num_vote
                group_data['processed_layout'] = process_votes(group_data['votes'])
                if group_data['processed_layout']:
                    # only save those have layout votes (heap votes)
                    dump_json(os.path.join(save_dir, projname, represent+'.json'), group_data)

            

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('equiv_vars_dir')
    parser.add_argument('prep_dir')
    parser.add_argument('data_root')
    parser.add_argument('save_dir')
    parser.add_argument("--test", action="store_true", help="Enable test mode")
    args = parser.parse_args()
    
    main(args.equiv_vars_dir, args.prep_dir, args.data_root, args.save_dir, args.test)
