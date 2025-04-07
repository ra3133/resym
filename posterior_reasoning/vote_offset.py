# osprey setting: THRESHOLD_WO_HEAP=1, THRESHOLD_WITH_HEAP=2, PROPOSE_ON=False, MAX_GUESS_SIZE=8, RANDOM_WHEN_DRAW=True, seed=0
# gh setting: THRESHOLD_WO_HEAP=1, THRESHOLD_WITH_HEAP=3, PROPOSE_ON=False, MAX_GUESS_SIZE=8, seed=10, RANDOM_WHEN_DRAW=False
from enum import Enum
import argparse
import json
from typing import List, Dict, Set, Any, Union
import os
from tqdm import tqdm
import sys
sys.path.append('../process_data')
sys.path.append('../scripts')
from utils import *
# from eval_utils import *
from collections import defaultdict 
from vote_utils import *
import pandas as pd
import random

random.seed(9)

# the choice of threshold may affect the results
THRESHOLD_WO_HEAP=1 # or 1
THRESHOLD_WITH_HEAP=2 #3

PROPOSE_ON=False
MAX_GUESS_SIZE=8
RANDOM_WHEN_DRAW = True
class RejReason(Enum):
    NeverPassedAsRef = 1
    VETO=2
    CHARPTR=3


class VoteError(Exception):
    pass

def early_reject(heap_offsets:Dict[int, int], has_cluster_predicted:bool) -> bool:
    # if only has one offset of beyond access, and the pointee size is 1. it is most likely a char*
    if not has_cluster_predicted and len(list(heap_offsets.keys())) == 1 and 1 in heap_offsets.values():
        return True
    return False


def guess_offset(offsets:Dict[int, int]) -> Dict[int, int]:
    sorted_offsets = sorted(list(offsets.keys()))
    curr_size = 0
    for off in sorted_offsets:
        if offsets[off] < 0:
            curr_size = off
            continue
        if off == curr_size:
            curr_size += offsets[off]
        elif off < curr_size:
            break
        elif off - curr_size <= MAX_GUESS_SIZE:
            offsets[curr_size] = off - curr_size
            curr_size += offsets[curr_size]
        else:
            curr_size = off
    return offsets


class LayoutVoter():
    def __init__(self):
        # ------------- before voting -------------
        self.heap_offsets:Dict[int, Dict[int,int]] = defaultdict(dict)  # offset -> size -> cnt (num_vote)
        self.stack_offsets:Dict[int, Dict[int,int]] = defaultdict(dict)  # offset -> size -> cnt (num_vote)
        self.total_size_votes:Dict[int, int] = {}


        
        # ------------- finalized  (after voting) -------------
        self.final_heap_offsets:Dict[int, int] = defaultdict(dict)  # offset -> size (vote heap offsets first, and treat as ground truth)

        # the total_size shall exceed max_heap_size (max_offset + its size)
        self.max_heap_size:int = -1
        

        self.final_offsets:Dict[int, int] = defaultdict(dict)  # offset -> size (combined both stack and heap)
        self.final_total_size:int = -1

    def update_heap_offset(self, offset:int, size:int):
        if size not in self.heap_offsets[offset]:
            self.heap_offsets[offset][size] = 0
        self.heap_offsets[offset][size] += 1

    def update_stack_offset(self, offset:int, size:int):
        if size not in self.stack_offsets[offset]:
            self.stack_offsets[offset][size] = 0
        self.stack_offsets[offset][size] += 1
    
    def update_total_size_vote(self, size):
        if size not in self.total_size_votes:
            self.total_size_votes[size] = 0
        self.total_size_votes[size] += 1

    def majority_vote(self, votes: Dict[Any, int], constraint_max=None, constraint_min=None, hints=[]) -> (Union[None, Any], bool):
        # return results, and a bool (if there's a draw)
        def _random_choose(candidate):
            valid_hints = [h for h in hints if h in candidate]
            if valid_hints:
                return random.choice(sorted(valid_hints))
            else:
                return random.choice(sorted(candidate))

        max_vote = 0
        max_vote_val = []
        for val, num_vote in votes.items():
            if constraint_max and val > constraint_max:
                continue
            
            if constraint_min and val < constraint_min:
                continue

            if num_vote < max_vote:
                continue
            elif num_vote > max_vote:
                max_vote = num_vote
                max_vote_val = [val]
            else :
                max_vote_val.append(val)
            
        
        if len(max_vote_val) == 1:
            return max_vote_val[0], False
        elif len(max_vote_val) > 1:
            return _random_choose(max_vote_val), True
            
            
        else: 
            # max_vote_val is empty possibly due to the constraints
            if not constraint_max and not constraint_min:
                return None
            else:
                candidate = []
                if constraint_max:
                    candidate.append(constraint_max)
                if constraint_min:
                    candidate.append(constraint_min)
                return _random_choose(candidate), True

    def vote_heap_offset(self):
        for offset, vote_data in self.heap_offsets.items():
            if len(list(vote_data.keys())) == 1:
                self.final_heap_offsets[offset] = list(vote_data.keys())[0]
            else:
                self.final_heap_offsets[offset], draw = self.majority_vote(vote_data)
                if draw and not RANDOM_WHEN_DRAW:
                    raise VoteError()
        # update self.max_heap_size
        if self.final_heap_offsets:
            max_heap_offset = max(list(self.final_heap_offsets.keys()))
            self.max_heap_size = max_heap_offset + self.final_heap_offsets[max_heap_offset]
        
    def vote_total_size(self):
        if self.total_size_votes:
            self.final_total_size, draw = self.majority_vote(self.total_size_votes, hints = [self.max_heap_size])
            if draw and not RANDOM_WHEN_DRAW:
                raise VoteError()

            if self.final_total_size is None:
                self.final_total_size = -1

    def vote_layout(self):
        '''
        final_offsets init as heap_offsets
        threshold = 2 if heap_offsets not empty else: 1
        for stack_offset not in heap_offsets:
            if num_vote < threshold -> discard
            if stack_offset >= totalsize -> discard
            else: vote_size with constraint of total_size
                  final_offsets[stack_offset] = voted_size 
        '''
        def _vote_stack_offset(offset) -> int:
            # vote whether it exists, if yes, its size
            if offset != 0:
                # return -1 if it doesn't exists
                total_num_vote = sum([self.stack_offsets[offset][size] for size in self.stack_offsets[offset]])
                # if total number of votes < threshold: veto
                if total_num_vote < threshold:
                    return -1
                # if stack_offset >= totalsize -> discard
                if self.final_total_size!= -1 and offset >= self.final_total_size:
                    return -1
           
            # else: majority vote on size
            if self.final_total_size != -1:
                constraint_max = self.final_total_size-offset
            else:
                constraint_max = None
            voted_size, draw = self.majority_vote(self.stack_offsets[offset], constraint_max=constraint_max)
            if draw and not RANDOM_WHEN_DRAW:
                raise VoteError()

            default_size = -1 if offset!=0 else self.final_total_size
            return voted_size if voted_size is not None else default_size
            
        threshold = THRESHOLD_WITH_HEAP if self.final_heap_offsets else THRESHOLD_WO_HEAP
        
        # vote on total size first
        self.vote_total_size()
        # print('self.final_total_size', self.final_total_size)
        self.final_offsets = self.final_heap_offsets.copy()

        for stack_offset in self.stack_offsets:
            if stack_offset not in self.final_heap_offsets:
                voted_size = _vote_stack_offset(stack_offset)
                if voted_size > 0:
                    self.final_offsets[stack_offset] = voted_size

    def guess(self,):
        # fill in blanks 
        self.final_offsets = guess_offset(self.final_offsets)


class Aggregator():
    # on bin level
    def __init__(self, prep_fname, prep_fpath, prepdata, equiv_vars, group_folder, save_folder, test_mode):
        self.prep_fname = prep_fname  # for csv purpose
        self.prep_fpath = prep_fpath  # for csv purpose
        self.prepdata = prepdata
        self.equiv_vars = equiv_vars   # callgraph
        self.group_folder = group_folder
        self.save_folder = save_folder
        self.seen_group:Set[str] = set()    # seen group (rep node)
        # self.out_df = out_df
        self.test_mode:bool = test_mode

    def parse_funvar(self, funvar) -> (str, str):
        fun, var = funvar.split('---')
        if var.startswith('&'):
            var = var.strip('&')
        return fun, var

    def get_stack_ground_truth_of_var(self, fun, var) -> (Union[None,str], Union[None,str]):
        if fun not in self.prepdata:
            return None, None
        if 'stack' not in self.prepdata[fun]:
            return None, None
        # print(fun, var)
        if var not in self.prepdata[fun]['stack']['ground_truth']:
            return None, None
        return self.prepdata[fun]['stack']['ground_truth'][var]

    def passed_as_ref(self, fun, var, var_ida_arr) -> bool:
        if not var_ida_arr:
            var = '&'+var
        for callee_data in self.prepdata[fun]['callee']:
            calleefun, calleeparams = callee_data['fun'], callee_data['args']
            if var in calleeparams:
                return True
            for param in calleeparams:
                # (__int64) v4
                if re.match(r'^\(.*?\){}'.format(var), param):
                    return True
                
        return False

    def get_fun_clusters(self, fun:str)-> (List[List], List[List]):
        def _get_fun_clusters_helper(funstackdata, order):
            last_not_dash = None
            curr_cluster = []
            cluster = []
            for var in order:
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

        order = self.prepdata[fun]['stack']['order']

        if self.test_mode:
            return _get_fun_clusters_helper(self.prepdata[fun]['stack']['inference'], order), []
        else:
            return _get_fun_clusters_helper(self.prepdata[fun]['stack']['inference'], order), _get_fun_clusters_helper(self.prepdata[fun]['stack']['ground_truth'], order)
        
    def fun_heap_vars(self, fun) -> List[str]:
        if 'heap' in self.prepdata[fun]:
            return set([self.prepdata[fun]['heap']['parsed'][expr]['varName'] for expr in self.prepdata[fun]['heap']['parsed']])
        else:
            return []

    def is_ida_arr(self, fun, var) -> bool:
        return self.prepdata[fun]['ida_type'][var][1]

    def is_valid_var(self, fun, var) -> bool:
        return var in self.prepdata[fun]['argument'] or var in self.prepdata[fun]['variable']


    def in_callgraph(self, fun, var_str):
        # var_str should consider "&"
        for rep, group in self.equiv_vars.items():
            if f"{fun}---{var_str}" in group:
                return True
        return False

    def joint_vote(self, rep, group_data, group_fpath) -> (int, Dict[int, int]):
        # return voted_total_size, voted_offsets (offset -> size)
        '''
        heap_offsets (consider as gt)
        threshold = 2 if heap_offsets not empty else: 1
        for stack_offset not in heap_offsets:
            vote 
        for heap_offset not in stack_offsets:
            check size
            add "-" (fn)

        '''

        layout_voter = LayoutVoter()
        
        # update heap offset to layout_voter
        for funvar, vote_data in group_data['votes'].items():
            pred_ptr = 'stack' not in vote_data or is_pred_ptr(vote_data['stack'][1])
            # print(funvar, pred_ptr)
            if pred_ptr and 'heap_offsets' in vote_data:
                for (offset, size) in vote_data['heap_offsets']:
                    layout_voter.update_heap_offset(offset, size)
        layout_voter.vote_heap_offset()

        has_cluster_predicted = False
        for funvar, vote_data in group_data['votes'].items():
            pred_ptr = 'stack' not in vote_data or is_pred_ptr(vote_data['stack'][1])
            
            if 'stack' in vote_data and not pred_ptr:
                if 'cluster' in vote_data and '&' in funvar and 'heap' not in vote_data:
                    has_cluster_predicted = True
                    for off, size in vote_data['stack_offsets']:
                        layout_voter.update_stack_offset(off, size)
                layout_voter.update_total_size_vote(sum(vote_data['stack_offsets'][-1]))


        # if early_reject(layout_voter.final_heap_offsets, has_cluster_predicted):
        #     self.veto_char_ptr(group_data['votes'], msg=has_cluster_predicted, rep=rep, group_fpath=group_fpath)
        #     layout_voter.final_heap_offsets = {}

        
        layout_voter.vote_layout()
        layout_voter.guess()

        voted_offsets = layout_voter.final_offsets
        voted_total_size = layout_voter.final_total_size

        return voted_total_size, voted_offsets 

    def veto_cluster(self, fun, cluster:List[str], gt_clusters: List[List], reason:RejReason, msg:str = "", rep:str="", group_fpath="", voted_offsets={}) -> bool: 
        if self.test_mode:
            correct_rej = "Unknown"
        else:
            gt_heads = [c[0] for c in gt_clusters]
            correct_rej = 'Correctly' if cluster[0] not in gt_heads else 'Incorrectly'
        print(f"{correct_rej} reject cluster {fun} --- {cluster} due to {reason}: {msg}")
        
        for var in cluster:
            self.veto_stack_var(fun, var, reason, msg, verbose=False)
        # new_row = {
        #     'prep_fname': self.prep_fname, 
        #     'prep_fpath': self.prep_fpath, 
        #     'fun': fun,
        #     'group_rep': rep, 
        #     'group_fpath': group_fpath, 
        #     'reject': cluster, 
        #     'reason': reason,
        #     'msg': msg,
        #     'correct': correct_rej=='Correctly',
        #     'voted_offsets': '---'.join([f"{k}:{v}" for k, v in voted_offsets.items()])
        # }
        # self.out_df.loc[len(self.out_df)] = new_row
        return correct_rej=='Correctly'

    def veto_stack_var(self, fun, var, reason:RejReason, msg:str = "", verbose:bool=True, rep:str="", group_fpath="", voted_offsets={}) -> bool: 
        
        if verbose:
            if self.test_mode:
                correct_rej = "Unknown"
            else:
                gt_name, gt_type = self.get_stack_ground_truth_of_var(fun, var)
                if gt_type:
                    if gt_type == '-':
                        correct_rej = "Incorrectly"
                    else:
                        correct_rej = 'Correctly'
                else:
                    correct_rej = 'Unknown'
                
            print(f"{correct_rej} reject {fun} --- {var} due to {reason}: {msg}")
            # new_row = {
            #     'prep_fname': self.prep_fname, 
            #     'prep_fpath': self.prep_fpath, 
            #     'fun': fun,
            #     'group_rep': rep, 
            #     'group_fpath': group_fpath, 
            #     'reject': var, 
            #     'reason': reason,
            #     'msg': msg,
            #     'correct': correct_rej=='Correctly',
            #     'voted_offsets': '---'.join([f"{k}:{v}" for k, v in voted_offsets.items()])
            # }
            # self.out_df.loc[len(self.out_df)] = new_row


            return correct_rej == 'Correctly'
            
        return False

    def veto_char_ptr(self, vote_data, msg="", rep="", group_fpath="", voted_offsets={}):
        correct_rej = "Unknown"
        if not self.test_mode:
            for funvar, data in vote_data.items():
                fun, var = self.parse_funvar(funvar)
                if 'heap' in data and 'stack' in data:
                    correct_rej = 'Incorrectly'
                    for expr, expr_data in self.prepdata[fun]['heap']['ground_truth'].items():
                        if expr in self.prepdata[fun]['heap']['parsed'] and var == self.prepdata[fun]['heap']['parsed'][expr]['varName']:
                            if expr_data[2] == '-' and re.search(r'char\s*\*', data['stack_gt'][1]):
                                correct_rej = "Correctly"
                                break
        reason = RejReason.CHARPTR
        # new_row = {
        #     'prep_fname': self.prep_fname, 
        #     'prep_fpath': self.prep_fpath, 
        #     'fun': fun,
        #     'group_rep': rep, 
        #     'group_fpath': group_fpath, 
        #     'reject': var, 
        #     'reason': reason,
        #     'msg': msg,
        #     'correct': correct_rej=='Correctly',
        #     'voted_offsets': '---'.join([f"{k}:{v}" for k, v in voted_offsets.items()])
        # }
        # self.out_df.loc[len(self.out_df)] = new_row


        print(f"{correct_rej} reject {fun} --- {var}: {msg} due to {reason}")
        return correct_rej == 'Correctly'
        
    def propose_stack_var(self, fun, var, head, msg:str = "") -> bool:
        
        if self.test_mode:
            correct = "Unknown"
        else:
            _, cluster_gt = self.get_fun_clusters(fun)
            correct = 'Incorrectly'
            for cluster in cluster_gt:
                if var in cluster and cluster[0] == head:
                    correct = "Correctly"
                    break

        print(f"{correct} propose {fun} --- {var}: {msg}")
        return correct == "Correctly"

    def propose_offset_size(self, fun, var, new_size, msg:str = ""):
        print(f"propose size {new_size} for {fun} --- {var}: {msg}")

    def iter_stack_head(self) -> List:
        rejected_cluster = []  # list of (fun, correct?, [var1, ...])
        for fun, fundata in self.prepdata.items():
            try:
                cluster_pred, cluster_gt = self.get_fun_clusters(fun)
            except:
                continue
            for cp in cluster_pred:
                head_pred = cp[0]
                head_ida_arr = self.is_ida_arr(fun, head_pred)
                head_ref_expr = "" if head_ida_arr else "&"
                head_ref_expr += head_pred
                if not self.passed_as_ref(fun, head_pred, head_ida_arr) and not self.in_callgraph(fun, head_ref_expr):
                    correct = self.veto_cluster(fun, cp, cluster_gt, reason=RejReason.NeverPassedAsRef)
                    rejected_cluster.append([fun, correct, cp])
                    continue
        return rejected_cluster
        
    def veto_based_on_vote_results(self, funvar, votedata, total_size, voted_offsets, rep, group_fpath) -> (List, List):
        fun, var = self.parse_funvar(funvar)
        rejected_cluster = []  # list of (fun, correct?, [var1, ...])
        rejected_stack_vars = [] # list of (fun, correct?, varname)
        rejjectd_vars_from_cluster = []
        voted_size = max(total_size, max(voted_offsets.keys()))
        if 'cluster' in votedata:
            for i, (offset, size) in enumerate(votedata['stack_offsets']):
                if offset!=0 and offset >= voted_size: #not in voted_offsets:
                    field_var = votedata['cluster'][i][0]
                    if field_var in rejjectd_vars_from_cluster:
                        continue
                    correct = self.veto_stack_var(fun, field_var, RejReason.VETO, msg=f"offset {offset} is larger than the voted size {voted_size}.", rep=rep, group_fpath=group_fpath, voted_offsets=voted_offsets)
                    rejected_stack_vars.append([fun, correct, field_var])

        return rejected_stack_vars

    def propose_stack_vars_into_cluster(self, funvar, votedata, total_size, voted_offsets) -> List:
        def _find_next_var(curr_var) -> (Union[None,str], Union[None,int]):
            try:
                curr_idx = self.prepdata[fun]['variable'].index(curr_var)
                next_var = self.prepdata[fun]['variable'][curr_idx+1]
                next_var_ida_size = self.prepdata[fun]['ida_size'][next_var]
                if next_var not in self.prepdata[fun]['stack']['inference']:
                    return None, None
            except Exception as e:
                return None, None
            return next_var, next_var_ida_size

        proposed_stack_vars = [] # list of (fun, correct?, varname, head)
        # only propose for stack variables

        if 'stack' not in votedata or is_pred_ptr(votedata['stack'][1]):
            return proposed_stack_vars
        
        corrected_offsets = {} #votedata['stack_offsets'].copy()
        tmp_max_offset = -1
        fun, var = self.parse_funvar(funvar)
        # correct offset (those with size -1)
        for offset, size in votedata['stack_offsets']:
            if size == -1 and offset in voted_offsets and voted_offsets[offset] > 0:
                corrected_offsets[offset] = voted_offsets[offset]
                self.propose_offset_size(fun, var, voted_offsets[offset])
            else:
                corrected_offsets[offset] = size
            tmp_max_offset = max(tmp_max_offset, offset)

        if tmp_max_offset>=0:
            curr_size = tmp_max_offset + corrected_offsets[tmp_max_offset]
        else:
            return proposed_stack_vars
        curr_var = var
        while curr_size < total_size:
            
            # ----------- find the next available offset -------------
            next_offset = -1
            for offset in sorted(voted_offsets.keys()):
                if offset >= curr_size:
                    next_offset = offset
                    break
            if next_offset < 0:
                break
            next_size = voted_offsets[next_offset]
            assert next_size > 0
            next_var, next_var_ida_size = _find_next_var(curr_var)
            if next_var is not None:
                correct = self.propose_stack_var(fun, next_var, var, msg=f"propose for offset {next_offset} with size {next_size} with head {var}")
                proposed_stack_vars.append([fun, correct, next_var, var])
                curr_size += max(1, next_var_ida_size) # in case next_var_ida_size is -1, add 1 anyway
                curr_var = next_var

            else:
                break

        return proposed_stack_vars


    def iter_group(self):
        for rep, group in self.equiv_vars.items():
            try:
                rejected_stack_vars, proposed_stack_vars = [], []
                group_fpath = os.path.join(self.group_folder, rep + '.json')
                if not os.path.exists(group_fpath):
                    continue

                group_data = read_json(group_fpath)
                total_size, voted_offsets = self.joint_vote(rep, group_data, group_fpath)  # vote based on heap and stack
                if not voted_offsets :
                    # rejected
                    continue
                for funvar, votedata in group_data['votes'].items():
                    tmp_rejected_stack_vars = self.veto_based_on_vote_results(funvar, votedata, total_size, voted_offsets, rep, group_fpath)
                    
                    if PROPOSE_ON:
                        tmp_proposed_stack_vars = self.propose_stack_vars_into_cluster(funvar, votedata, total_size, voted_offsets)
                    else:
                        tmp_proposed_stack_vars = []

                    rejected_stack_vars+= tmp_rejected_stack_vars
                    proposed_stack_vars += tmp_proposed_stack_vars
                
                group_data['final'] = {
                    'rejected_stack_vars': rejected_stack_vars,
                    'proposed_stack_vars': proposed_stack_vars,
                    'total_size': total_size,
                    'voted_offsets': voted_offsets
                } 
                dump_json(os.path.join(self.save_folder, rep+'.json'), group_data)
            except VoteError:
                continue
            
    def vote(self):
        rejected_cluster = self.iter_stack_head()
        dump_json(os.path.join(self.save_folder, 'rejected_stack_cluster.json'), rejected_cluster)
        self.iter_group()
        

def main(prep_dir, equiv_vars_dir, group_folder, save_folder, test_mode):
    init_folder(save_folder)
    # out_df = pd.DataFrame(columns=['prep_fname', 'prep_fpath', 'fun', 'group_rep', 'group_fpath', 'reject', 'voted_offsets', 'reason', 'msg', 'correct'])
    for binfile in tqdm(get_file_list(prep_dir)):
        binname = binfile.replace('.json', "")
        bin_fpath = os.path.join(prep_dir, binfile)
        prepdata = read_json(bin_fpath)
        equiv_vars = read_json(os.path.join(equiv_vars_dir, binfile))
        bin_save_folder = os.path.join(save_folder, binname)
        init_folder(bin_save_folder)
        aggregator = Aggregator(binfile, bin_fpath, prepdata, equiv_vars, os.path.join(group_folder, binname), bin_save_folder, test_mode)

        print(f'------------ {binfile} ------------')
        aggregator.vote()

    # if out:
    #     out_df.to_csv(out, index=False)


def test():
    def _test_guess_offset(offsets):
        out = guess_offset(offsets.copy())
        print(f"{offsets} -> {out}")
    _test_guess_offset({0:8, 16:8})
    _test_guess_offset({8:8})
    _test_guess_offset({24:8})
    _test_guess_offset({4:4})

# test()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('equiv_vars_dir')
    parser.add_argument('prep_dir')
    parser.add_argument('group_folder')
    parser.add_argument('save_folder')
    parser.add_argument("--test", action="store_true", help="Enable test mode")
    # parser.add_argument("--out", help="out put csv file path", default="")
    args = parser.parse_args()
    
    main(args.prep_dir, args.equiv_vars_dir, args.group_folder, args.save_folder, args.test)

