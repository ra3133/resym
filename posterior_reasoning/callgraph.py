import argparse
from typing import List, Dict, Union
from tqdm import tqdm
from utils import *
from collections import defaultdict 
from vote_utils import *


NUM_CALLER_THRESHOLD=5
CHECK_NUM_CALLER= True
CHECK_CALL_STACK=False
LEVEL=10

class CallGraph():
    def __init__(self):
        self.nodes:List[str] = []
        self.represent: List[int] = []
        self.node2idx:Dict[str,int] = {}

    def find_rep(self, i, rank=0) -> int:
        # return the index of node i's represent
        if self.represent[i] == i:
            return i, rank
        else:
            return self.find_rep(self.represent[i], rank+1)
    
    def add_node(self, fun, var):
        name = encode_name(fun, var)
        if name not in self.node2idx:
            new_node_idx = len(self.nodes)
            self.node2idx [name] = new_node_idx
            self.nodes.append(name)
            self.represent.append(new_node_idx) # point to itself
            return new_node_idx
        else:
            return self.node2idx[name]

    def add_edge(self, fun1, var1, fun2, var2, bin_data=None):
        idx1 = self.add_node(fun1, var1)
        idx2 = self.add_node(fun2, var2)

        rep1, rank1 = self.find_rep(idx1, 0)
        rep2, rank2 = self.find_rep(idx2, 0)

        # union by rank, otherwise max recursion step
        if rank1 > rank2:
            self.represent[rep2] = rep1
        else:
            self.represent[rep1] = rep2

    def connected_componenet(self, bin_data, valid_only:bool, test_mode:bool) -> Dict[str,List[str]]:
        # valid_only: if set to True, discard the connected components that are all from the same function
        components = defaultdict(set)
        for node_idx, node in enumerate(self.nodes):
            rep, _ = self.find_rep(node_idx)
            components[self.nodes[rep]].add(node)
        
        # conver to list
        for node, conn in components.items():
            components[node] = sorted(list(conn))

        if not valid_only:
            return components
        else: 
            valid_components = defaultdict(set)
            for node, conn in components.items():
                valid_conn = []
                for funvar in conn:
                    fun, var = funvar.split('---')
                    # remove nodes with no ground truth
                    if fun not in bin_data:
                        continue
                    if test_mode or var_aligned(bin_data[fun], var):
                        valid_conn.append(funvar)
                # remove clusters from only one function
                conn_funs = [funvar.split('---')[0] for funvar in valid_conn]
                if len(set(conn_funs)) <= 1:
                    continue
                else:
                    valid_components[node] = valid_conn
            return valid_components


def encode_name(fun, var):
    return  f'{fun}---{var}'  

def var_aligned(fundata, var) -> bool:
    # return true if the variable has ground truth in either stack or heap
    if var.startswith('&'):
        var = var.strip('&')
    if 'stack' in fundata and var in fundata['stack']['ground_truth']:
        return True
    if 'heap' in fundata and var in [fundata['heap']['parsed'][expr]['varName'] for expr in fundata['heap']['parsed']]:
        return True

    return False

# TODO: pass as `(__int64)v13`
def process_bin(binname, bin_data, test_mode)->CallGraph:
    def is_var_ida_arr(fun, var) -> bool:
        if fun in bin_data and var in bin_data[fun]['ida_type']:
            # print('found')
            return bin_data[fun]['ida_type'][var][1]
        else:
            print(f"[{binname}] cannot find ida type for {fun} -- {var}")
            return False
        
        
    def _var_consider_type(fun, varname, addrof=False) -> str:
        # addrof: bool, is it passed with &
        var_arr_type = is_var_ida_arr(fun, varname)
        
        if var_arr_type:
            varname =  "&"+varname
            
        if addrof:
            return "&"+varname
        else:
            return varname
       
    def _beyond_access_in_callstack(fun, var, level:int) -> bool:
        # TODO consider data flow
        # return True if the variable var has beyond access in `level` levels of callstack
        if fun not in bin_data:
            return False
        fundata = bin_data[fun]
        if 'heap' in bin_data:
            for expr, exprdata in bin_data['heap']['parsed'].items():
                if exprdata['varName'] == var:
                    return True

        if level == 0:  # reach the end of level
            return False
        
        if var not in fundata['ida_type']:
            return False

        
        is_ida_arr = fundata['ida_type'][var][1]
        if is_ida_arr:  # impossible to direct pass 
            return False

        for calleedata in fundata['callee']:
            for i, arg in enumerate(calleedata['args']):
                if arg == var or re.match(r'^\(.*?\){}$'.format(var), arg):
                    return _beyond_access_in_callstack(calleedata['fun'], f"a{i+1}", level-1)

        return False

    def _add_edge_helper(fun1, var1, fun2, var2, bin_data=None):
        if var1 in invalid_direct_pass_vars:
            return
        if CHECK_CALL_STACK:
            if '&' not in var1 and not _beyond_access_in_callstack(fun2, var2, level=LEVEL):
                # if it is a direct pass and do not has beyond access in `LEVEL` levels of callstack
                return
        callgraph.add_edge(fun1, var1, fun2, var2, bin_data)

    callgraph = CallGraph()
    all_heap_vars = set()
    skip_fun = set()

    if CHECK_NUM_CALLER:
        for fun, fundata in bin_data.items():
            if len(fundata['caller']) >= NUM_CALLER_THRESHOLD:
                skip_fun.add(fun) 

    for fun, fundata in bin_data.items():
        if fun in skip_fun:
            continue

        cluster_pred = []
        if 'stack' in fundata:
            cluster_pred = get_fun_clusters(fundata['stack']['inference'], fundata['stack']['order'])
        pred_heads = [c[0] for c in cluster_pred]
        invalid_direct_pass_vars = []  # if these vars are passed directly (not as &, consider the edge invalid)
        for head in pred_heads:
            if not fundata['ida_type'][head][1]: # if is not ida arr
                invalid_direct_pass_vars.append(head)

        # if fun!='sub_41CB5E':
        #     continue
        # print(invalid_direct_pass_vars)
        for var, equiv_var_ls in fundata['dataflow'].items():
            for equiv_var in equiv_var_ls:
                # print(var, equiv_var)
                if var not in invalid_direct_pass_vars and equiv_var not in invalid_direct_pass_vars:
                    _add_edge_helper(
                        fun, 
                        var,
                        fun, 
                        equiv_var,
                        bin_data,
                    )

        
        if 'heap' in fundata:
            for heap_expr in fundata['heap']['parsed']:
                heap_var = fundata['heap']['parsed'][heap_expr]['varName']
                all_heap_vars.add(encode_name(fun, heap_var))
                
        # only analyze callee 
        for callee_data in fundata['callee']:
            callee_fun, callee_params = callee_data['fun'], callee_data['args']
            if callee_fun in skip_fun:
                continue
            for i, param in enumerate(callee_params):
                if re.match(r'^[a-zA-Z0-9_]+$', param):
                    
                    if (param in fundata['argument'] or param in fundata['variable']): 
                        var_expr = _var_consider_type(fun, param, False)
                        _add_edge_helper(
                            fun, 
                            # param,
                            var_expr,
                            callee_fun, 
                            f"a{i+1}",
                            bin_data
                            )
                        continue

                # TODO: need to consider assignee type. if both ptr, then its equivlent
                
                match = re.match(r'^&([a-zA-Z0-9_]+)$', param)
                if match:
                    # f1(..){ f2(&v1) } ->  f2(a1)
                    # regardless of types of v1 and a1, always &v1=a1
                    v = match.group(1)
                    if v in fundata['argument'] or v in fundata['variable']:
                        var_expr = _var_consider_type(fun, v, True)
                        _add_edge_helper(
                            fun, 
                            # "&"+v, 
                            var_expr, 
                            callee_fun, 
                            f"a{i+1}",
                            bin_data
                            )
                        continue
                # (__int64)v4 
                match = re.match(r'^\(.*?\)([a-zA-Z0-9_]+)$', param)
                if match:
                    v = match.group(1)
                    
                    if (v in fundata['argument'] or v in fundata['variable']):
                        var_expr = _var_consider_type(fun, v, False)
                        _add_edge_helper(
                            fun, 
                            # v,
                            var_expr, 
                            callee_fun, 
                            f"a{i+1}",
                            bin_data
                            )
                
                match = re.match(r'^\(.*?\)&([a-zA-Z0-9_]+)$', param)
                if match:
                    v = match.group(1)
                    if v in fundata['argument'] or v in fundata['variable']:
                        var_expr = _var_consider_type(fun, v, True)
                        # print(binname, fun, v)
                        _add_edge_helper(
                            fun, 
                            # v,
                            var_expr, 
                            callee_fun, 
                            f"a{i+1}",
                            bin_data
                            )

    # for idx, num_caller in callgraph.nodes_num_caller.items():
    #     if num_caller >= NUM_CALLER_THRESHOLD:
    #         skip_fun.append(callgraph.nodes[idx].split('---')[0])

    return callgraph


def main(prep_aggregation_dir, save_dir, test_mode):
    init_folder(save_dir)
    callgraphs:Dict[str: CallGraph] = {}   # binname -> CallGraph
    
    for binfile in tqdm(get_file_list(prep_aggregation_dir)):
        if not binfile.endswith('.json'):
            continue
        bin_data = read_json(os.path.join(prep_aggregation_dir, binfile))
        binname = binfile.split('.')[0]

        # first iter, get skip_fun
        tmp_callgraph = process_bin(binname, bin_data, test_mode)
        callgraphs[binname] = tmp_callgraph
        conn_comp = callgraphs[binname].connected_componenet(bin_data, valid_only=True, test_mode=test_mode)
        
        dump_json(os.path.join(save_dir, binfile), conn_comp)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prep_aggregation_dir')
    parser.add_argument('save_dir')
    parser.add_argument("--test", action="store_true", help="Enable test mode")
    args = parser.parse_args()
    main(args.prep_aggregation_dir, args.save_dir, args.test)
