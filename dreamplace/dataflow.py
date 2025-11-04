import logging
from collections import defaultdict
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import numpy as np
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.configure as configure
import Params
import PlaceDB
import Timer
import NonLinearPlace
import pdb
from typing import Tuple
import pickle

class dStack:
    def __init__(self,node):
        self.nodes = [node]

    def push(self, node):
        """ Push a new node onto the stack and update tracking sets. """
        self.nodes.append(node)

    def pop(self):
        """ Pop the top node from the stack and update tracking sets. """
        node = self.nodes.pop()
        return node
       

    def depth(self):
        return len(self.nodes)-1#去掉首尾的macro为depth 此处在过程中尾部不是macro
    

    def is_empty(self):
        """ Check if the stack is empty. """
        return len(self.nodes) == 0

    def top(self):
        """ Get the top element of the stack. """
        return self.nodes[-1] if not self.is_empty() else None

    def is_in_path(self, node):
        """ Check if a node, net, or pin is already in the path. """
        return node in self.nodes

def get_neighbors(placedb: PlaceDB.PlaceDB,node):
        neighbors = []
        for pin in placedb.node2pin_map[node]:
            if placedb.pin_direct[pin] == b'OUTPUT':
                net = placedb.pin2net_map[pin]#pin2net
                for net_pin in placedb.net2pin_map[net]:
                    if placedb.pin_direct[net_pin] == b'INPUT':
                        neighbor_node = placedb.pin2node_map[net_pin]
                        if neighbor_node != node:
                            neighbors.append(neighbor_node)
        return neighbors

def get_neighbors_no_direct(placedb: PlaceDB.PlaceDB,node):
        neighbors = []
        for pin in placedb.node2pin_map[node]:
            net = placedb.pin2net_map[pin]#pin2net
            for net_pin in placedb.net2pin_map[net]:
                    neighbor_node = placedb.pin2node_map[net_pin]
                    if neighbor_node != node:
                        neighbors.append(neighbor_node)
        return neighbors

class dPath:
    def __init__(self, path_nodes=[]):
        #建立path时首尾都是macro,尾部未添加
        self.path_nodes = path_nodes
    

    def end_add(self, node):
        self.path_nodes.append(node)

    def print_path(self):
        print(self.path_nodes)
    
    @property
    def start_node(self):
        return self.path_nodes[0]
    
    @property
    def depth(self):
        return len(self.path_nodes)
    
    @property
    def end_node(self):
        return self.path_nodes[-1]


class dDataflowCaler:
    def __init__(self, placedb: PlaceDB.PlaceDB,params: Params.Params, depth_max, depth_max_no_direct):
        self.placedb = placedb
        self.depth_max = depth_max
        self.depth_max_no_direct = depth_max_no_direct
        self.all_macro_to_macro_path = []
        self.all_macro_to_macro_path_no_direct = []
        with open(f"benchmarks/{params.design_name()}/macro_names.pkl","rb") as f:
            macro: dict = pickle.load(f)
        self.macro_ids = np.array(list(macro.values()))
        self.macro_id_map = {}
        self.num_macro = len(self.macro_ids)
        for i,macro_id in enumerate(self.macro_ids):
            self.macro_id_map[macro_id] = i
        self.initialize_data_structures()
    
    def compute(self):
        self.compute_macro_to_macro_path_no_direct()
        self.compute_macro_to_macro_dataflow_no_direct()
        self.compute_macro_to_macro_path()
        self.compute_macro_to_macro_dataflow()

    def compute_macro_to_macro_path(self):
        for i,macro in enumerate(self.macro_ids):
            print(f"compute path for {macro}: {i}/{self.num_macro}")
            stack = dStack(macro)#stack用于遍历，path是遍历结束后用于创建,其中记录node的都是list
            node_paths = self.dfs(stack)
            self.all_macro_to_macro_path.extend(node_paths)

    def dfs(self, stack:dStack):
        if stack.is_empty() or stack.depth() > self.depth_max:
            return []
        current_node = stack.top()
        paths = []
        for neighbor in get_neighbors(self.placedb,current_node):
            if stack.is_in_path(neighbor):
                continue
            elif neighbor in self.macro_ids:
                new_path = dPath(stack.nodes[:])
                new_path.end_add(neighbor)
                paths.append(new_path)
            else:
                stack.push(neighbor)
                paths.extend(self.dfs(stack))
                stack.pop()
        return paths
    
    def initialize_data_structures(self):
        self.path_matrices = [np.zeros((self.num_macro, self.num_macro)) for _ in range(self.depth_max)]
        self.path_matrices_no_direct = [np.zeros((self.num_macro, self.num_macro)) for _ in range(self.depth_max_no_direct)]
        self.macro_to_macro_flow = np.zeros((self.num_macro, self.num_macro))
        self.macro_to_macro_flow_no_direct = np.zeros((self.num_macro, self.num_macro))
      

    def compute_macro_to_macro_dataflow(self):
        for path in self.all_macro_to_macro_path:
            k = path.depth - 2  # Path length excluding endpoints
            start = self.macro_id_map[path.start_node]
            end = self.macro_id_map[path.end_node]
            flow_increment = 0.5 ** k
            self.macro_to_macro_flow[start, end] += flow_increment
            if 0 <= k < self.depth_max:
                self.path_matrices[k][start, end] += 1
        print(f"finish compute dataflow: {np.where(self.macro_to_macro_flow>0)[0].size}")
    
    def compute_macro_to_macro_path_no_direct(self):
        for i,macro in enumerate(self.macro_ids):
            print(f"compute no direct path for {macro}: {i}/{self.num_macro}")
            stack = dStack(macro)#stack用于遍历，path是遍历结束后用于创建,其中记录node的都是list
            node_paths = self.dfs_no_direct(stack)
            self.all_macro_to_macro_path_no_direct.extend(node_paths)

    def dfs_no_direct(self, stack:dStack):
        if stack.is_empty() or stack.depth() > self.depth_max_no_direct:
            return []
        current_node = stack.top()
        paths = []
        for i,neighbor in enumerate(get_neighbors_no_direct(self.placedb,current_node)):
            if stack.is_in_path(neighbor):
                continue
            elif neighbor in self.macro_ids:
                new_path = dPath(stack.nodes[:])
                new_path.end_add(neighbor)
                paths.append(new_path)
            else:
                stack.push(neighbor)
                paths.extend(self.dfs_no_direct(stack))
                stack.pop()
        return paths

    def compute_macro_to_macro_dataflow_no_direct(self):
        for path in self.all_macro_to_macro_path_no_direct:
            k = path.depth - 2  # Path length excluding endpoints
            start = self.macro_id_map[path.start_node]
            end = self.macro_id_map[path.end_node]
            flow_increment = 0.5 ** k
            self.macro_to_macro_flow_no_direct[start, end] += flow_increment
            if 0 <= k < self.depth_max_no_direct:
                self.path_matrices_no_direct[k][start, end] += 1
        print(f"finish compute no direct dataflow: {np.where(self.macro_to_macro_flow_no_direct>0)[0].size}")

    def print_macro_to_macro_flow(self):
        width = 10
        header = ''.join(f"{index:>{width}}" for index in range(self.num_macro))
        print(f"{header:>{width}}")
        for i in range(self.num_macro):
            row_data = ''.join(f"{self.d_macro_to_macro_flow[i, j]:{width}.2f}" for j in range(self.num_macro))
            print(f"{i:>{width}}{row_data}")

    def print_macro_to_macro_path(self):
        for i, path in enumerate(self.all_macro_to_macro_path):
            print(f"Path {i:3}: ", end='')
            path.print_path()
            k = path.length() - 2
            weight = 0.5 ** k
            print(f"    {weight:3.2f}")

    def write_macro_to_macro_flow(self, filename,pickle_name):
        #macro标记以dmp中的placedb.node为标记
        with open(filename, 'w') as fw:
            header = ','.join(str(i) for i in self.macro_ids)
            fw.write(f",{header}\n")
            for i in range(self.num_macro):
                row_data = ','.join(f"{self.macro_to_macro_flow[i, j]:.3f}" for j in range(self.num_macro))
                fw.write(f"{self.macro_ids[i]},{row_data}\n")
        fw.close()
        with open(pickle_name,'wb') as f:
            pickle.dump(self.macro_to_macro_flow,f)
        f.close()

    def write_macro_to_macro_path(self, filename,pickle_name):
        with open(filename, 'w') as fw:
            header = ','.join(str(i) for i in self.macro_ids)
            fw.write(f",{header}\n")
            for j in range(self.num_macro):
                line = f'{self.macro_ids[j]}'
                for k in range(self.num_macro):
                    path_data = ' '.join(f"{path_matrix[j, k]}" for path_matrix in self.path_matrices)
                    line = ','.join([line,'('+path_data+')'])
                fw.write(f"{line}\n")
        fw.close()
        with open(pickle_name,'wb') as f:
            pickle.dump(self.path_matrices,f)
        f.close()


    def write_macro_to_macro_flow_no_direct(self, filename,pickle_name):
        #macro标记以dmp中的placedb.node为标记
        with open(filename, 'w') as fw:
            header = ','.join(str(i) for i in self.macro_ids)
            fw.write(f",{header}\n")
            for i in range(self.num_macro):
                row_data = ','.join(f"{self.macro_to_macro_flow_no_direct[i, j]:.3f}" for j in range(self.num_macro))
                fw.write(f"{self.macro_ids[i]},{row_data}\n")
        fw.close()
        with open(pickle_name,'wb') as f:
            pickle.dump(self.macro_to_macro_flow_no_direct,f)
        f.close()

    def write_macro_to_macro_path_no_direct(self, filename,pickle_name):
        with open(filename, 'w') as fw:
            header = ','.join(str(i) for i in self.macro_ids)
            fw.write(f",{header}\n")
            for j in range(self.num_macro):
                line = f'{self.macro_ids[j]}'
                for k in range(self.num_macro):
                    path_data = ' '.join(f"{path_matrix[j, k]}" for path_matrix in self.path_matrices_no_direct)
                    line = ','.join([line,'('+path_data+')'])
                fw.write(f"{line}\n")
        fw.close()
        with open(pickle_name,'wb') as f:
            pickle.dump(self.path_matrices_no_direct,f)
        f.close()


if __name__ == "__main__":
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params.Params()
    params.printWelcome()
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.load('test/iccad2015.ot/superblue1_nt.json')
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")
        params.printHelp()
        exit()
    else:
        # load parameters
        params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    logging.info("reading database takes %.2f seconds" % (time.time() - tt))

    direct_depth = 10
    nondirect_depth = 0
    cdf = dDataflowCaler(placedb,params,direct_depth,nondirect_depth)
    cdf.compute()
    df_file = f"benchmarks/{params.design_name()}/dataflow_{direct_depth}.csv"
    df_pkl = f"benchmarks/{params.design_name()}/dataflow_{direct_depth}.pkl"
    pt_file = f"benchmarks/{params.design_name()}/path.csv"
    pt_pkl = f"benchmarks/{params.design_name()}/path.pkl"
    cdf.write_macro_to_macro_flow(df_file,df_pkl)
    cdf.write_macro_to_macro_path(pt_file,pt_pkl)

    df_file = f"benchmarks/{params.design_name()}/dataflow_nd.csv"
    df_pkl = f"benchmarks/{params.design_name()}/dataflow_nd.pkl"
    pt_file = f"benchmarks/{params.design_name()}/path_nd.csv"
    pt_pkl = f"benchmarks/{params.design_name()}/path_nd.pkl"
    cdf.write_macro_to_macro_flow_no_direct(df_file,df_pkl)
    cdf.write_macro_to_macro_path_no_direct(pt_file,pt_pkl)
