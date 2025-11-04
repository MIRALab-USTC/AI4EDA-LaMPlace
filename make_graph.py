import dgl
import torch
import pickle
import numpy as np
import pickle
import argparse
import pdb
from dreamplace.PlaceDB import PlaceDB
from dreamplace.Params import Params
def dataflow_graph(placedb: PlaceDB,params: Params,depth):
    with open(f"benchmarks/{params.design_name()}/dataflow_{depth}.pkl","rb") as f:
        dataflow = pickle.load(f)
    with open(f"benchmarks/{params.design_name()}/path.pkl","rb") as f:
        datapaths = pickle.load(f)
    with open(f"benchmarks/{params.design_name()}/dataflow_nd.pkl","rb") as f:
        dataflow_nd = pickle.load(f)
    with open(f"benchmarks/{params.design_name()}/path_nd.pkl","rb") as f:
        datapaths_nd = pickle.load(f)
    with open(f"benchmarks/{params.design_name()}/macro_names.pkl","rb") as f:
        macro_name = pickle.load(f)
    macro_id = np.array(list(macro_name.values())) 
    src,dst = np.where(dataflow_nd > 0)
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    #edge_fea df 0p 1p 2p ... depthp 
    edge_fea_df = torch.tensor(dataflow[src,dst])
    edge_fea_pt = [torch.tensor(datapath[src,dst]) for datapath in datapaths]
    edge_fea_df_nd = torch.tensor(dataflow_nd[src,dst])
    edge_fea_pt_nd = [torch.tensor(datapath[src,dst]) for datapath in datapaths_nd]
    edge_fea = [edge_fea_df_nd]
    edge_fea.extend(edge_fea_pt_nd)
    edge_fea.append(edge_fea_df)
    edge_fea.extend(edge_fea_pt)
    edge_fea = torch.stack(tuple(edge_fea),dim = 1)

    node_size_x = torch.tensor(placedb.node_size_x[macro_id])
    node_size_y = torch.tensor(placedb.node_size_y[macro_id])
    node_num_pins = torch.tensor(np.array([len(placedb.node2pin_map[m_id]) for m_id in macro_id]))
    node_fea = torch.stack((node_size_x,node_size_y,node_num_pins),dim = 1)
    
    graph = dgl.graph((src,dst),num_nodes = len(macro_id))
    graph.edata['dataflow'] = edge_fea
    graph.ndata['x'] = node_fea

    graph = dgl.add_self_loop(graph)
    print(f"graph with {graph.num_nodes()} nodes, {graph.num_edges()} edges")
    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="superblue4")
    args = parser.parse_args()
    benchmark = args.benchmark
    placedb=PlaceDB()
    params=Params()
    #benchmark = "superblue18"
    depth = 10
    config=f'./test/iccad2015.ot/{benchmark}.json'
    params.load(config)
    placedb(params)
    g = dataflow_graph(placedb,params,depth)
    dgl.save_graphs(f'benchmarks/{benchmark}/{benchmark}.bin', g)
