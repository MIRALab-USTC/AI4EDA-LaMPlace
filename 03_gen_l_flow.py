import torch
import torch.nn as nn
import os
import dgl
import numpy as np
import pickle
from graph_model import GCN
import argparse
import pickle
EPSILON=1e-10
device = 2
seed = 2024
device = 'cuda:%d' % device 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
np.random.seed(seed)

benchmarklist = [
    'superblue1',
    'superblue3',
    'superblue4',
    'superblue5',
    'superblue7',
    'superblue10',
    'superblue16',
    'superblue18'    
]


def normalize_graph_features(graph):
        with open("graph_data/norm_graph/norm_info.pkl","rb") as f:
            norm_info = pickle.load(f)
        node_mean = norm_info["n_m"].to(device)
        node_std = norm_info["n_s"].to(device)
        edge_mean = norm_info["e_m"].to(device)
        edge_std = norm_info["e_s"].to(device)
        graph.ndata['x'] = (graph.ndata['x']-node_mean)/node_std
        graph.edata['dataflow'] = (graph.edata['dataflow']-edge_mean)/edge_std
        return graph

def main(benchmark,a1,a2,a3,a4):
    #load graph
    g,_ = dgl.load_graphs(f'benchmarks/{benchmark}/{benchmark}.bin')
    g = g[0].to(device)
    node_feats = g.ndata['x']
    edge_feats = g.edata['dataflow']
    in_features = node_feats.size()[-1]
    in_edge_features = edge_feats.size()[-1]
    hidden_features = 256
    gcn = GCN(
            in_feats=in_features,
            in_edge_feats = in_edge_features,
            out_feats=256,
            n_metric=4,
            n_hidden=hidden_features,
            n_layers=5,
            dropout=0,
            n_lflow = 4,
            activation=nn.ReLU()
        ).to(device)
    # load model
    gcn.load_state_dict(torch.load('./graph_data/model/model.pth'))
    g = normalize_graph_features(g)
    f = g.ndata['x']
    e = g.edata['dataflow']
    gcn.eval()
    dataflow = gcn(f,e,g)[0]
    df1 = dataflow[:,:,:4]
    df0 = dataflow[:,:,4:8]
    df_1 = dataflow[:,:,8:12]
    df_2 = dataflow[:,:,12:]
    weights = torch.tensor([a1, a2, a3, a4], dtype=torch.float32).to(device)
    weights = weights.unsqueeze(0).unsqueeze(0) 
    logits1  = torch.sum(df1 * weights, dim=-1)
    logits2  = torch.sum(df0 * weights, dim=-1)
    logits3  = torch.sum(df_1 * weights, dim=-1)
    logits4  = torch.sum(df_2 * weights, dim=-1)
    np.save(f'graph_data/l_flow/{benchmark}.npy', torch.stack((logits1,logits2,logits3,logits4),dim = -1).cpu().detach().numpy())




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="superblue4")
    parser.add_argument("--a1", default="0.25")
    parser.add_argument("--a2", default="0.25")
    parser.add_argument("--a3", default="0.25")
    parser.add_argument("--a4", default="0.25")
    args = parser.parse_args()
    benchmark = args.benchmark
    a1 = float(args.a1)
    a2 = float(args.a2)
    a3 = float(args.a3)
    a4 = float(args.a4)
    main(benchmark,a1,a2,a3,a4)