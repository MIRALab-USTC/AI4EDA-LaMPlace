import torch
import torch.nn as nn
import dgl

class GCN(nn.Module):
    def __init__(
        self, in_feats, in_edge_feats, out_feats, n_hidden,n_layers,n_metric,activation, dropout,n_lflow
    ):
        super(GCN, self).__init__()
        self.out_feats = out_feats
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.n_metric = n_metric
        self.n_lflow = n_lflow
        self.node_encoder = MLP(in_feats,[n_hidden],n_hidden)
        self.edge_encoder = MLP(in_edge_feats,[n_hidden],n_hidden)
        for i in range(n_layers):
            self.layers.append(
                CustomGraphConv(n_hidden, n_hidden, activation=activation)
            )
        self.dropout = nn.Dropout(p=dropout)
        self.mlp_out = MLP(n_hidden,[out_feats],out_feats)
        self.encoder = nn.ModuleList()
        for _ in range(n_lflow*n_metric):
            self.encoder.append(nn.Linear(out_feats, out_feats))
            
    def forward(self, node_features,edge_features,bg):
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features.float())
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            h = layer(bg, h, e) + h
        h =self.mlp_out(h)
        bg.ndata['h'] = h
        gs = dgl.unbatch(bg)
        output = [] 
        for g in gs:
            lflow = []
            h = g.ndata['h']
            for i in range(self.n_lflow * self.n_metric):
                encoder1 = self.encoder[i]
                df = encoder1(h) @ h.T
                lflow.append(df)            
            lflow = torch.stack(lflow, dim=-1)
            output.append(lflow)
        return output
    


class CustomGraphConv(nn.Module):
    def __init__(self, in_feat, out_feat,activation):
        super(CustomGraphConv, self).__init__()
        self.mlp1 = MLP(2*in_feat,[in_feat],in_feat,activation)
        self.mlp2 = MLP(in_feat,[out_feat],out_feat,activation)
    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_feats
            def message_func(edges):
                return {'m': self.mlp1(torch.cat([edges.src['h'], edges.data['e']], dim=1))}
            def reduce_func(nodes):
                return {'h': torch.sum(nodes.mailbox['m'], dim=1) + nodes.data['h']}
            g.update_all(message_func, reduce_func)
            g.ndata['h'] = self.mlp2(g.ndata['h'])
            return g.ndata['h']



class CustomGraphConv(nn.Module):
    def __init__(self, in_feat, out_feat,activation):
        super(CustomGraphConv, self).__init__()
        self.mlp1 = MLP(2*in_feat,[in_feat],in_feat,activation)
        self.mlp2 = MLP(in_feat,[out_feat],out_feat,activation)
    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_feats
            def message_func(edges):
                return {'m': self.mlp1(torch.cat([edges.src['h'], edges.data['e']], dim=1))}
            def reduce_func(nodes):
                return {'h': torch.sum(nodes.mailbox['m'], dim=1) + nodes.data['h']}
            g.update_all(message_func, reduce_func)
            g.ndata['h'] = self.mlp2(g.ndata['h'])
            
            return g.ndata['h']

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_fn=nn.ReLU()):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation_fn)
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
