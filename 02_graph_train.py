import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dgl.data import DGLDataset
import dgl
from torch.utils.data import Subset
import numpy as np
import pickle
from scipy.stats import kendalltau
# import matplotlib.pyplot as plt
from graph_model import GCN
from torch.utils.tensorboard import SummaryWriter
import pickle
import datetime

current_time = datetime.datetime.now()
EPSILON=1e-10
device = 1
seed = 2025
device = 'cuda:%d' % device 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
np.random.seed(seed)
writer = SummaryWriter(f'runs/train_{current_time}')
benchmarklist = [
    'superblue1',
    'superblue3',
    'superblue4',
    'superblue5',
    'superblue7',
    'superblue10',
]

bench_ids=[1,
           3,
           4,
           5,
           7,
           10,
           ]


def collate_fn(batch):
    graphs, pos, hpwl, con, tns, wns = zip(*batch)
    batched_graph = dgl.batch(graphs)
    pos_padded = torch.nn.utils.rnn.pad_sequence(pos, batch_first=True, padding_value=-1000)
    hpwl = torch.stack(hpwl)
    con = torch.stack(con)
    tns = torch.stack(tns)
    wns = torch.stack(wns)
    return batched_graph, pos_padded, hpwl, con, tns, wns

def restore_original_tensors(padded_tensors, pad_value=-1000):
    restored_tensors = []
    for tensor in padded_tensors:
        mask = (tensor == pad_value)
        mask = mask.all(dim=1)
        if mask.any():
            first_pad_index = mask.nonzero(as_tuple=True)[0][0].item()
            restored_tensor = tensor[:first_pad_index]
        else:
            restored_tensor = tensor
        restored_tensors.append(restored_tensor)
    
    return restored_tensors

class placeset(DGLDataset):
    def __init__(self, graphs, pos, benchmarks, con, hpwl, tns, wns):
        super().__init__(name='my_custom_dataset')
        self.graphs = self.normalize_graph_features(graphs)
        self.pos = [p.to(device) for p in pos]
        self.hpwl = self.norm(hpwl).to(device)
        self.con = self.norm(con).to(device)
        self.tns = self.norm(-tns).to(device)
        self.wns = self.norm(-wns).to(device)
        self.benchmarks = benchmarks

    def __getitem__(self, idx):
        return (
            self.graphs[self.benchmarks[idx].item()],
            self.pos[idx],
            self.hpwl[idx],
            self.con[idx],
            self.tns[idx],
            self.wns[idx]
        )

    def process(self):
        pass
    
    def __len__(self):
        return len(self.benchmarks)
    
    def norm(self,label):
        mean = label.mean()
        std = label.std()
        label = (label-mean)/std
        return label

    def load(self):
        self.graphs, label_dict = dgl.load_graphs('graph_data.bin')
        self.labels = label_dict['labels']

    def normalize_graph_features(self,graphs):
        batched_graph = dgl.batch(list(graphs.values()))
        
        # node feature normalization
        if batched_graph.ndata['x'].nelement() != 0:
            node_features = batched_graph.ndata['x']
            node_mean = node_features.mean(dim=0, keepdim=True)
            node_std = node_features.std(dim=0, keepdim=True)
            node_std[node_std == 0] = 1 
            batched_graph.ndata['x'] = (node_features - node_mean) / node_std

        # edge feature normalization
        if batched_graph.edata['dataflow'].nelement() != 0:
            edge_features = batched_graph.edata['dataflow']
            edge_mean = edge_features.mean(dim=0, keepdim=True)
            edge_std = edge_features.std(dim=0, keepdim=True)
            edge_std[edge_std == 0] = 1 
            batched_graph.edata['dataflow'] = (edge_features - edge_mean) / edge_std
        norm_info = {
            "n_m": node_mean,
            "n_s": node_std,
            "e_m": edge_mean,
            "e_s": edge_std,
        }
        with open("graph_data/norm_graph/norm_info.pkl","wb") as f:
            pickle.dump(norm_info,f)
        normalized_graphs = dgl.unbatch(batched_graph)
        return dict(zip(list(graphs.keys()),normalized_graphs))


def lambda_rank_loss(pred, label):
    """
    LambdaRank loss function.
    
    Args:
    pred (torch.Tensor): Model predictions.
    label (torch.Tensor): Ground truth labels.
    
    Returns:
    torch.Tensor: LambdaRank loss.
    """
    # Compute difference in predictions
    pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    # Compute difference in relevance
    relevance_diff = label.unsqueeze(1) - label.unsqueeze(0)
    # Compute sign of relevance difference
    relevance_sign = torch.sign(relevance_diff) 
    # Compute detal z
    exp_r = torch.exp(label)
    T_l = exp_r/torch.sum(exp_r)
    T_l = T_l.unsqueeze(1) - T_l.unsqueeze(0)
    # Compute LambdaRank loss
    loss = torch.log(1 + torch.exp(-relevance_sign * pred_diff)) * torch.abs(T_l)
    # Compute mean over all sample pairs
    loss_mean = torch.mean(loss)
    return loss_mean


def gen_value(lflow,pos):
    pos_x = pos[:,0].unsqueeze(1)
    pos_y = pos[:,1].unsqueeze(1)
    num = pos.shape[0]
    dx = abs(pos_x-pos_x.transpose(0,1))
    dy = abs(pos_y-pos_y.transpose(0,1))
    dr = torch.sqrt(torch.square(dx)+torch.square(dy))
    dr_1 = 1.0/dr
    diagonal_indices = torch.arange(num).long()
    dr_1[diagonal_indices,diagonal_indices] = 0
    lf1 = lflow[:,:,:4]
    lf0 = lflow[:,:,4:8]
    lf_1 = lflow[:,:,8:12]
    lf_2 = lflow[:,:,12:]
    dr_1 = torch.clamp(dr_1, min=None, max=100)
    
    results = (lf1 *dr.unsqueeze(-1)+ lf0 + lf_1 * dr_1.unsqueeze(-1)+lf_2*torch.pow(dr_1,2).unsqueeze(-1)).mean(dim=(0, 1))
    return results

def train(model, dataset):
    num_examples = len(dataset)
    num_train = int(num_examples * 0.8)
    num_val = num_examples - num_train
    indices = list(range(num_examples))
    np.random.shuffle(indices)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    val_dataset = Subset(dataset, val_indices)
    train_dataset = Subset(dataset,train_indices)
    val_loader = dgl.dataloading.GraphDataLoader(val_dataset, batch_size=num_val, shuffle=False,collate_fn=collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.985)
    best_val_ken = 0.
    start_rank = 0
    for epoch in range(1000000):
        # Forward
        dataloader = dgl.dataloading.GraphDataLoader(
            train_dataset, batch_size=60, shuffle=True,collate_fn=collate_fn)
        total_loss = 0.0
        total_train_ken = 0.0
        total_val_ken = 0.0
        total_lmse = 0.0
        total_lrank = 0.0
        n_batches = len(dataloader)
        for index, data in enumerate(dataloader):
            model.train()
            bg = data[0]
            pos = restore_original_tensors(data[1], -1000)
            labels = data[2:]
            edge_fea = bg.edata['dataflow']
            lflows = gcn(bg.ndata['x'],edge_fea,bg)
            preds =  []
            for i, lflow in enumerate(lflows):
                pred = gen_value(lflow,pos[i])
                preds.append(pred)
            preds = torch.stack(preds,dim = 1)
            Loss = []
            loss1 = 0
            loss2 = 0
            loss = 0
            Eva = []
            for i,label in enumerate(labels):
                pred = preds[i]
                loss_f1 = F.mse_loss
                eva_f = kendalltau
                L1 = 0.001*loss_f1(pred, label.float()).requires_grad_(True).to(torch.float32)
                loss_f2 = lambda_rank_loss
                L2 = loss_f2(pred,label.float()).requires_grad_(True).to(torch.float32)
                ken = eva_f(pred.cpu().detach().numpy(),label.cpu().detach().numpy()).correlation
                if L1 < 0.0005:
                    start_rank = 1
                if not start_rank:
                    L = L1
                else: 
                    L = L2
                Loss.append((L1,L2))
                loss1 += L1
                loss2 += L2
                loss += L
                Eva.append(ken)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            Eva = np.array(Eva)
            train_ken = Eva.mean()
            with torch.no_grad():
                model.eval()
                for val_data in val_loader:
                    bg = val_data[0]
                    pos = restore_original_tensors(val_data[1], -1000)
                    labels = val_data[2:]
                    edge_fea = bg.edata['dataflow']
                    lflows = gcn(bg.ndata['x'],edge_fea,bg)
                    preds =  []
                    for i, lflow in enumerate(lflows):
                        pred = gen_value(lflow,pos[i])
                        preds.append(pred)
                    preds = torch.stack(preds,dim = 1)
                    Eva = []
                    for i,label in enumerate(labels):
                        pred = preds[i]
                        eva_f = kendalltau
                        ken = eva_f(pred.cpu().detach().numpy(),label.cpu().detach().numpy()).correlation
                        Eva.append(ken)
                    Eva = np.array(Eva)
                    val_ken = Eva.mean()
            if best_val_ken < val_ken:
                best_val_ken = val_ken
                torch.save(model.state_dict(), f'graph_data/model/model.pth')

            total_loss += loss.item()
            total_train_ken += train_ken.item()
            total_val_ken += val_ken.item()
            total_lmse += loss1.item()
            total_lrank += loss2
            writer.add_scalar('Training Loss', loss.item(), epoch * len(dataloader) + index)
            writer.add_scalar('train ken', train_ken.item(), epoch * len(dataloader) + index)
            writer.add_scalar('val ken', val_ken.item(), epoch * len(dataloader) + index)
            writer.add_scalar('lmse', loss1.item(), epoch * len(dataloader) + index)
            writer.add_scalar('lrank', loss2, epoch * len(dataloader) + index)
            writer.add_scalar('hpwl loss', Loss[0][1].item(), epoch * len(dataloader) + index)
            writer.add_scalar('con loss', Loss[1][1].item(), epoch * len(dataloader) + index)
            writer.add_scalar('tns loss', Loss[2][1].item(), epoch * len(dataloader) + index)
            writer.add_scalar('wns loss', Loss[3][1], epoch * len(dataloader) + index)
            writer.add_scalar('hpwl ken', Eva[0], epoch * len(dataloader) + index)
            writer.add_scalar('con ken', Eva[1], epoch * len(dataloader) + index)
            writer.add_scalar('tns ken', Eva[2], epoch * len(dataloader) + index)
            writer.add_scalar('wns ken', Eva[3], epoch * len(dataloader) + index)
            

        if epoch%2 == 0:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']  
        avg_loss = total_loss / n_batches
        avg_train_ken = total_train_ken / n_batches
        avg_val_ken = total_val_ken / n_batches
        avg_lmse = total_lmse / n_batches
        avg_lrank = total_lrank / n_batches

        print(f"Epoch {epoch + 1}")
        print(f"Avg Training Loss: {avg_loss:.4f}")
        print(f"Avg Train Ken: {avg_train_ken:.4f}")
        print(f"Avg Val Ken: {avg_val_ken:.4f}")
        print(f"Avg LMSE: {avg_lmse:.4f}")
        print(f"Avg LRank: {avg_lrank:.4f}")
        print(f"LR: {current_lr:.1e}\n")


if __name__ == "__main__":
    graphs = []
    with open("log.txt", "w") as f:
            f.write("")
    for benchmark in benchmarklist:
        g, _ = dgl.load_graphs(f'benchmarks/{benchmark}/{benchmark}.bin')
        g = g[0].to(device)
        graphs.append(g)
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
    # load dataset
    with open(f"./graph_data/dataset/dataset.pkl","rb") as f:
        data = pickle.load(f)
    pos = data['pos']
    con = data['con']
    tns = data['tns']
    hpwl = data['hpwl']
    wns = data['wns']
    bench = data['bench']
    g = {}
    for i,benchmark in enumerate(bench_ids):
        g[benchmark] = graphs[i]
    dataset = placeset(
        graphs=g,
        pos = pos,
        benchmarks=bench,
        con = con,
        hpwl = hpwl,
        tns = tns,
        wns = wns
    )
    train(dataset = dataset, model = gcn)