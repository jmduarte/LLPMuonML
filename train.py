import torch
from torch_geometric.data import Data, DataLoader
from graph_data import GraphDataset
import os
import os.path as osp
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv

class EdgeNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=1, n_iters=1,aggr='add'):
        super(EdgeNet, self).__init__()
        convnn = nn.Sequential(nn.Linear(2*(hidden_dim + input_dim), hidden_dim),
                               nn.ReLU(),
                               nn.Linear(hidden_dim, hidden_dim),
                               nn.ReLU()
        )
        self.n_iters = n_iters
        
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.edgenetwork = nn.Sequential(nn.Linear(2*(hidden_dim+input_dim),
                                                   output_dim),
                                         nn.Sigmoid())
        
        self.nodenetwork = EdgeConv(nn=convnn,aggr=aggr)

    def forward(self, data):
        X = data.x
        H = self.inputnet(X)
        data.x = torch.cat([H,X],dim=-1)
        for i in range(self.n_iters):
            H = self.nodenetwork(data.x,data.edge_index)
            data.x = torch.cat([H,X],dim=-1)
        row,col = data.edge_index        
        output = self.edgenetwork(torch.cat([data.x[row],data.x[col]],dim=-1)).squeeze(-1)
        print(output.shape)
        return output




#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('using device %s'%device)

full_dataset = GraphDataset(root='data/')
print(full_dataset)

data = full_dataset.get(0)
print(data)
fulllen = len(full_dataset)
tv_frac = 0.10
tv_num = math.ceil(fulllen*tv_frac)
splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
batch_size = 32

train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)

train_samples = len(train_dataset)
valid_samples = len(valid_dataset)

model = EdgeNet(input_dim=4,hidden_dim=8,n_iters=1).to(device)
model.eval()

print(train_loader)
data = data.to(device)
batch_target = data.y
batch_output = model(data)
print(batch_target.shape)
print(batch_output.shape)



  
