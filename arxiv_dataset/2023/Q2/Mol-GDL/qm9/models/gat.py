import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
import pandas as pd
import numpy as np

# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=62, n_output=2, num_features_xt=954, output_dim=128, dropout=0.2, num_net=5, file=None):
        super(GATNet, self).__init__()
        self.num_net = num_net
        self.drug1_gcn1 = nn.ModuleList([GATConv(num_features_xd, output_dim, heads=10, dropout=dropout) for i in range(num_net)])
        self.drug1_gcn2 = nn.ModuleList([GATConv(output_dim * 10, output_dim, dropout=dropout) for i in range(num_net)])
        self.drug1_fc_g1 = nn.ModuleList([nn.Linear(output_dim , output_dim) for i in range(num_net)])  
        
        
#        self.fc1_otherNet_re = nn.Linear(output_dim, int(output_dim/(num_net-1)))
#        self.fc1_otherNet_re = nn.Linear(output_dim, 8)
        
        self.fc1_single = nn.Linear(output_dim * num_net, output_dim)
        
        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim * 2),
            nn.ReLU()
        )


        # combined layers
        self.fc1 = nn.Linear(output_dim * 4, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def get_col_index(self, x):
        row_size = len(x[:, 0])
        row = np.zeros(row_size)
        col_size = len(x[0, :])
        for i in range(col_size):
            row[np.argmax(x[:, i])] += 1
        return row

    def save_num(self, d, path):
        d = d.cpu().numpy()
        ind = self.get_col_index(d)
        ind = pd.DataFrame(ind)
        ind.to_csv('data/case_study/' + path + '_index.csv', header=0, index=0)
        # 下面是load操作
        # read_dictionary = np.load('my_file.npy').item()
        # d = pd.DataFrame(d)
        # d.to_csv('data/result/' + path + '.csv', header=0, index=0)

    def forward(self, data1, data2, device):
#        print("进来了")
#        print(len(data1))
        for i, module in enumerate(zip(self.drug1_gcn1,self.drug1_gcn2,self.drug1_fc_g1)):
            data11 = data1[i].to(device)
            data22 = data2[i].to(device)
            
            x1, edge_index1, batch1, cell = data11.x, data11.edge_index, data11.batch, data11.cell
            x2, edge_index2, batch2 = data22.x, data22.edge_index, data22.batch

            drug1_gcn1 = module[0]
            drug1_gcn2 = module[1]
            drug1_fc_g1 = module[2]

            x1 = drug1_gcn1(x1, edge_index1)
            x1 = F.elu(x1)
            x1 = F.dropout(x1, p=0.2, training=self.training)
            x1 = drug1_gcn2(x1, edge_index1)
            x1 = F.elu(x1)
            x1 = F.dropout(x1, p=0.2, training=self.training)
    
            batch1 = batch1.type(torch.LongTensor)
            x1 = gmp(x1, batch1)         # global max pooling
            x1 = drug1_fc_g1(x1)
            x1 = self.relu(x1)


            # deal drug2
            # begin_x2 = np.array(x2.cpu().detach().numpy())
            x2 = drug1_gcn1(x2, edge_index2)
            x2 = F.elu(x2)
            x2 = F.dropout(x2, p=0.2, training=self.training)
            x2 = drug1_gcn2(x2, edge_index2)
            x2 = F.elu(x2)
            x2 = F.dropout(x2, p=0.2, training=self.training)
            # fin_x2 = np.array(x2.cpu().detach().numpy())
            
            batch2 = batch2.type(torch.LongTensor)
            x2 = gmp(x2, batch2)  # global max pooling
            x2 = drug1_fc_g1(x2)
            x2 = self.relu(x2)


#            if i == 0:
#                x11 = x1
#                x22 = x2
#            else:
##                x1 = self.fc1_otherNet_re(x1)
##                x2 = self.fc1_otherNet_re(x2)
#                
#                x11 = torch.cat((x11, x1), 1)
#                x22 = torch.cat((x22, x2), 1)
                
            if i == 0:
                x11 = x1 * 1/self.num_net
                x22 = x2 * 1/self.num_net
            else:
                
                x11 = x11 + x1 * 1/self.num_net
                x22 = x22 + x2 * 1/self.num_net

#        x11 = self.fc1_single(x11) 
#        x22 = self.fc1_single(x22) 

        # deal cell
        cell = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell)

        # concat
        xc = torch.cat((x11, x22, cell_vector), 1)
        xc = F.normalize(xc, 2, 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
