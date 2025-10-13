import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

# convs: ['gcn', 'gat', 'gcn2conv', 'sageconv', 'sgconv'] ...
class GRASP(nn.Module):
    def __init__(self, in_dim, hidden_dim_1, hidden_dim_2, n_classes=2, conv_layer='gcn'):
        super(GRASP, self).__init__()
        self.n_classes = n_classes
        self.conv_layer = conv_layer
        if conv_layer == 'gcn':
            self.conv1 = dglnn.GraphConv(in_dim, hidden_dim_1, norm='both', weight=True, bias=True)
            self.conv2 = dglnn.GraphConv(hidden_dim_1, hidden_dim_1, norm='both', weight=True, bias=True)
            self.conv3 = dglnn.GraphConv(hidden_dim_1, hidden_dim_2, norm='both', weight=True, bias=True)
        elif conv_layer == 'gat':
            self.conv1 = dglnn.GATConv(in_dim, hidden_dim_1, num_heads = 1)
            self.conv2 = dglnn.GATConv(hidden_dim_1, hidden_dim_1, num_heads=1)
            self.conv3 = dglnn.GATConv(hidden_dim_1, hidden_dim_2, num_heads=1)
        elif conv_layer == 'gcn2conv':
            self.conv1 = dglnn.GCN2Conv(in_dim, layer=1)
            self.conv2 = dglnn.GraphConv(in_dim, hidden_dim_1, norm='both', weight=True, bias=True)
            self.conv3 = dglnn.GraphConv(hidden_dim_1, hidden_dim_2, norm='both', weight=True, bias=True)
        elif conv_layer == 'sageconv':
            self.conv1 = dglnn.SAGEConv(in_dim, hidden_dim_1, aggregator_type='pool')
            self.conv2 = dglnn.SAGEConv(hidden_dim_1, hidden_dim_1, aggregator_type='pool')
            self.conv3 = dglnn.SAGEConv(hidden_dim_1, hidden_dim_2, aggregator_type='pool')
        elif conv_layer == 'sgconv':
            self.conv1 = dglnn.SGConv(in_dim, hidden_dim_1)
            self.conv2 = dglnn.SGConv(hidden_dim_1, hidden_dim_1)
            self.conv3 = dglnn.SGConv(hidden_dim_1, hidden_dim_2)
        self.attention = nn.Sequential(nn.Linear(hidden_dim_2, hidden_dim_2),
                                       nn.Tanh(),
                                       nn.Linear(hidden_dim_2, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden_dim_2, hidden_dim_2),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim_2, n_classes))

    def forward(self, g, h):
        # Apply graph convolution and activation.
        if self.conv_layer == 'gcn2conv':
            h = F.relu(self.conv1(g, h, h))
        else:
            h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        #print(h.shape)
        with g.local_scope():
            g.ndata['h'] = h
            batch_size = g.batch_size
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            A = self.attention(h)
            out = self.classifier(hg).reshape((batch_size, self.n_classes))
            return out, A
