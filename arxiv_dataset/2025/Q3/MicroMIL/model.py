import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from dgl.nn import GATConv, GCNConv
from torchvision.models import regnet_y_400mf,resnet18,resnet34
from typing import Optional
import dgl
class DEC(nn.Module):

    def __init__(self, cluster_number: int, embedding_dimension: int, alpha: float = 1.0, cluster_centers: Optional[torch.Tensor] = None):
        super(DEC, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        initial_cluster_centers = cluster_centers if cluster_centers is not None else torch.zeros(self.cluster_number, self.embedding_dimension, dtype=torch.float)
        nn.init.xavier_uniform_(initial_cluster_centers)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        #print(batch.size())
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power

        cluster_assignment =numerator / torch.sum(numerator, dim=1, keepdim=True)
        return  cluster_assignment

class ClusterAttentionModel(nn.Module):
    def __init__(self, args,in_dim=512, hidden_dim=128, n_classes=2,drop_rate = 0.5):
        super(ClusterAttentionModel, self).__init__()

        if args.model_name == 'resnet18':
            self.feature_extractor = resnet18(pretrained=True).to(args.device)
            in_dim = 512
        elif  args.model_name == 'resnet34':
            self.feature_extractor = resnet34(pretrained=True).to(args.device)
            in_dim = 512
        elif args.model_name == 'regnet400':
            self.feature_extractor = regnet_y_400mf(pretrained=True).to(args.device)
            in_dim = 440

        self.feature_extractor.fc = nn.Identity()
        self.image_feature = nn.Linear(in_dim,in_dim)
        self.cluster_number =args.cluster_number
        self.dec = DEC(cluster_number=args.cluster_number, embedding_dimension=in_dim)
        self.attn = nn.Linear(in_dim, 1)
        self.layers = nn.ModuleList([GATConv(in_dim, hidden_dim, num_heads=1)])
        for _ in range(args.layer - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim, num_heads=1))
        self.classify = nn.Linear(hidden_dim, n_classes)

        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop_rate)

        self.args = args
        self.shuffle  = args.shuffle

    def forward(self, bags):
        instances = bags.squeeze(0)
        if len(instances) ==1:
            instances = torch.cat([instances,instances])
        batch_size = 512
        batch_instances = []
        with torch.no_grad():
            for i in range(0, len(instances), batch_size):
                batch_instance = instances[i:i+batch_size]
                batch_instances.extend(self.feature_extractor(batch_instance))
                del batch_instance
                torch.cuda.empty_cache()
        batch_instances =self.leaky_relu(self.image_feature(torch.stack(batch_instances,dim=0)))

        cluster_assignments = self.dec(batch_instances)
        
        self.cluster_assignments = [F.gumbel_softmax(self.attn(batch_instances * cluster_assignments[:, i:i+1]).squeeze(), dim=0, hard=True) for i in range(self.cluster_number)]
        
        gumbel_scores = torch.stack(self.cluster_assignments, dim=1)
        rep_features = torch.matmul(gumbel_scores.T, batch_instances)
        
        grid, dim = rep_features.size()

        similarity_matrix = F.cosine_similarity(rep_features.unsqueeze(1), rep_features.unsqueeze(0), dim=2) 
        self.attention_scores = F.gumbel_softmax(similarity_matrix.view(-1, grid), hard=True).view(-1, grid, grid)

        nonzero_indices = self.attention_scores.nonzero(as_tuple=True)
        x =nonzero_indices[1]

        if self.shuffle == 'shuffle':
            x = x[torch.randperm(x.size(0))]

        g = dgl.graph((x, nonzero_indices[2])).to(self.args.device) # Move graph to CPU
        g = dgl.add_self_loop(g)
        g = g.to(self.args.device)  # Move graph back to CUDA if necessary

        h = rep_features.view(-1, dim)

        for layer in self.layers:
            h = layer(g, h)
            h = self.dropout(h)
            h = self.leaky_relu(h)
        g.ndata['h'] = h
        self.h = g.ndata['h']
        output = self.leaky_relu(self.classify(dgl.mean_nodes(g, 'h')).squeeze())
        return output
