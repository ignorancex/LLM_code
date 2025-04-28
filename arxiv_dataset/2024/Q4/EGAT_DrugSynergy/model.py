from torch_geometric.nn import GATv2Conv, global_max_pool, Sequential, GCNConv
from torch_geometric.nn.norm import GraphNorm
from torch.nn import Linear, Sequential, ELU, Sigmoid, GELU
import torch
import torch.nn.functional as F
from EGNN import EGNN


class Encoder(torch.nn.Module):
    def __init__(self, embedding_size=256, motif_embedding_size=32, heads=10, dropout=0.2, layers=3): # 32 worked well
        super(Encoder, self).__init__()

        self.motif_embedding_size = motif_embedding_size
        self.embedding_size = embedding_size
        self.heads = heads
        self.dropout = dropout
        self.layers = layers

        self.Drug1Conv = EGNN(49, self.embedding_size*2, heads=self.heads, dropout=self.dropout)
        self.GraphNorm1 = GraphNorm(self.embedding_size * 2)
        self.head_transform = Linear(self.embedding_size * self.heads * 2, self.embedding_size * 2)
        self.head_transform1 = Linear(self.embedding_size * self.heads, self.embedding_size)

        self.Conv1 = EGNN(self.embedding_size*2, self.embedding_size, heads=self.heads, dropout=self.dropout)
        self.GraphNorm2 = GraphNorm(self.embedding_size)
        self.Conv = EGNN(self.embedding_size, self.embedding_size, heads=self.heads, dropout=self.dropout)

        self.Motif1Conv = GATv2Conv(-1, self.motif_embedding_size * 2, heads=self.heads, dropout=self.dropout, edge_dim=10)
        self.MotifGraphNorm1 = GraphNorm(self.motif_embedding_size * self.heads * 2)

        self.Motifhead_transform1 = Linear(self.motif_embedding_size * self.heads * 2, self.motif_embedding_size * 2)
        self.Motifhead_transform = Linear(self.motif_embedding_size * self.heads, self.motif_embedding_size)

        self.MotifConv1 = GATv2Conv(self.motif_embedding_size*2, self.motif_embedding_size, heads=self.heads, dropout=self.dropout, edge_dim=10)
        self.MotifGraphNorm2 = GraphNorm(self.motif_embedding_size*self.heads)
        self.MotifConv = GATv2Conv(self.motif_embedding_size, self.motif_embedding_size, heads=self.heads, dropout=self.dropout, edge_dim=10)

        self.ExpressionLinear = Sequential(
            Linear(25266, 2048).double(),
            ELU(),
            Linear(2048, 1024).double(),
            ELU(),
            Linear(1024, embedding_size*2 + motif_embedding_size*2).double(),
        )

        self.FC_Linear = Sequential(
            Linear(embedding_size*2 + motif_embedding_size*2, embedding_size * 2 + motif_embedding_size * 2),
            ELU(),
            Linear(embedding_size * 2 + motif_embedding_size * 2, 512),
            ELU(),
            Linear(512, 256),
            ELU(),
            Linear(256, 128),
            ELU(),
        )

    def forward(self, drug1_x, drug1_edge_index, drug1_edge_attr, drug1_coord, drug1_batch, drug2_x,
                drug2_edge_index, drug2_edge_attr, drug2_coord, drug2_batch, motif1_x, motif1_edge_index,
                motif1_batch, motif2_x, motif2_edge_index, motif2_batch, gene_expression):

        gene_expression = gene_expression.reshape(int(gene_expression.shape[0]/25266), 25266)

        # Process Drug 1
        drug1_x1, coord = self.Drug1Conv(drug1_x, drug1_edge_index, drug1_coord, drug1_edge_attr)
        drug1_x1 = self.head_transform(drug1_x1)
        drug1_x1 = self.GraphNorm1(drug1_x1, drug1_batch)

        drug1_x2, coord = self.Conv1(drug1_x1, drug1_edge_index, coord, drug1_edge_attr)
        drug1_x2 = self.head_transform1(drug1_x2)
        drug1_x2 = self.GraphNorm2(drug1_x2, drug1_batch)

        for i in range(self.layers-2):
            drug1_x3, coord = self.Conv(drug1_x2, drug1_edge_index, coord, drug1_edge_attr)
            drug1_x3 = self.head_transform1(drug1_x3)
            drug1_x3 = self.GraphNorm2(drug1_x3, drug1_batch)

        # Process Drug 2
        drug2_x1, coord = self.Drug1Conv(drug2_x, drug2_edge_index, drug2_coord, drug2_edge_attr)
        drug2_x1 = self.head_transform(drug2_x1)
        drug2_x1 = self.GraphNorm1(drug2_x1, drug2_batch)

        drug2_x2, coord = self.Conv1(drug2_x1, drug2_edge_index, coord, drug2_edge_attr)
        drug2_x2 = self.head_transform1(drug2_x2)
        drug2_x2 = self.GraphNorm2(drug2_x2, drug2_batch)

        for i in range(self.layers-2):
            drug2_x3, coord = self.Conv(drug2_x2, drug2_edge_index, coord, drug2_edge_attr)
            drug2_x3 = self.head_transform1(drug2_x3)
            drug2_x3 = self.GraphNorm2(drug2_x3, drug2_batch)

        # Process Motif 1
        motif1_x1 = F.relu(self.Motif1Conv(motif1_x, motif1_edge_index))
        motif1_x1 = self.MotifGraphNorm1(motif1_x1, motif1_batch)
        motif1_x1 = F.relu(self.Motifhead_transform1(motif1_x1))

        motif1_x2 = F.relu(self.MotifConv1(motif1_x1, motif1_edge_index))
        motif1_x2 = self.MotifGraphNorm2(motif1_x2, motif1_batch)
        motif1_x2 = F.relu(self.Motifhead_transform(motif1_x2))

        for i in range(self.layers-2):
            motif1_x3 = F.relu(self.MotifConv(motif1_x2, motif1_edge_index))
            motif1_x3 = self.MotifGraphNorm2(motif1_x3, motif1_batch)
            motif1_x3 = F.relu(self.Motifhead_transform(motif1_x3))

        # Process Motif 2
        motif2_x1 = F.relu(self.Motif1Conv(motif2_x, motif2_edge_index))
        motif2_x1 = self.MotifGraphNorm1(motif2_x1, motif2_batch)
        motif2_x1 = F.relu(self.Motifhead_transform1(motif2_x1))

        motif2_x2 = F.relu(self.MotifConv1(motif2_x1, motif2_edge_index))
        motif2_x2 = self.MotifGraphNorm2(motif2_x2, motif2_batch)
        motif2_x2 = F.relu(self.Motifhead_transform(motif2_x2))

        for i in range(self.layers-2):
            motif2_x3 = F.relu(self.MotifConv(motif2_x2, motif2_edge_index))
            motif2_x3 = self.MotifGraphNorm2(motif2_x3, motif2_batch)
            motif2_x3 = F.relu(self.Motifhead_transform(motif2_x3))

        expression_x = self.ExpressionLinear(gene_expression)

        x = F.relu(F.normalize(torch.cat([global_max_pool(drug1_x3, drug1_batch),
                                          global_max_pool(drug2_x3, drug2_batch),
                                          global_max_pool(motif1_x3, motif1_batch),
                                          global_max_pool(motif2_x3, motif2_batch)],
                                         dim=1), 2, 1) + F.normalize(expression_x, 2, 1))

        x = self.FC_Linear(x)

        return x


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.classifier = Sequential(Linear(128, 1), Sigmoid())

    def forward(self, x):
        x = self.classifier(x)

        return x.reshape(x.shape[0])