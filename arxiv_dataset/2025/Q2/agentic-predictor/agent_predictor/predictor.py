import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch_geometric.nn import GCNConv, GINEConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import aggr, global_mean_pool, global_max_pool

class CrossGraphAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_views):
        # x_views: [num_nodes, num_views, dim]
        residual = x_views
        x_attn, _ = self.attn(x_views, x_views, x_views)  # self-attention across views
        x_out = self.norm(x_attn + residual)
        return x_out  # shape: [num_nodes, num_views, dim]


class ViewAttentionPooling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.attn_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_views):
        # x_views: [num_nodes, num_views, dim]
        scores = self.attn_proj(x_views)  # [num_nodes, num_views, 1]
        weights = torch.softmax(scores, dim=1)  # [num_nodes, num_views, 1]
        fused = (weights * x_views).sum(
            dim=1
        )  # weighted sum over views → [num_nodes, dim]
        return fused


# 3 Graphs -> Multi-View Graph
# 2 Global Contexts (Text + Code Views)
class AgentPredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        n_layers=2,
        n_mlplayers=2,
        dropout=0.5,
        base_conv=GCNConv,
        views="multi-view",
    ):
        super(AgentPredictor, self).__init__()
        # define graph views
        self.gnn_prompts = nn.ModuleList()
        # self.gnn_ops = nn.ModuleList()
        self.gnn_codes = nn.ModuleList()
        self.base_conv = base_conv
        self.views = sorted(views) if isinstance(views, list) else views

        if self.base_conv in [GCNConv]:
            self.gnn_prompts.append(base_conv(input_dim, hidden_dim))
            # self.gnn_ops.append(base_conv(input_dim, hidden_dim))
            self.gnn_codes.append(base_conv(input_dim * 2, hidden_dim))
            for _ in range(n_layers - 1):
                self.gnn_prompts.append(base_conv(hidden_dim, hidden_dim))
                # self.gnn_ops.append(base_conv(hidden_dim, hidden_dim))
                self.gnn_codes.append(base_conv(hidden_dim, hidden_dim))

        elif self.base_conv in [GATConv, GATv2Conv, TransformerConv]:
            self.atten_norm = nn.LayerNorm(hidden_dim)
            self.gnn_prompts.append(base_conv(input_dim, hidden_dim // 8, heads=8))
            # self.gnn_ops.append(base_conv(input_dim, hidden_dim // 8, heads=8))
            self.gnn_codes.append(base_conv(input_dim * 2, hidden_dim // 8, heads=8))
            for i in range(n_layers - 1):
                self.gnn_prompts.append(base_conv(hidden_dim, hidden_dim // 8, heads=8))
                # self.gnn_ops.append(base_conv(hidden_dim, hidden_dim // 8, heads=8))
                self.gnn_codes.append(base_conv(hidden_dim, hidden_dim // 8, heads=8))

        # dropout rate
        self.dropout = dropout

        # cross-view attention and pooling
        self.cross_attention = CrossGraphAttention(dim=hidden_dim, num_heads=8)
        self.view_fusion = ViewAttentionPooling(dim=hidden_dim)

        # global-code view
        self.code_projector = nn.ModuleList()
        self.code_projector.append(nn.Linear(input_dim * 2, hidden_dim))
        for i in range(n_mlplayers):
            self.code_projector.append(nn.Linear(hidden_dim, hidden_dim))
            
        # global-text view
        self.inst_projector = nn.ModuleList()
        self.inst_projector.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_mlplayers):
            self.inst_projector.append(nn.Linear(hidden_dim, hidden_dim))

        # Contrastive projection head (fuse all 5 views)
        self.view_projector = nn.ModuleList()

        if self.views == "multi-view" or isinstance(self.views, list):
            view_input_dim = 0
            if self.views == "multi-view":
                view_input_dim = hidden_dim + (input_dim * 2) + input_dim
            elif self.views == sorted(["code", "multi-graph"]) or self.views == sorted(
                ["code", "graph"]
            ):
                view_input_dim = hidden_dim + input_dim * 2
            elif self.views == sorted(["code", "text"]):
                view_input_dim = hidden_dim + input_dim
            elif self.views == sorted(["text", "multi-graph"]) or self.views == sorted(
                ["text", "graph"]
            ):
                view_input_dim = hidden_dim + input_dim

            self.view_projector.append(nn.Linear(view_input_dim, hidden_dim))
            for i in range(n_mlplayers):
                self.view_projector.append(nn.Linear(hidden_dim, hidden_dim))

        # projection layers for task embedding
        self.task_projector = nn.ModuleList()
        self.task_projector.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_mlplayers):
            self.task_projector.append(nn.Linear(hidden_dim, hidden_dim))

        # prediction heads
        self.final_mlp_layers = nn.ModuleList()
        self.final_mlp_layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
        for i in range(n_mlplayers - 1):
            self.final_mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_mlp_layers.append(nn.Linear(hidden_dim, 1))

    def reset_parameters(self):
        for gnn in self.gnn_prompts:
            gnn.reset_parameters()
        for gnn in self.gnn_codes:
            gnn.reset_parameters()

        self.view_projector.reset_parameters()
        self.final_mlp_layers.reset_parameters()
        self.inst_projector.reset_parameters()
        self.code_projector.reset_parameters()
        self.task_projector.reset_parameters()

    def forward_gnns(self, gnn_layers, node_embedding, edge_index):
        for i, gnn in enumerate(gnn_layers):
            if edge_index.dtype != torch.long:
                edge_index = edge_index.type(torch.long)
            if self.base_conv in [
                GCNConv,
                GATConv,
                GATv2Conv,
                TransformerConv,
            ]:
                node_embedding = gnn(node_embedding, edge_index)
                if self.base_conv in [GATConv, GATv2Conv, TransformerConv]:
                    node_embedding = self.atten_norm(node_embedding)
            else:
                raise NotImplementedError
            if i != len(gnn_layers) - 1:
                node_embedding = F.relu(node_embedding)
                node_embedding = F.dropout(
                    node_embedding, p=self.dropout, training=self.training
                )
        return node_embedding

    def forward(self, data, args):
        batch_size = data.y.shape[0]
        # [x] graph (node features with edges): prompts, operator code blocks, implementation function calls
        # [data.graph_prompts, data.node_ops, data.node_code vs. data.x]
        # node_embedding = data.x  # i.e., data.graph_prompts
        graph_prompts = data.graph_prompts
        graph_ops = data.graph_ops
        graph_codes = data.graph_codes

        # [x] text (full instruction prompt): full_prompts [data.text]
        instruction_embedding = data.instruction_text
        # [x] code (full workflow code): workflow_code [data.code]
        workflow_embedding = data.workflow_code

        edge_index = data.edge_index
        batch = data.batch

        # VIEW-SPECIFIC LEARNING
        if args.encoding == "graph":
            node_embedding = self.forward_gnns(
                self.gnn_prompts, graph_prompts, edge_index
            )

        elif (
            args.encoding in ["multi-graph", "multi-view"]
            or "multi-graph" in args.encoding
        ):
            # Encode each view separately with graph encoder
            node_embedding_prompt = self.forward_gnns(
                self.gnn_prompts, graph_prompts, edge_index
            )
            node_embedding_ops = self.forward_gnns(
                self.gnn_codes, graph_ops, edge_index
            )
            node_embedding_code = self.forward_gnns(
                self.gnn_codes, graph_codes, edge_index
            )

            node_embedding = torch.stack(
                [node_embedding_prompt, node_embedding_ops, node_embedding_code], dim=1
            )

            # Apply cross-view attention across graph-encoded views # [num_nodes, num_views, out_dim]
            node_embedding = self.cross_attention(node_embedding)
            node_embedding = self.view_fusion(node_embedding)

        # GLOBAL-CONTEXT LEARNING
        if (
            args.encoding in ["graph", "multi-graph", "multi-view"]
            or "multi-graph" in args.encoding
        ):
            # summarize graph features
            pooled_node_embedding = global_mean_pool(node_embedding, batch)
            pooled_node_embedding = F.normalize(pooled_node_embedding, p=2, dim=1)
            if args.encoding != "multi-view":
                embedding = pooled_node_embedding

        elif args.encoding in ["code", "multi-view"] or "code" in args.encoding:
            # process workflow code embedding
            for i, proj in enumerate(self.code_projector):
                workflow_embedding = proj(workflow_embedding)
                if i != len(self.code_projector) - 1:
                    workflow_embedding = F.relu(workflow_embedding)
                    workflow_embedding = F.dropout(
                        workflow_embedding, p=self.dropout, training=self.training
                    )
            workflow_embedding = F.normalize(workflow_embedding, p=2, dim=1)
            if args.encoding != "multi-view":
                embedding = workflow_embedding

        elif args.encoding in ["text", "multi-view"] or "text" in args.encoding:
            # process instruction prompt embedding
            for i, proj in enumerate(self.inst_projector):
                instruction_embedding = proj(instruction_embedding)
                if i != len(self.inst_projector) - 1:
                    instruction_embedding = F.relu(instruction_embedding)
                    instruction_embedding = F.dropout(
                        instruction_embedding,
                        p=self.dropout,
                        training=self.training,
                    )
            instruction_embedding = F.normalize(instruction_embedding, p=2, dim=1)
            if args.encoding != "multi-view":
                embedding = instruction_embedding

        # concat with node_embedding before project heads
        if args.encoding == "multi-view" or isinstance(args.encoding, list):
            # final embedding used to combine with input task embedding for final prediction
            if args.encoding == "multi-view":
                embedding = torch.cat(
                    [pooled_node_embedding, workflow_embedding, instruction_embedding],
                    dim=1,
                )
            else:
                embeddings = []
                for view in args.encoding:
                    if view == "text":
                        embeddings.append(instruction_embedding)
                    elif view == "code":
                        embeddings.append(workflow_embedding)
                    elif view in ["graph", "multi-graph"]:
                        embeddings.append(pooled_node_embedding)
                embedding = torch.cat(embeddings, dim=1)
            for i, proj in enumerate(self.view_projector):
                embedding = proj(embedding)
                if i != len(self.view_projector) - 1:
                    embedding = F.relu(embedding)
                    embedding = F.dropout(
                        embedding, p=self.dropout, training=self.training
                    )

        if args.mode in ["train", "train-test"]:
            # [x] task embedding: task [data.task_embedding]
            # only require for supervised classification case
            task_embedding = data.task_embedding
            if isinstance(task_embedding, list):
                task_embedding = np.array(task_embedding)
                task_embedding = torch.tensor(task_embedding).to("cuda")

            # task embedding -> mlps
            for i, proj in enumerate(self.task_projector):
                task_embedding = proj(task_embedding)
                if i != len(self.task_projector) - 1:
                    task_embedding = F.relu(task_embedding)
                    task_embedding = F.dropout(
                        task_embedding, p=self.dropout, training=self.training
                    )
            task_embedding = F.normalize(task_embedding, p=2, dim=1)

            # process combining for projection heads
            embedding = torch.cat([embedding, task_embedding], dim=1)
            for i, proj in enumerate(self.final_mlp_layers):
                embedding = proj(embedding)
                if i != len(self.final_mlp_layers) - 1:
                    embedding = F.relu(embedding)
                    embedding = F.dropout(
                        embedding, p=self.dropout, training=self.training
                    )

            # binary classification
            score = torch.sigmoid(embedding).reshape(-1)
            return score

        elif args.mode == "pretrain":
            # process multi-task self-supervised learning with reconstruction and contrastive losses
            # return reconstructed_input and embedding to compute losses in the pretraining loop
            return reconstructed_input, embedding
