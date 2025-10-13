import torch
import torch.nn as nn
from .module import MPNNLayer, DecLayer, EncLayer
from .feature import RNAFeatures
import torch.nn.functional as F

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


class RDesign_Model(nn.Module):
    def __init__(self, args):
        super(RDesign_Model, self).__init__()

        self.device = 'cuda:0'
        self.smoothing = args.smoothing
        self.node_features = self.edge_features =  args.hidden
        self.hidden_dim = args.hidden
        self.vocab = args.vocab_size

        self.features = RNAFeatures(
            args.hidden, args.hidden, 
            top_k=args.k_neighbors, 
            dropout=args.dropout,
            node_feat_types=args.node_feat_types, 
            edge_feat_types=args.edge_feat_types,
            args=args
        )

        layer = MPNNLayer
        self.W_s = nn.Embedding(args.vocab_size, self.hidden_dim)
        self.W_E = nn.Linear(args.hidden, self.hidden_dim)
        self.W_V = nn.Linear(args.hidden, self.hidden_dim)
        
        self.encoder_layers = nn.ModuleList([
            EncLayer(self.hidden_dim, self.hidden_dim*2, dropout=args.dropout)
            for _ in range(args.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([
            DecLayer(self.hidden_dim, self.hidden_dim*3, dropout=args.dropout)
            for _ in range(args.num_decoder_layers)])

        self.projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        )

        self.readout = nn.Linear(self.hidden_dim, args.vocab_size, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def cat_neighbors_nodes(self, h_nodes, h_neighbors, E_idx):
        h_nodes = gather_nodes(h_nodes, E_idx)
        h_nn = torch.cat([h_neighbors, h_nodes], -1)
        return h_nn

    def forward(self, X, S, mask):
        X, gt_S, h_V, h_E, E_idx, batch_id, mask_attend = self.features(X, S, mask) 
        # supposed to be
        # h_V.shape, h_E.shape, E_idx.shape torch.Size([1, 118, 1536]) torch.Size([1, 118, 48, 1536]) torch.Size([1, 118, 48])
        # now
        # h_V.shape, h_E.shape, E_idx.shape torch.Size([2804, 128]) torch.Size([77187, 128]) torch.Size([2, 77187])
        #  , ,[64, 447, 30])
        #mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        #mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = self.cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = self.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = self.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=h_E.device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=h_E.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = self.cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)

        log_probs = self.readout(h_V)
        return torch.masked_select(log_probs, (mask==1).unsqueeze(-1)).view(-1, 4), torch.masked_select(S, (mask==1)), mask
        #return log_probs, S, mask

    def sample(self, X, S_gtt, mask=None):
        X, gt_S, h_V, h_E, E_idx, batch_id, mask_attend = self.features(X, S_gtt, mask) 
        # supposed to be
        # h_V.shape, h_E.shape, E_idx.shape torch.Size([1, 118, 1536]) torch.Size([1, 118, 48, 1536]) torch.Size([1, 118, 48])
        # now
        # h_V.shape, h_E.shape, E_idx.shape torch.Size([2804, 128]) torch.Size([77187, 128]) torch.Size([2, 77187])
        #  , ,[64, 447, 30])
        #mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        #mask_attend = mask.unsqueeze(-1) * mask_attend

        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

        chain_M = mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=h_E.device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=h_E.device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        N_batch, N_nodes = mask.size(0), mask.size(1)
        all_probs = torch.zeros((N_batch, N_nodes, 4), device=X.device, dtype=torch.float32)
        h_S = torch.zeros_like(h_V, device=X.device)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=X.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V, device=X.device) for _ in range(len(self.decoder_layers))]

        #h_ES = self.cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = self.cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = self.cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        
        for t_ in range(N_nodes):
            t = decoding_order[:, t_]
            E_idx_t = torch.gather(E_idx, 1, t[:,None,None].repeat(1,1,E_idx.shape[-1]))
            h_E_t = torch.gather(h_E, 1, t[:,None,None,None].repeat(1,1,h_E.shape[-2], h_E.shape[-1]))
            h_ES_t = self.cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            h_EXV_encoder_t = torch.gather(h_EXV_encoder_fw, 1, t[:,None,None,None].repeat(1,1,h_EXV_encoder_fw.shape[-2], h_EXV_encoder_fw.shape[-1]))
            mask_t = torch.gather(mask, 1, t[:,None])
            for l, layer in enumerate(self.decoder_layers):
                # Updated relational features for future states
                h_ESV_decoder_t = self.cat_neighbors_nodes(h_V_stack[l], h_ES_t, E_idx_t)
                h_V_t = torch.gather(h_V_stack[l], 1, t[:,None,None].repeat(1,1,h_V_stack[l].shape[-1]))
                h_ESV_t = torch.gather(mask_bw, 1, t[:,None,None,None].repeat(1,1,mask_bw.shape[-2], mask_bw.shape[-1])) * h_ESV_decoder_t + h_EXV_encoder_t
                h_V_stack[l+1].scatter_(1, t[:,None,None].repeat(1,1,h_V.shape[-1]), layer(h_V_t, h_ESV_t, mask_V=mask_t))
            # Sampling step
            h_V_t = torch.gather(h_V_stack[-1], 1, t[:,None,None].repeat(1,1,h_V_stack[-1].shape[-1]))[:,0]
            logits = self.readout(h_V_t)
            prob = F.softmax(logits, dim=-1)
            S_t = torch.multinomial(prob, 1)
            all_probs.scatter_(1, t[:,None,None].repeat(1,1,4), (logits[:,None,:]).float())
            temp1 = self.W_s(S_t)
            h_S.scatter_(1, t[:,None,None].repeat(1,1,temp1.shape[-1]), temp1)
            S.scatter_(1, t[:,None], S_t)
            
        return torch.masked_select(all_probs, (mask==1).unsqueeze(-1)).view(-1, 4), torch.masked_select(S_gtt, (mask==1))
        #return logits, gt_S