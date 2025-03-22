import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class RationaleRegularizationLoss(nn.Module):
    def __init__(self, lambda_sparse=0.01, lambda_continuity=0.01):
        """
        Implements rationale regularization as a PyTorch module.

        Args:
            lambda_sparse (float): Coefficient for sparsity loss.
            lambda_continuity (float): Coefficient for continuity loss.
        """
        super(RationaleRegularizationLoss, self).__init__()
        self.lambda_sparse = lambda_sparse
        self.lambda_continuity = lambda_continuity

    def forward(self, pt, mask=None):
        """
        Computes the regularization loss.

        Args:
            pt (torch.Tensor): Probability of each token being part of the rationale.
                               Shape: (batch_size, seq_len, 1)
            mask (torch.Tensor, optional): Mask for padding positions.
                                           Shape: (batch_size, seq_len)

        Returns:
            torch.Tensor: Total regularization loss (scalar).
        """
        pt = pt.squeeze(-1)  

        if mask is not None:
            pt = pt * mask  

        # Sparsity loss: L1 norm sum over sequence, then averaged over the batch
        sparsity_loss = pt.sum(dim=-1).mean()

        # Continuity loss: sum of squared differences between adjacent tokens
        continuity_loss = ((pt[:, 1:] - pt[:, :-1]) ** 2).mean()

        # Weighted sum of the two losses
        total_reg_loss = self.lambda_sparse * sparsity_loss + self.lambda_continuity * continuity_loss

        return total_reg_loss

class RTLoss(nn.Module):
    
    def __init__(self, device = 'cuda'):
        super(RTLoss, self).__init__()
        self.device = device
    
    def forward(self, pt: torch.Tensor, Tagging:  torch.Tensor):
        '''
        Tagging: list paragraphs contain value token. If token of the paragraphas is rationale will labeled 1 and other will be labeled 0 
        
        RT: 
                    p^r_t = sigmoid(w_2*RELU(W_1.h_t))
            
            With:
                    p^r_t constant
                    w_2 (d x 1)
                    W_1 (d x d)
                    h_t (1 x d)
                    
            This formular is compute to each token in paraphase. I has convert into each paraphase
            
                    p^r_t = sigmoid(w_2*RELU(W_1.h))
                    
                    With:
                            p^r (1 x n) with is number of paraphase
                            w_2 (d x 1)
                            W_1 (d x d)
                            h (n x d) 
                            
        '''
        
        Tagging = torch.tensor(Tagging, dtype=torch.float32).to(pt.device)
                
        total_loss = torch.tensor(0, dtype= torch.float32).to(pt.device)
        
        N = pt.shape[0]
                
        for i, text in enumerate(pt):
            T = len(Tagging[i])
            Lrti = -(1/T) * (Tagging[i]@torch.log(text) + (1.0 - Tagging[i]) @ torch.log(1.0 - text) )[0]
            total_loss += Lrti
            
        return total_loss/N


class comboLoss(nn.Module):
    def __init__(self, config):
        
        super(comboLoss, self).__init__()
        self.alpha = config.alpha
        self.beta = config.beta
        # self.BaseLoss = BaseLoss()
        self.RTLoss = RTLoss()
        self.reg_loss_fn = RationaleRegularizationLoss(lambda_sparse=0.01, lambda_continuity=0.01)
        self.config = config
        
    def forward(self, output: dict):
        attention_mask = output['attention_mask']
        start_logits = output['start_logits']
        end_logits = output['end_logits']
        
        start_positions = output['start_positions']
        end_positions = output['end_positions']
        
        Tagging = output['Tagging']
        pt = output['pt']

        loss_base = 0
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            loss_base = (start_loss + end_loss) / 2
        retation_tagg_loss  = self.RTLoss(pt = pt, Tagging = Tagging)
        # retation_tagg_loss = nn.BCELoss()(pt, Tagging)
        # retation_tagg_loss = 0
        reg_loss = self.reg_loss_fn(pt, mask=attention_mask)
        total_loss = self.alpha*loss_base + reg_loss + self.beta*retation_tagg_loss 
        
        return total_loss

