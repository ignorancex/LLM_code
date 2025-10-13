import torch


class BinaryDice3D(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(BinaryDice3D, self).__init__()
        self.reduction = reduction
        self.smooth = 1

    def forward(self, predict, target, annotation_mask=None):
        num = torch.sum(predict*target, (2, 3, 4))
        den = torch.sum(predict, (2, 3, 4)) + torch.sum(target, (2, 3, 4))

        dice_score = ((2 * num) + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice_score

        if annotation_mask is not None:
            dice_loss = torch.mean(torch.sum(dice_loss * annotation_mask, -1) / torch.sum(annotation_mask, -1))
        else:
            dice_loss = dice_loss.mean()

        return dice_loss

def apply_l1_shrinkage_to_vector(gating_vector, lambda_t, lr):
    #Applies L1-based shrinkage to enforce sparsity. 
    return torch.sign(gating_vector) * torch.relu(torch.abs(gating_vector) - lambda_t * lr)



def apply_proximal_updates_to_model_gating(model, lambda_t, lr): 
    for name, param in model.named_parameters(): 
        if 'gating_vector' in name and 'qkv' in name: 
            with torch.no_grad():
                updated_param = apply_l1_shrinkage_to_vector(param, lambda_t, lr)
                param.copy_(updated_param)

