import torch
import torch.nn.functional as F
def marginal_loss(score,labels):
    """
    args:
        score:batch * top_k
        labels: batch * top_k
    """
    predict = F.softmax(score, dim=-1)
    loss = predict * labels #element-wise
    loss = loss.sum(dim=-1)                   # sum all positive scores
    loss = loss[loss > 0]                     # filter sets with at least one positives
    loss = torch.clamp(loss, min=1e-9, max=1) # for numerical stability
    loss = -torch.log(loss)                   # for negative log likelihood
    if len(loss) == 0:
        loss = loss.sum()                     # will return zero loss
    else:
        loss = loss.mean()
    return loss


    