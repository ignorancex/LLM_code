import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

__all__ = ['InfoNCE', 'info_nce']



class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None, symmetric=False):
        return self.info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode,
                        symmetric=symmetric)


    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired', symmetric=False):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors (normalize by euclidean distance)
        query, positive_key, negative_keys = self.normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ self.transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ self.transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ self.transpose(positive_key)

            # Positive keys are the entries on the diagonal (class indices for each row)
            labels = torch.arange(len(query), device=query.device)
            
            # symmetric contrastive loss 
            if symmetric:
                logits2 = positive_key @ self.transpose(query)
                loss = 0.5*F.cross_entropy(logits / temperature, labels, reduction=reduction) + 0.5*F.cross_entropy(logits2 / temperature, labels, reduction=reduction)
            else:
                loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
                
        return loss

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class InfoNCE_with_false_negative_elimination(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning using only the top percent% most dissimilar negative samples in each batch.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
    and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with a selected subset of negative keys.

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
 
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).

    Returns:
         Value of the InfoNCE Loss.

    Example:
        >>> loss_fn = InfoNCE_onlymostdissimilar(temperature=0.1)
        >>> batch_size, embedding_size = 32, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> loss = loss_fn(query, positive_key, symmetric=True)
    """
    def __init__(self, temperature=0.1, reduction='mean', percent=0.5):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.percent = percent

    def forward(self, query, positive_key, symmetric=False):
        return self.info_nce(query, positive_key,
                             temperature=self.temperature,
                             reduction=self.reduction,
                             symmetric=symmetric)

    def info_nce(self, query, positive_key, temperature=0.1, reduction='mean', symmetric=False):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')

        # Check matching number of samples.
        if query.shape[0] != positive_key.shape[0]:
            raise ValueError('<query> and <positive_key> must have the same number of samples.')

        # Embedding vectors should have the same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')

        # Normalize to unit vectors.
        query, positive_key = self.normalize(query, positive_key)

        # Cosine similarity between all combinations.
        logits = query @ self.transpose(positive_key)

        # print(logits.mean(), logits.std(), logits.min(), logits.max())
        # Compute filtered loss for one direction.
        loss_a = self.compute_filtered_loss(logits, temperature)

        if symmetric:
            # Compute the symmetric loss (swapping query and positive_key)
            logits2 = positive_key @ self.transpose(query)
            loss_b = self.compute_filtered_loss(logits2, temperature)
            loss = 0.5 * (loss_a + loss_b)
        else:
            loss = loss_a

        return loss

    def compute_filtered_loss(self, logits, temperature):
        """
        For each row in the logits matrix, this function:
            - Extracts the positive logit (diagonal element).
            - Collects the negative logits (all off-diagonals).
            - Sorts negatives in ascending order (lowest similarity first, i.e. most dissimilar).
            - Selects the top 25% of negatives (at least one).
            - Constructs a new logits vector [positive, selected_negatives].
            - Computes cross-entropy loss with the correct label at index 0.
        Finally, the loss is averaged over all rows.
        """
        losses = []
        N = logits.shape[0]
        for i in range(N):
            row_logits = logits[i]  # shape: (N,)
            # Positive sample is at index i.
            positive_logit = row_logits[i].unsqueeze(0)  # shape: (1,)
            # Gather negatives: all elements except the diagonal.
            if i == 0:
                negative_logits = row_logits[1:]
            elif i == N - 1:
                negative_logits = row_logits[:N-1]
            else:
                negative_logits = torch.cat([row_logits[:i], row_logits[i+1:]], dim=0)
            # Determine number of negatives to select: top % most dissimilar.
            k = max(1, int(self.percent * negative_logits.shape[0]))
            # Sort negatives in ascending order (lowest similarity = most dissimilar).
            sorted_negatives, sorted_indices = torch.sort(negative_logits, descending=False)
            selected_negatives = sorted_negatives[:k]
            # Construct the new logits vector with the positive first.
            new_logits = torch.cat([positive_logit, selected_negatives], dim=0)
            # The label is 0 because the positive sample is at the first position.
            # new_logits is of shape (k+1,) so we add a batch dimension.
            loss_i = F.cross_entropy(new_logits.unsqueeze(0) / temperature,
                                     torch.tensor([0], device=new_logits.device),
                                     reduction='mean')
            losses.append(loss_i)
        # Average loss over all samples.
        loss = torch.mean(torch.stack(losses))
        return loss

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


