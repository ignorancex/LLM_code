
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from utils import time_performance_decorator
from detect_methods.perplexity import calcu_perplexity

EPSILON = 1e-4


@torch.no_grad()
def vocab_entropies_from_src(
    src_hidden_state: Tensor,  # batch_size, src_seq_len, hidden_dim
    embd_to_id_layer: torch.nn.Linear,
    # mask: Tensor | None = None
) -> Tensor:
    """
    src_hidden_state:  batch_size, seq_len, hidden_dim

    hidden_src_hidden_state of one layer of src steps.

    return vocab_entropies: batch_size, vocab_size
    """

    projection = embd_to_id_layer.forward(src_hidden_state)

    # vocab_activation_scores: batch_size, seq_len, vocab_size
    vocab_activation_scores = F.softmax(projection, dim=-1)

    # vocab_probs: batch_size, vocab_size, seq_len
    vocab_probs = vocab_activation_scores / torch.sum(
        vocab_activation_scores, dim=1, keepdim=True
    )

    vocab_entropies: Tensor = torch.distributions.Categorical(
        probs=vocab_probs.permute(0, 2, 1),
        validate_args=False,
    ).entropy()

    # TODO[jayden]: apply mask
    # if mask is not None:
    #     index_no_top = torch.argwhere(mask)

    return vocab_entropies


@torch.no_grad()
def entropy_fixed_probs(
    # src_hidden_state: Tensor,  # batch_size, src_seq_len, hidden_dim
    entropies: Tensor,  # batch_size, vocab_size
    tgt_tokens: Tensor,  # int batch_size, tgt_seq_len
    tgt_logits: Tensor,  # batch_size, tgt_seq_len, vocab_size
    alpha=1.,
) -> Tensor:
    """entropies: Tensor, batch_size, vocab_size

    tgt_tokens: Tensor int, batch_size, tgt_seq_len

    tgt_logits: Tensor, batch_size, tgt_seq_len, vocab_size

    return probs: batch_size, tgt_seq_len
    """
    tgt_logits = tgt_logits.to(device=entropies.device)
    tgt_tokens = tgt_tokens.to(device=entropies.device)
    if alpha <= 1:
        out_logits = tgt_logits + alpha * (-entropies.unsqueeze(1))
    else:
        out_logits = -entropies.unsqueeze(1)

    probs = F.softmax(out_logits, dim=-1)
    return probs.gather(
        dim=-1, index=tgt_tokens.unsqueeze(-1)
    ).squeeze(-1)


@dataclass
class EntropyShapnessOutput:

    entropies: list[float]
    probs: list[float]

    @property
    @torch.no_grad()
    def perplexity(self):
        ppl = calcu_perplexity(
            torch.tensor(self.probs, dtype=torch.float64).unsqueeze(0)
        )
        if ppl != ppl or ppl == float('inf'):  # nan or inf
            print(f"WARNING! perplexity: {ppl}")
            return None
        return ppl

    @property
    def entropy_avg(self):
        if len(self.entropies) == 0:
            return .0
        return sum(self.entropies) / len(self.entropies)


@torch.no_grad()
def calcu_sharpnesses_for_one_layer(
    src_hidden_state: Tensor,
    tgt_tokens: Tensor,  # int batch_size, tgt_seq_len
    tgt_logits: Tensor,
    embd_to_id_layer: torch.nn.Linear,
    alpha=1.,
):

    tgt_logits = tgt_logits.to(device=src_hidden_state.device)
    tgt_tokens = tgt_tokens.to(device=src_hidden_state.device)

    entropies_vocab: Tensor = vocab_entropies_from_src(
        src_hidden_state, embd_to_id_layer,
    )
    probs: Tensor = entropy_fixed_probs(
        entropies_vocab, tgt_tokens, tgt_logits, alpha,
    )

    entropies_vocab = entropies_vocab.to(device=src_hidden_state.device)
    probs = probs.to(device=src_hidden_state.device)

    entropies = (
        entropies_vocab.unsqueeze(1)
        .repeat(1, tgt_tokens.shape[1], 1)
        .gather(dim=-1, index=tgt_tokens.unsqueeze(-1))
        .squeeze(-1)
    )

    return EntropyShapnessOutput(
        entropies=entropies[0].to("cpu").numpy().tolist(),
        probs=probs[0].to("cpu").numpy().tolist(),
    )

@time_performance_decorator(enable=False)
@torch.no_grad()
def calcu_sharpnesses_for_layers(
    src_hidden_states: tuple[Tensor],
    tgt_tokens: Tensor,
    tgt_logits: Tensor,
    embd_to_id_layer: torch.nn.Linear,
    alpha=1.,
):
    return [
        calcu_sharpnesses_for_one_layer(
            src_hidden_state,
            tgt_tokens,
            tgt_logits,
            embd_to_id_layer,
            alpha,
        ) for src_hidden_state in src_hidden_states
    ]
