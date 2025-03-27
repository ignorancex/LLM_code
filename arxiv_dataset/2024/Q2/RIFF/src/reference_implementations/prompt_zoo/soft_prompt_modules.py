"""A module that defines the modifications to the roberta model and defines a
new embedding for soft-prompt tuning."""
import random

import torch
from absl import flags
from transformers import RobertaForMaskedLM, T5ForConditionalGeneration

FLAGS = flags.FLAGS


class PromptEmbedding(torch.nn.Module):
    """We implement a new Embedding module for the prompt parameters.

    We only update the prompt vectors during training. This
    PromptEmbedding will have a reference to the normal embedding matrix
    of the roberta model which will be populated when we load the
    encoder from the huggingface. prompt tokens are always at the first
    prompt_length steps of the input after the BOS token (first token).
    """

    def __init__(
        self,
        prompt_length: int,
        embedding_dim: int,
        normal_embedder: torch.nn.Embedding,
        normal_vocab_size: int,
        lm_type: str,
    ) -> None:
        """
        Args:
            prompt_length (int): length of the prompt tokens which are prepended to the input.
            embedding_dim (int): the size of each embedding vector
            normal_embedder (torch.nn.Embedding): this is the embedding table for the normal tokens
            of the input/output sequence used by roberta model.
        """
        super().__init__()
        self.prompt_length = prompt_length

        self.lm_type = lm_type

        self.normal_embedder = normal_embedder

        self.prompt_embedder = torch.nn.Embedding(prompt_length, embedding_dim)

        # sample prompt_length vectors from the normal embedding table to initialize the prompt vectors.
        sampled_indices = random.choices(list(range(normal_vocab_size)), k=prompt_length)
        self.prompt_embedder.weight.data = self.normal_embedder.weight.data[sampled_indices, :]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """T5 model doesn't have BOS, so the fist prompt_length steps
        are going to be part of the prompt embeddings.

        Prompt tokens are always at the first prompt_length steps of the
        input after the BOS token. split the input sequences into three parts:

            1 - the first BOS token to be embedded by the normal embedding.
            2 - the next prompt_length tokens should be mapped to prompt vectors.
            3 - the rest should be embedded by the normal embedding table of roberta defined for english tokens.
        concatinate the embedded splits into a single split along the sequence dimension.
        """
        batch_size, sequence_length = input.size()

        if self.lm_type == "roberta":
            bos_input, prompt_input, normal_input = torch.split(
                input, [1, self.prompt_length, sequence_length - self.prompt_length - 1], dim=1
            )

            # prompt_embedded has shape: (batch_size,  self.prompt_length, embedding_dim)
            prompt_embedded = self.prompt_embedder(prompt_input)

            # normal_input_embedded has shape: (batch_size,  sequence_length - self.prompt_length, embedding_dim)
            normal_input_embedded = self.normal_embedder(normal_input)

            bos_input_embedded = self.normal_embedder(bos_input.view(batch_size, 1))

            # concat along the dimension 1
            return torch.cat((bos_input_embedded, prompt_embedded, normal_input_embedded), dim=1)

        elif self.lm_type == "t5":
            prompt_input, normal_input = torch.split(
                input, [self.prompt_length, sequence_length - self.prompt_length], dim=1
            )

            # prompt_embedded has shape: (batch_size,  self.prompt_length, embedding_dim)
            prompt_embedded = self.prompt_embedder(prompt_input)

            # normal_input_embedded has shape: (batch_size,  sequence_length - self.prompt_length, embedding_dim)
            normal_input_embedded = self.normal_embedder(normal_input)
            # concat along the dimension 1
            return torch.cat((prompt_embedded, normal_input_embedded), dim=1)


def create_softprompt_roberta(lm_type: str) -> torch.nn.Module:
    """This function implements the modifications to the roberta module of the
    huggingface to include the soft prompt vectors in the input."""
    # prompt length
    p_len = FLAGS.prompt_length

    if lm_type == "roberta":
        # let the RobertaForMaskedLM load the initial checkpoint of the roberta
        # with the normal embedding table.
        roberta_model = RobertaForMaskedLM.from_pretrained(FLAGS.roberta_pretrained_model)

        d_model = roberta_model.config.hidden_size
        vocab_size = roberta_model.config.vocab_size

        prompt_embedding = PromptEmbedding(
            p_len, d_model, roberta_model.roberta.get_input_embeddings(), vocab_size, lm_type=lm_type
        )

        # update the general embedding module of huggingface roberta.
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
        roberta_model.roberta.set_input_embeddings(prompt_embedding)
        return roberta_model

    elif lm_type == "t5":
        # let the T5ForConditionalGeneration load the initial checkpoint of the T5
        # with the normal embedding table.
        t5_model = T5ForConditionalGeneration.from_pretrained(FLAGS.t5_pretrained_model)

        # t5_model.config.d_model is from the class T5ForConditionalGeneration
        # which is the embedding dimension of the embedding table of the T5.
        d_model = t5_model.config.d_model
        vocab_size = t5_model.config.vocab_size

        prompt_embedding = PromptEmbedding(p_len, d_model, t5_model.shared, vocab_size, lm_type=lm_type)

        # update the general shared embedding module of huggingface T5.
        # now every call by t5_model.shared(input_ids) will use our forward method of the PromptEmbedding
        # we don't want to update the decoder embedding to add the prompt tokens for the output tokens.
        # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1344
        t5_model.shared = prompt_embedding
        t5_model.encoder.embed_tokens = prompt_embedding
        return t5_model
