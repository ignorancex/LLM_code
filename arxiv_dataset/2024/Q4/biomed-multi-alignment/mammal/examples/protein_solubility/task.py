from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.protein_solubility.pl_data_module import (
    ProteinSolubilityDataModule,
)
from mammal.keys import (
    CLS_PRED,
    DECODER_INPUTS_ATTENTION_MASK,
    DECODER_INPUTS_STR,
    DECODER_INPUTS_TOKENS,
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    LABELS_ATTENTION_MASK,
    LABELS_STR,
    LABELS_TOKENS,
    SCORES,
)
from mammal.metrics import classification_metrics
from mammal.task import (
    MammalTask,
    MetricBase,
)


class ProteinSolubilityTask(MammalTask):
    def __init__(
        self,
        *,
        name: str,
        tokenizer_op: ModularTokenizerOp,
        data_module_kwargs: dict,
        seed: int,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            name=name,
            logger=logger,
            tokenizer_op=tokenizer_op,
        )
        self._data_module_kwargs = data_module_kwargs
        self._seed = seed

        self.preds_key = CLS_PRED
        self.scores_key = SCORES
        self.labels_key = LABELS_TOKENS

    def data_module(self) -> pl.LightningDataModule:
        return ProteinSolubilityDataModule(
            tokenizer_op=self._tokenizer_op,
            seed=self._seed,
            data_preprocessing=self.data_preprocessing,
            **self._data_module_kwargs,
        )

    def train_metrics(self) -> dict[str, MetricBase]:
        metrics = super().train_metrics()
        metrics.update(
            classification_metrics(
                self.name(),
                class_position=1,
                tokenizer_op=self._tokenizer_op,
                class_tokens=["<0>", "<1>"],
            )
        )

        return metrics

    def validation_metrics(self) -> dict[str, MetricBase]:
        validation_metrics = super().validation_metrics()
        validation_metrics.update(
            classification_metrics(
                self.name(),
                class_position=1,
                tokenizer_op=self._tokenizer_op,
                class_tokens=["<0>", "<1>"],
            )
        )
        return validation_metrics

    @staticmethod
    def data_preprocessing(
        sample_dict: dict,
        *,
        protein_sequence_key: str,
        tokenizer_op: ModularTokenizerOp,
        solubility_label_key: int | None = None,
        protein_max_seq_length: int = 1250,
        encoder_input_max_seq_len: int | None = 1260,
        labels_max_seq_len: int | None = 4,
        device: str | torch.device = "cpu",
    ) -> dict:
        """
        :param sample_dict: a dictionary with raw data
        :param protein_sequence_key: sample_dict key which points to protein sequence
        :param solubility_label_key: sample_dict key which points to label
        :param protein_max_seq_length: max sequence length of a protein. Will be used to truncate the protein
        :param encoder_input_max_seq_len: max sequence length of labels. Will be used to truncate/pad the encoder_input.
        :param labels_max_seq_len: max sequence length of labels. Will be used to truncate/pad the labels.
        :param tokenizer_op: tokenizer op

        """
        protein_sequence = sample_dict[protein_sequence_key]
        solubility_label = sample_dict.get(solubility_label_key, None)

        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=AA><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SOLUBILITY><SENTINEL_ID_0><@TOKENIZER-TYPE=AA@MAX-LEN={protein_max_seq_length}><SEQUENCE_NATURAL_START>{protein_sequence}<SEQUENCE_NATURAL_END><EOS>"
        )
        tokenizer_op(
            sample_dict=sample_dict,
            key_in=ENCODER_INPUTS_STR,
            key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
            key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
            max_seq_len=encoder_input_max_seq_len,
        )
        sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(
            sample_dict[ENCODER_INPUTS_TOKENS], device=device
        )
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
            sample_dict[ENCODER_INPUTS_ATTENTION_MASK], device=device
        )

        if solubility_label is not None:
            pad_id = tokenizer_op.get_token_id("<PAD>")
            ignore_token_value = -100
            sample_dict[LABELS_STR] = (
                f"<@TOKENIZER-TYPE=AA><SENTINEL_ID_0><{solubility_label}><EOS>"
            )
            tokenizer_op(
                sample_dict=sample_dict,
                key_in=LABELS_STR,
                key_out_tokens_ids=LABELS_TOKENS,
                key_out_attention_mask=LABELS_ATTENTION_MASK,
                max_seq_len=labels_max_seq_len,
            )
            sample_dict[LABELS_TOKENS] = torch.tensor(
                sample_dict[LABELS_TOKENS], device=device
            )
            sample_dict[LABELS_ATTENTION_MASK] = torch.tensor(
                sample_dict[LABELS_ATTENTION_MASK], device=device
            )
            # replace pad_id with -100 to
            pad_id_tns = torch.tensor(pad_id)
            sample_dict[LABELS_TOKENS][
                (sample_dict[LABELS_TOKENS][..., None] == pad_id_tns).any(-1).nonzero()
            ] = ignore_token_value

            sample_dict[DECODER_INPUTS_STR] = (
                f"<@TOKENIZER-TYPE=AA><DECODER_START><SENTINEL_ID_0><{solubility_label}><EOS>"
            )
            tokenizer_op(
                sample_dict=sample_dict,
                key_in=DECODER_INPUTS_STR,
                key_out_tokens_ids=DECODER_INPUTS_TOKENS,
                key_out_attention_mask=DECODER_INPUTS_ATTENTION_MASK,
                max_seq_len=labels_max_seq_len,
            )
            sample_dict[DECODER_INPUTS_TOKENS] = torch.tensor(
                sample_dict[DECODER_INPUTS_TOKENS], device=device
            )
            sample_dict[DECODER_INPUTS_ATTENTION_MASK] = torch.tensor(
                sample_dict[DECODER_INPUTS_ATTENTION_MASK], device=device
            )

        return sample_dict

    @staticmethod
    def process_model_output(
        tokenizer_op: ModularTokenizerOp,
        decoder_output: np.ndarray,
        decoder_output_scores: np.ndarray,
    ) -> dict:
        """
        Extract predicted solubility class and scores
        expecting decoder output to be <SENTINEL_ID_0><0><EOS> or <SENTINEL_ID_0><1><EOS>
        note - the normalized version will calculate the positive ('<1>') score divided by the sum of the scores for both '<0>' and '<1>'
            BE CAREFUL as both negative and positive absolute scores can be drastically low, and normalized score could be very high.
        outputs a dictionary containing:
            dict(
                predicted_token_str = #... e.g. '<1>'
                not_normalized_score = #the score for the positive token... e.g.  0.01
                normalized_score = #... (positive_token_score) / (positive_token_score+negative_token_score)
            )
            if there is any error in parsing the model output, None is returned.
        """

        negative_token_id = tokenizer_op.get_token_id("<0>")
        positive_token_id = tokenizer_op.get_token_id("<1>")
        label_id_to_int = {
            negative_token_id: 0,
            positive_token_id: 1,
        }
        classification_position = 1

        if decoder_output_scores is not None:
            not_normalized_score = decoder_output_scores[
                classification_position, positive_token_id
            ]
            normalized_score = not_normalized_score / (
                not_normalized_score
                + decoder_output_scores[classification_position, negative_token_id]
                + 1e-10
            )
        ans = dict(
            pred=label_id_to_int.get(int(decoder_output[classification_position]), -1),
            not_normalized_scores=not_normalized_score,
            normalized_scores=normalized_score,
        )

        return ans
