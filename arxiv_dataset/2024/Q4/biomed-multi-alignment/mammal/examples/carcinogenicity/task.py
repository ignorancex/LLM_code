from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

from mammal.examples.carcinogenicity.pl_data_module import CarcinogenicityDataModule
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


class CarcinogenicityTask(MammalTask):
    def __init__(
        self,
        *,
        tokenizer_op: ModularTokenizerOp,
        data_module_kwargs: dict,
        logger: Any | None = None,
    ) -> None:
        super().__init__(
            name="carcinogenicity",
            logger=logger,
            tokenizer_op=tokenizer_op,
        )
        self._data_module_kwargs = data_module_kwargs

        self.preds_key = CLS_PRED
        self.scores_key = SCORES
        self.labels_key = LABELS_TOKENS

    def data_module(self) -> pl.LightningDataModule:
        return CarcinogenicityDataModule(
            tokenizer_op=self._tokenizer_op,
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
        sequence_key: str,
        label_key: int | None = None,
        drug_max_seq_length: int = 1250,
        encoder_input_max_seq_len: int | None = 1260,
        labels_max_seq_len: int | None = 4,
        tokenizer_op: ModularTokenizerOp,
        device: str | torch.device = "cpu",
    ) -> dict:
        drug_sequence = sample_dict[sequence_key]
        label = sample_dict.get(label_key, None)

        sample_dict[ENCODER_INPUTS_STR] = (
            f"<@TOKENIZER-TYPE=SMILES><CARCINOGENICITY><SENTINEL_ID_0><MOLECULAR_ENTITY><MOLECULAR_ENTITY_SMALL_MOLECULE><@TOKENIZER-TYPE=SMILES@MAX-LEN={drug_max_seq_length}><SEQUENCE_NATURAL_START>{drug_sequence}<SEQUENCE_NATURAL_END><EOS>"
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

        if label is not None:
            pad_id = tokenizer_op.get_token_id("<PAD>")
            ignore_token_value = -100
            sample_dict[LABELS_STR] = (
                f"<@TOKENIZER-TYPE=SMILES><SENTINEL_ID_0><{label}><EOS>"
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
            sample_dict[LABELS_TOKENS][
                (sample_dict[LABELS_TOKENS][..., None] == torch.tensor(pad_id))
                .any(-1)
                .nonzero()
            ] = ignore_token_value

            sample_dict[DECODER_INPUTS_STR] = (
                f"<@TOKENIZER-TYPE=SMILES><DECODER_START><SENTINEL_ID_0><{label}><EOS>"
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
