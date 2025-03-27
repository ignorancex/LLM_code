"""This is a module to train a classifier on top of the LM."""
from typing import Dict, Iterator

import numpy
import torch
from absl import flags

from src.reference_implementations.prompt_zoo.data_utility import augment_batch
from src.reference_implementations.prompt_zoo.prompted_lm import LMPrompted

FLAGS = flags.FLAGS


class ClassifierLM(LMPrompted):
    """Wrapper class around the LM Model with a classifier on top of the LM."""

    def __init__(
        self, seed: int, enable_data_augmentation: int, enable_paraphrase_training: int, load_paraphraser: int
    ) -> None:
        super().__init__(seed, enable_data_augmentation, enable_paraphrase_training, load_paraphraser)

    def no_ensemble_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop using a separate classifier."""

        self.predict_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask"])

        encoder = self.model_pool[f"{self.lm}_model"]
        classifier_model = self.model_pool["classifier_model"]

        output = encoder(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )

        encoder_hidden_states = output.last_hidden_state

        _, logits = classifier_model(encoder_hidden_states, loaded_batch["attention_mask"])

        prediction_logits = logits.cpu().detach().numpy()

        inputs_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        for index, input_str in enumerate(inputs_str):
            for class_idx in range(FLAGS.num_classes):
                output_row = {
                    "potential_class": self.tokenizer.id_to_class[str(class_idx)],
                    "original_prediction_score": prediction_logits[index][class_idx],
                    "original_inputs": input_str,
                    "gold_class": batch["gold_outputs"][index],
                }
                yield output_row

    def paraphrase_and_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """Generate multiple paraphrases along the original input and return
        all the results.

        For each example, the first score belongs to the original input.
        """
        self.predict_mode_on()
        # for classifier_fine-tuning, the dummy labels doesn't have any effect.
        inputs_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        paraphrases = self.draw_samples_for_augmentation(batch, for_train=False)
        augment_batch(
            batch,
            paraphrases,
            self.tokenizer,
            num_return_seq=FLAGS.test_sample_size,
        )

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask"])

        encoder = self.model_pool[f"{self.lm}_model"]
        classifier_model = self.model_pool["classifier_model"]

        output = encoder(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )

        encoder_hidden_states = output.last_hidden_state

        _, logits = classifier_model(encoder_hidden_states, loaded_batch["attention_mask"])

        prediction_logits = logits.cpu().detach().numpy()

        for index, input_str in enumerate(inputs_str):
            scores = prediction_logits[
                index * (FLAGS.test_sample_size + 1) : (index + 1) * (FLAGS.test_sample_size + 1), :
            ]
            for class_idx in range(FLAGS.num_classes):
                avg_score = numpy.mean(scores[1:, class_idx])
                output_row = {
                    "potential_class": self.tokenizer.id_to_class[str(class_idx)],
                    "prediction_score": avg_score,
                    "all_prediction_score": 0.5 * avg_score + 0.5 * scores[0, class_idx],
                    "original_prediction_score": scores[0, class_idx],
                    "gold_class": batch["gold_outputs"][index],
                    "paraphrases": paraphrases[index * FLAGS.test_sample_size : (index + 1) * FLAGS.test_sample_size],
                    "original_inputs": input_str,
                }
                yield output_row

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The classifier training step."""
        self.train_mode_on()
        batch["class_indices"] = torch.tensor(
            [int(self.tokenizer.class_to_id[label]) for label in batch["gold_outputs"]]
        )
        if self.enable_data_augmentation == 1:
            paraphrases = self.draw_samples_for_augmentation(batch)
            augment_batch(batch, paraphrases, self.tokenizer, num_return_seq=FLAGS.test_sample_size)

            batch_size = batch["class_indices"].size()[0]
            batch["class_indices"] = (
                batch.pop("class_indices")
                .reshape(batch_size, 1)
                .expand(batch_size, 1 + FLAGS.test_sample_size)
                .reshape(
                    -1,
                )
            )

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask", "class_indices"])

        encoder = self.model_pool[f"{self.lm}_model"]
        classifier_model = self.model_pool["classifier_model"]

        output = encoder(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )
        encoder_hidden_states = output.last_hidden_state
        loss = classifier_model.compute_loss(
            encoder_hidden_states,
            loaded_batch["attention_mask"],
            loaded_batch["class_indices"],
            enable_data_augmentation=self.enable_data_augmentation == 1,
        )

        loss.backward()

        # updates the main language model.
        self.optimizer.step()
        loss_value = loss.item()

        return {"loss_value": loss_value}
