"""This is the main module to launch the training of the different experiments
with the backbone language model.

The module also has some utility functions useful during model
training/inference.
"""

import csv
import io
import os
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter

from src.reference_implementations.prompt_zoo.classifier import ClassifierLM
from src.reference_implementations.prompt_zoo.data_utility import create_sentiment_dataset
from src.reference_implementations.prompt_zoo.gradient_search import SearchLM
from src.reference_implementations.prompt_zoo.metrics import sentiment_metric
from src.reference_implementations.prompt_zoo.model_utility import set_random_seed
from src.reference_implementations.prompt_zoo.prompted_lm import LMPrompted, MyBaseLM

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_epochs", 20, "The maximum number of epochs for training.")
flags.DEFINE_integer("training_steps", 100, "The number of training steps for each epoch.")
flags.DEFINE_integer("steps_per_checkpoint", 50, "keep checkpoint of the model every this number of steps")
flags.DEFINE_string("prediction_file", "/tmp/predictions.csv", "the path/name for saving the predictions.")
flags.DEFINE_string("paraphrase_save_path", "/tmp/paraphrases.json", "path to save the generated paraphrases.")

flags.DEFINE_string("dev_file", "/tmp/dev.csv", "the path/name of the dev file.")
flags.DEFINE_string("test_file", "/tmp/test.csv", "the path/name of the test file.")
flags.DEFINE_string("task_name", "semeval_3_class_sentiment", "the name of the downstream nlp task.")
flags.DEFINE_string("train_file", "/tmp/train.csv", "the path/name of the train file.")
flags.DEFINE_string(
    "metric_to_save", "accuracy", "Which metric to use to save the best model on the internal dev data."
)
flags.DEFINE_integer("enable_data_augmentation", 0, "To train with data augmentation generated with paraphrasing.")
flags.DEFINE_integer("enable_paraphrase_training", 0, "To train the paraphraser with some objective.")
flags.DEFINE_integer(
    "load_paraphraser", 0, "Whether to load the paraphraser from a specific path while doing data augmentation"
)


def start_predicting(model: MyBaseLM, dataloader: torch.utils.data.DataLoader, prediction_file: str) -> None:
    """Read batches from the dataloader and predict the outputs from the model
    for the correct experiment and save the results in the prediction_file as
    csv format row by row."""
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
        header_written = False
        for batch in dataloader:
            for ret_row in model.predict(batch):
                if not header_written:
                    headers = ret_row.keys()
                    writer.writerow(headers)
                    header_written = True
                writer.writerow(list(ret_row.values()))
    return


def start_training(model: MyBaseLM, dataloader: torch.utils.data.DataLoader) -> Iterator[Tuple[int, float]]:
    """Pick a batch from the dataloader, and train the model for one step."""
    step = 0
    for batch in dataloader:
        loss_values = model.train(batch)
        step += 1
        yield step, loss_values["loss_value"]


def train_model(
    model: MyBaseLM,
    metric: Callable[[str, int], Dict[str, float]],
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
) -> None:
    """Run the model on input data; for training or inference."""
    num_labels = len(list(model.tokenizer.class_to_id.keys()))
    if FLAGS.mode == "train":
        start_time = time.time()
        writer = SummaryWriter(FLAGS.model_path)
        epoch = 0
        global_step = 0
        total_loss = []
        eval_file = os.path.join(FLAGS.model_path, "temp_eval.csv")
        start_predicting(model, eval_dataloader, eval_file)
        scores = metric(eval_file, num_labels)  # type: ignore
        for score_name, score_val in scores.items():
            writer.add_scalar(f"{score_name}/dev", score_val, 0)
            if score_name == FLAGS.metric_to_save:
                best_score = score_val
                model.save("best_step")

        while epoch < FLAGS.max_epochs and global_step < FLAGS.training_steps:
            print("\nEpoch:{0}\n".format(epoch))
            epoch_loss = []
            for step, loss in start_training(model, train_dataloader):
                global_step += 1
                total_loss.append(loss)
                epoch_loss.append(loss)
                mean_total_loss = np.mean(total_loss)
                mean_epoch_loss = np.mean(epoch_loss)
                print(
                    f"\rEpoch: {epoch} | Batch: {step} | Mean Loss: {mean_total_loss} | "
                    f"Epoch Loss: {mean_epoch_loss} | Loss: {loss}\n"
                )

                if global_step % FLAGS.steps_per_checkpoint == 0:
                    start_predicting(model, eval_dataloader, eval_file)
                    scores = metric(eval_file, num_labels)  # type: ignore
                    for score_name, score_val in scores.items():
                        writer.add_scalar(f"{score_name}/dev", score_val, global_step)
                        if score_name == FLAGS.metric_to_save:
                            if score_val > best_score:
                                best_score = score_val
                                model.save("best_step")
                            elif score_val < best_score and FLAGS.exp_type in ["gradient_search"]:
                                # re-load the best previous template searched so far!
                                # the previous templates was not good!
                                FLAGS.checkpoint = "best_step"
                                model.load_from_checkpoint()

                writer.add_scalar("Mean_Total_Loss/train", mean_total_loss, global_step)
                writer.add_scalar("Mean_Epoch_Loss/train", mean_epoch_loss, global_step)
                writer.flush()
                if global_step == FLAGS.training_steps:
                    # stop training in this epoch.
                    break

            # do final evaluation on the dev data at the end of epoch.
            start_predicting(model, eval_dataloader, eval_file)
            scores = metric(eval_file, num_labels)  # type: ignore
            for score_name, score_val in scores.items():
                writer.add_scalar(f"{score_name}/dev", score_val, global_step)
                if score_name == FLAGS.metric_to_save:
                    if score_val > best_score:
                        best_score = score_val
                        model.save("best_step")
                    elif score_val < best_score and FLAGS.exp_type in ["gradient_search"]:
                        # re-load the best previous template searched so far!
                        # the previous templates was not good!
                        FLAGS.checkpoint = "best_step"
                        model.load_from_checkpoint()
            epoch += 1

        writer.close()

        # delete the eval_file
        os.remove(eval_file)
        end_time = time.time()
        print(f"Training finished in {end_time - start_time} seconds!")

    else:
        raise Exception(f"the mode {FLAGS.mode} is not for training.")


def test_model(
    model: MyBaseLM,
    test_dataloader: torch.utils.data.DataLoader,
    metric: Optional[Callable[[str, int], Dict[str, float]]] = None,
) -> None:
    writer = SummaryWriter(FLAGS.model_path)
    num_labels = len(list(model.tokenizer.class_to_id.keys()))
    if FLAGS.mode in ["test", "inference", "eval", "no_finetune_test"]:
        print("Predicting...")
        start_predicting(model, test_dataloader, FLAGS.prediction_file)
        if metric is not None:
            scores = metric(FLAGS.prediction_file, num_labels)  # type: ignore
            for score_name, score_val in scores.items():
                writer.add_scalar(f"{score_name}/test", score_val, 0)
    else:
        raise Exception(f"the mode {FLAGS.mode} is not for testing.")


def launch_test_or_train() -> None:
    """Launch the testing or training phase for the prompting experiments."""

    if FLAGS.exp_type == "gradient_search":
        model = SearchLM(
            FLAGS.seed,
            FLAGS.task_name,
            FLAGS.enable_data_augmentation,
            FLAGS.enable_paraphrase_training,
            FLAGS.load_paraphraser,
        )

    elif FLAGS.exp_type == "classifier_finetune":
        model = ClassifierLM(
            FLAGS.seed, FLAGS.enable_data_augmentation, FLAGS.enable_paraphrase_training, FLAGS.load_paraphraser
        )

    elif FLAGS.exp_type == "no_finetune":
        FLAGS.mode = "no_finetune_test"
        model = LMPrompted(
            FLAGS.seed, FLAGS.enable_data_augmentation, FLAGS.enable_paraphrase_training, FLAGS.load_paraphraser
        )
    else:
        model = LMPrompted(
            FLAGS.seed, FLAGS.enable_data_augmentation, FLAGS.enable_paraphrase_training, FLAGS.load_paraphraser
        )

    para_tokenizer = None
    if FLAGS.enable_data_augmentation == 1 or FLAGS.enable_paraphrase_training == 1:
        para_tokenizer = model.para_tokenizer

    if FLAGS.mode == "train":
        if FLAGS.classification_type == "fewshot":
            train_dataloader = create_sentiment_dataset(
                tokenizer=model.tokenizer,
                train_file_name=FLAGS.train_file,
                task_name=FLAGS.task_name,
                para_tokenizer=para_tokenizer,
            )
            eval_dataloader = create_sentiment_dataset(
                tokenizer=model.tokenizer,
                dev_file_name=FLAGS.dev_file,
                task_name=FLAGS.task_name,
                para_tokenizer=para_tokenizer,
            )
        else:
            raise ValueError("Is this fewshot learning?")

        train_model(
            model=model,
            metric=sentiment_metric,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
        )

    elif FLAGS.mode in ["test", "inference", "eval", "no_finetune_test"]:
        test_dataloader = create_sentiment_dataset(
            tokenizer=model.tokenizer,
            test_file_name=FLAGS.test_file,
            task_name=FLAGS.task_name,
            para_tokenizer=para_tokenizer,
        )
        test_model(model=model, metric=sentiment_metric, test_dataloader=test_dataloader)


def main(argv: Any) -> None:
    """Main function to switch over the t5 experiment type and launch the
    correct train script."""

    # set random seed. internal module will set it differently.
    set_random_seed(FLAGS.seed)

    if FLAGS.exp_type in [
        "soft_prompt_finetune",
        "all_finetune",
        "input_finetune",
        "output_finetune",
        "gradient_search",
        "classifier_finetune",
        "no_finetune",
        "lora_finetune",
    ]:
        launch_test_or_train()


if __name__ == "__main__":
    app.run(main)
