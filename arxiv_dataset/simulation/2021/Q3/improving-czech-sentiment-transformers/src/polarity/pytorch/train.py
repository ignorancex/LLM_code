import logging
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import wandb

from config import LOGGING_FORMAT, LOGGING_DATE_FORMAT
from src.utils import format_time

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)

FAST_DEBUG = True

class SentimentClassifier(nn.Module):
    """"
    Custom Model build on top of hugging face model
    """

    def __init__(self, n_classes, hugging_face_model, dropout_p=0.1):
        super(SentimentClassifier, self).__init__()
        # bert, albert whatever...
        self.hugging_face_model = hugging_face_model
        self.drop = nn.Dropout(p=dropout_p)
        # self.out = nn.Linear(self.hugging_face_model.config.hidden_size, n_classes)
        self.out = nn.Linear(32768, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.hugging_face_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # print("")
        # todo

        # pooled_output = self.hugging_face_model(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask
        # )
        #
        # size = len(pooled_output[0])
        # # print("Size:" + str(size))
        #
        # pooled_output = torch.stack(pooled_output, dim=1)
        #
        # size = len(pooled_output)
        # # print("Size:" + str(size))
        # # print("Shape:" + str(pooled_output.shape))
        # pooled_output = torch.squeeze(pooled_output)
        # # print("Shape:" + str(pooled_output.shape))
        # pooled_output = torch.flatten(pooled_output, start_dim=1)
        # # print("Shape:" + str(pooled_output.shape))
        output = self.drop(pooled_output)

        return self.out(output)


# Training loop
def run_training(num_epochs, model, train_data_loader, val_data_loader, loss_fn, optimizer, scheduler,
                 use_custom_model, device, validation_size, print_stat_freq, enable_wandb, data_parallel,
                 args, torch_tuner=None):
    """

    :param num_epochs:
    :param model:
    :param train_data_loader:
    :param val_data_loader:
    :param loss_fn:
    :param optimizer:
    :param scheduler:
    :param use_custom_model:
    :param device:
    :param validation_size:
    :param print_stat_freq:
    :param enable_wandb:
    :param data_parallel:
    :param full_mode_eval: instance of TorchTuner
    :return:
    """
    history = defaultdict(list)
    best_accuracy = 0

    t00 = time.time()
    for epoch in range(num_epochs):
        logger.info('')
        logger.info('======== Epoch {:} / {:} ========'.format(epoch + 1, num_epochs))
        t0 = time.time()
        logger.info('-' * 30)

        # Report progress.
        # logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(0, len(train_data_loader), elapsed))

        # logger.info(f'Epoch {epoch + 1}/{num_epochs}')

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            epoch+1,
            use_custom_model,
            print_stat_freq,
            enable_wandb,
            data_parallel
        )

        train_time = format_time(time.time() - t0)
        logger.info(f'Total train time for epoch:{train_time}')
        logger.info(f'Train loss: {train_loss} accuracy: {train_acc}')

        t0 = time.time()

        if validation_size == 0:
            val_acc = val_loss = 0
        else:
            val_acc, val_loss = eval_model(
                model,
                val_data_loader,
                loss_fn,
                device,
                validation_size,
                use_custom_model,
                data_parallel
            )

        val_time = format_time(time.time() - t0)
        logger.info(f'Val   loss {val_loss} accuracy {val_acc}')
        logger.info(f'Total validation time for epoch:{val_time}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['train_time'].append(train_time)
        history['val_time'].append(val_time)

        if enable_wandb is True:
            try:
                wandb.log({'train_acc': train_acc, 'train_loss': train_loss, 'val_acc': val_acc, 'val_loss': val_loss,
                           'train_time': train_time, 'val_time': val_time})
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))


        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

        total_time = time.time() - t00
        if torch_tuner != None:
            torch_tuner.perform_train_eval(model, args, total_time, None, epoch+1)


    return history


# Train utils
def train_epoch(model, data_loader, loss_fn, optimizer,
                device, scheduler, epoch, use_custom_model=False,
                print_batch_freq=10, enable_wandb=False, data_parallel=False):
    """
    Run one training epoch
    :param print_batch_freq: stats print frequency
    :param model: torch model
    :param data_loader: data loader
    :param loss_fn: loss function, if for_sequence_model is True it is ignored
    :param optimizer: optimizer algorithm
    :param device: device to run
    :param scheduler: learning rate scheduler
    :param use_custom_model: whether to use custom layer or BertForSequence Classification
    :return: tuple of (accuracy, loss)
    """
    model = model.train()

    losses = []
    correct_predictions = 0
    correct_pred_tmp = 0

    running_loss = 0.0
    n_examples = 0
    total_processed_examples = 0
    # time since epoch started
    t0 = time.time()

    data_loader_len = len(data_loader)

    # the true number wil be little bit lower, bcs we do not align the data
    total_examples = data_loader_len * data_loader.batch_size

    for i, data in enumerate(data_loader, 0):
        if FAST_DEBUG is True:
            # only for testing purposes
            if i == 5:
                break

        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        labels = data["labels"].to(device)

        # get size of batch
        data_size = list(data['labels'].shape)[0]
        # print(f"Data size:{data_size}")
        n_examples += data_size
        total_processed_examples += data_size


        if use_custom_model is False:
            # XXXForSequenceClassifier
            loss, outputs = model(input_ids,
                                  token_type_ids=None,
                                  attention_mask=attention_mask,
                                  labels=labels)
        else:
            # Custom model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)

        _, preds = torch.max(outputs, dim=1)

        tmp = torch.sum(preds == labels)
        correct_pred_tmp += tmp
        correct_predictions += tmp
        if data_parallel is True:
            # https://discuss.pytorch.org/t/loss-function-in-multi-gpus-training-pytorch/76765
            # https://discuss.pytorch.org/t/how-to-fix-gathering-dim-0-warning-in-multi-gpu-dataparallel-setting/41733
            # print("loss:" + str(loss))
            # loss_mean = loss.mean()
            # print("Loss mean:" + str(loss_mean))
            loss = loss.mean()

        loss_item = loss.item()
        # print("Loss item:" + str(loss_item))
        # print("===")

        losses.append(loss_item)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # print statistics
        running_loss += loss_item
        # print every n mini-batches
        if i % print_batch_freq == 0 and not i == 0:
            # logger.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i, total_processed_examples, elapsed))
            try:
                last_lr = scheduler.get_last_lr()
                last_lr = last_lr[0]
            except Exception as e:
                last_lr = 0
                logger.error("Cannot parse acutal learning rate")

            elapsed = format_time(time.time() - t0)
            avg_loss = running_loss / print_batch_freq
            acc = correct_pred_tmp.double() / n_examples
            logger.info('Batch: %5d/%-5d avg loss: %.4f  acc: %.3f processed: %d/%d examples,  epoch-time: %s, actual lr:%.12f' %
                        (i, len(data_loader), avg_loss, acc,
                         total_processed_examples, total_examples, elapsed, last_lr))

            if enable_wandb is True:
                try:
                    wandb.log({'epoch' : epoch, 'batch' : i, 'avg_loss' : avg_loss, 'avg_accuracy' : acc, 'current_lr' : last_lr})
                except Exception as e:
                    logger.error("Error WANDB with exception e:" + str(e))


            running_loss = 0.0
            n_examples = 0.0
            correct_pred_tmp = 0.0


    logger.info(f"Number of examples:{total_processed_examples}")
    logger.info(f"Correct predictions:{correct_predictions}")
    return correct_predictions.double() / total_processed_examples, np.mean(losses)


# Eval model
def eval_model(model, data_loader, loss_fn, device, n_examples, use_custom_model=False,
               data_parallel=False):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for i, d in enumerate(data_loader):
            if FAST_DEBUG is True:
                # only for testing purposes
                if i == 5:
                    break

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            # https://huggingface.co/transformers/migration.html
            # we are using BertForSequenceClassification
            if use_custom_model is False:
                loss, outputs = model(input_ids,
                                      token_type_ids=None,
                                      attention_mask=attention_mask,
                                      labels=labels)
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = loss_fn(outputs, labels)

            _, preds = torch.max(outputs, dim=1)

            correct_predictions += torch.sum(preds == labels)
            if data_parallel is True:
                # loss_mean = loss.mean()
                # print("Loss mean:" + str(loss_mean))
                loss = loss.mean()

            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader, use_custom_model, device, batch_size=16, print_progress=False):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    if batch_size is None:
        logger.info("Batch size not specified for priting info setting it to 16")
        batch_size = 16

    t0 = time.time()

    epoch_time = t0
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            if FAST_DEBUG is True:
                # only for testing purposes
                if i == 5:
                    break

            if print_progress is True:
                if i % 1 == 0:
                    cur_time = time.time() - t0
                    epoch_time = time.time() - epoch_time
                    print("total time:" + str(cur_time) + "s 10 epochs:" + str(epoch_time) + " s  Predicted:" + str(i*batch_size) + " examples current batch:" + str(i))
                    epoch_time = time.time()

            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            # we are using BertForSequenceClassification
            if use_custom_model is False:
                loss, outputs = model(input_ids,
                                      token_type_ids=None,
                                      attention_mask=attention_mask,
                                      labels=labels)
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values
