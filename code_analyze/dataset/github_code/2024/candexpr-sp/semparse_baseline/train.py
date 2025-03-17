import os
import argparse
import json
import time
import warnings
# from tqdm import tqdm
# from datetime import date

import torch
import torch.optim as optim
# import torch.nn as nn
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
# import torch.optim as optim
from kopl.kopl import KoPLEngine

from dhnamlib.pylib.filesys import make_logger

from .data import DataLoader
from .predict import validate

from kqapro_util.misc import seed_everything, ProgressBar
# from kqapro_util.misc import MetricLogger
from kqapro_util.lr_scheduler import get_linear_schedule_with_warmup
from .util import register

# warnings.simplefilter("ignore") # hide warnings that caused by invalid sparql query

new_tokens = ['<func>', '<arg>']

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger = register.retrieve('logger')

    logger.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    if args.using_shuffled_train:
        train_pt_file_name = 'train-shuffled.pt'
    else:
        train_pt_file_name = 'train.pt'
    train_pt = os.path.join(args.input_dir, train_pt_file_name)
    val_pt = os.path.join(args.input_dir, 'val.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True, percent=args.train_set_percent)
    if args.train_set_percent != 100:
        from dhnamlib.pylib.torchlib.data_processing import EpochRepeatingDataLoader
        num_epoch_repeats = 100 / args.train_set_percent
        old_train_loader = train_loader
        train_loader = EpochRepeatingDataLoader(train_loader, num_epoch_repeats=num_epoch_repeats)
        train_loader.dataset = old_train_loader.dataset
        del old_train_loader
    val_loader = DataLoader(vocab_json, val_pt, args.pred_batch_size)

    engine = KoPLEngine(json.load(open(os.path.join(args.input_dir, 'kb.json'))))
    logger.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    added_tokens_num = tokenizer.add_tokens(new_tokens, special_tokens=True)
    print('added_tokens_num:', added_tokens_num)
    if added_tokens_num > 0:
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    logger.info(model)
    t_total = ((len(train_loader) + args.gradient_accumulation_steps - 1) //
               args.gradient_accumulation_steps) * args.num_train_epochs  # Prepare optimizer and schedule (linear warmup and decay)
    # no_decay = ["bias", "LayerNorm.weight"]
    no_decay = ["bias", "layernorm", "layer_norm"]
    bart_param_optimizer = list(model.named_parameters())

    # 'weight_decay' is for AdamW
    # 'lr' values become LambdaLR's base_lrs which are multiplied by lr_lambdas
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_loader.dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_loader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_loader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    logger.info('Checking...')
    logger.info("===================Dev==================")
    validate(model, val_loader, device, tokenizer, engine, args.postprocessing_answer)
    # tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    best_accuracy = float('-inf')
    accuracy_history = []
    for epoch_num in range(1, int(args.num_train_epochs) + 1):
        if args.using_progress_bar:
            pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        for step, batch in enumerate(train_loader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(device) for t in batch)
            pad_token_id = tokenizer.pad_token_id
            source_ids, source_mask, y = batch[0], batch[1], batch[-2]
            y_ids = y[:, :-1].contiguous()  # except the last tokens
            # lm_labels = y[:, 1:].clone()
            lm_labels = y[:, 1:].contiguous()  # except the first tokens
            # lm_labels[y[:, 1:] == pad_token_id] = -100  # it causes error when 2 ** 31 - 1 is used instead of -100

            inputs = {
                "input_ids": source_ids.to(device),
                "attention_mask": source_mask.to(device),
                "decoder_input_ids": y_ids.to(device),
                "labels": lm_labels.to(device),
            }
            # when decoder_attention_mask is not given, the default masking behavior is used to ignore the effect of pad tokens.
            outputs = model(**inputs)
            loss = outputs['loss']  # outputs[0]
            loss.backward()
            if args.using_progress_bar:
                pbar(step, {'loss': loss.item()})
            # tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
        accuracy = validate(model, val_loader, device, tokenizer, engine, args.postprocessing_answer)
        accuracy_history.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy

            # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cmd_args_file_path = os.path.join(output_dir, 'cmd_args.json')
            if not os.path.isfile(cmd_args_file_path):
                with open(cmd_args_file_path, 'w') as f:
                    json.dump(vars(args), f, indent=4)
            with open(os.path.join(output_dir, 'check_point_info.json'), 'w') as f:
                check_point_info = dict(epoch=epoch_num,
                                        global_step=global_step,
                                        accuracy=best_accuracy,
                                        accuracy_history=accuracy_history)
                json.dump(check_point_info, f, indent=4)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training. (e.g. "model = DistributedDataParallel(module=model)")
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            # tokenizer.save_vocabulary(output_dir)
            if args.saving_optimizer:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                logger.info("Saving the optimizer state to %s", output_dir)
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving the scheduler state to %s", output_dir)
            logger.info("\n")
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
    return global_step  # , tr_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt')

    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--num_train_epochs', default=25, type=int)
    parser.add_argument('--save_steps', default=448, type=int)
    parser.add_argument('--logging_steps', default=448, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    # Note:
    # Small 'max_grad_norm' is inevitable when 'batch_size' is large
    # e.g. batch_size=128, max_grad_norm=0.1
    parser.add_argument('--postprocessing-answer', dest='postprocessing_answer', action='store_true',
                        help='post-processing answers')
    parser.add_argument('--train-set-percent', dest='train_set_percent', default=100, type=float,
                        help='train set percent')
    parser.add_argument('--use-shuffled-train', dest='using_shuffled_train', action='store_true')
    parser.add_argument('--save-optimizer', dest='saving_optimizer', action='store_true')
    parser.add_argument('--disable-progress-bar', dest='using_progress_bar', action='store_false')
    parser.add_argument('--pred_batch_size', default=None, type=int)
    
    
    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default=1e-4, type=float)
    args = parser.parse_args()

    if args.pred_batch_size is None:
        args.pred_batch_size = args.batch_size * 4

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    register('logger', make_logger('train', os.path.join(args.save_dir, '{}.log'.format(current_time))))
    logger = register.retrieve('logger')

    # args display
    for k, v in vars(args).items():
        logger.info(k+':'+str(v))

    if not args.using_progress_bar:
        from . import predict as _predict

        def no_tqdm(iterable, *args, **kwargs):
            return iterable

        _predict.tqdm = no_tqdm

    seed_everything(42)

    train(args)


if __name__ == '__main__':
    main()
