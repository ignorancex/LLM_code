import logging
import os
import random
import time
import gc
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
import math

import torch
from sklearn.metrics import classification_report
from torch import nn
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup, \
    BertForSequenceClassification, AlbertForSequenceClassification, BertTokenizerFast, \
    BertModel, AlbertModel, get_constant_schedule, BertForPreTraining, XLMForSequenceClassification, \
    XLMTokenizer, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, get_cosine_schedule_with_warmup, AutoConfig, \
    AutoModelForSequenceClassification, AutoTokenizer

from config import RANDOM_SEED, MODELS_DIR, TRANSFORMERS_TRAINED_MODELS, WANDB_DIR
from src.polarity.baseline.utils import evaluate_predictions, get_table_result_string
from src.polarity.data.loader import DATASET_LOADERS
from src.polarity.lstm_baseline.lr_schedulers.schedule import get_transformer_polynomial_decay_schedule_with_warmup
from src.polarity.lstm_baseline.nn_utils import save_model_transformer
from src.polarity.pytorch.dataset import build_data_loader
from src.polarity.pytorch.train import run_training, eval_model, get_predictions, SentimentClassifier
from src.utils import visaulize_training, format_time, disable_tensorflow_gpus

logger = logging.getLogger(__name__)

SCHEDULERS = ["linear_wrp", "constant", "cosine_wrp", "polynomial_wrp"]



class TorchTuner(object):
    def __init__(self, args):
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch_num
        self.dataset_loader = DATASET_LOADERS[args.dataset_name](args.max_train_data, args.binary)
        self.use_custom_model = args.use_custom_model
        self.use_only_train_data = args.use_only_train_data
        self.data_parallel = args.data_parallel

        if args.use_random_seed is False:
            # init RANDOM_SEED
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed_all(RANDOM_SEED)

        if args.use_cpu is True:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running fine-tuning on:{self.device}")
        # print_gpu_info()

        try:
            GPUs_count = torch.cuda.device_count()
            logger.info("We have avialiable more GPUs:" + str(GPUs_count))

            logger.info("We try to run it on multiple GPUs:" + str(self.data_parallel))

        except Exception as e:
            logger.info("Trying to init data paralelsim")


        self.tokenizer = self.load_tokenizer(args)

        # Load dataset
        logger.info("Loading dataset")
        self.print_dataset_info(args)
        if self.use_only_train_data:
            self.train_size = len(self.dataset_loader.get_train_dev_data())
            self.dev_size = 0
        else:
            self.train_size = len(self.dataset_loader.get_train_data())
            self.dev_size = len(self.dataset_loader.get_dev_data())

        self.test_size = len(self.dataset_loader.get_test_data())
        self.num_labels = self.dataset_loader.get_class_num()

        if self.use_only_train_data:
            self.train_data_loader = build_data_loader(self.dataset_loader.get_train_dev_data(), self.tokenizer,
                                                       self.max_seq_len, self.batch_size)
            self.dev_data_loader = None
        else:
            self.train_data_loader = build_data_loader(self.dataset_loader.get_train_data(), self.tokenizer,
                                                       self.max_seq_len, self.batch_size)
            self.dev_data_loader = build_data_loader(self.dataset_loader.get_dev_data(), self.tokenizer, self.max_seq_len,
                                                     self.batch_size)

        self.test_data_loader = build_data_loader(self.dataset_loader.get_test_data(), self.tokenizer, self.max_seq_len,
                                                  self.batch_size)

        if args.dataset_name == 'imdb-csfd':
            self.dev_data_czech_loader = build_data_loader(self.dataset_loader.get_dev_data_czech(), self.tokenizer,
                                                       self.max_seq_len, self.batch_size)
            self.dev_czech_size = len(self.dataset_loader.get_dev_data_czech())

        if args.dataset_name == 'csfd-imdb':
            self.dev_data_eng_loader = build_data_loader(self.dataset_loader.get_dev_data_eng(), self.tokenizer,
                                                           self.max_seq_len, self.batch_size)
            self.dev_eng_size = len(self.dataset_loader.get_dev_data_eng())

        logger.info("Train size:" + str(self.train_size))
        logger.info("Dev size:" + str(self.dev_size))
        logger.info("Test size:" + str(self.test_size))
        if args.dataset_name == 'imdb-csfd':
            logger.info("Czech dev size:" + str(self.dev_czech_size))
        if args.dataset_name == 'csfd-imdb':
            logger.info("English dev size:" + str(self.dev_eng_size))

        logger.info("Dataset loaded")
        logger.info(f"Number of labels in dataset:{self.num_labels}")

    # TODO Autotokenizer
    def load_tokenizer(self, args):
        if args.tokenizer_type == 'berttokenizer':
            tokenizer = BertTokenizer.from_pretrained(args.model_name)

        elif args.tokenizer_type == 'berttokenizerfast':
            tokenizer_path = os.path.abspath(os.path.join(args.model_name, "vocab.txt"))
            print("Tokenizer Path:" + tokenizer_path)
            tokenizer = BertTokenizerFast(tokenizer_path, strip_accents=False)

        elif args.tokenizer_type == 'berttokenizerfast-cased':
            tokenizer_path = os.path.abspath(os.path.join(args.model_name, "vocab.txt"))
            print("Tokenizer Path:" + tokenizer_path)
            tokenizer = BertTokenizerFast(tokenizer_path, strip_accents=False, do_lower_case=False)
            # tokenizer = AutoTokenizer(tokenizer_path, strip_accents=False, do_lower_case=False)

        elif args.tokenizer_type == 'xlmtokenizer':
            tokenizer = XLMTokenizer.from_pretrained(args.model_name)

        elif args.tokenizer_type == 'xlm-r-tokenizer':
            tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
        else:
            raise Exception(f"Unknown type of tokenizer{args.tokenizer_type}")

        return tokenizer

    # in order to preserve the v3.x behavior we have to set
    # https://huggingface.co/transformers/migration.html
    # return dict as true
    # TODO Autotokenizer
    def load_model(self, args):
        # TODO custom model
        if args.from_tf:
            disable_tensorflow_gpus()
        if args.use_custom_model:
            if args.model_type == 'bert':
                hugging_face_model = BertModel.from_pretrained(args.model_name,
                                                               from_tf=args.from_tf,
                                                               return_dict=False)
            elif args.model_type == 'albert':
                hugging_face_model = AlbertModel.from_pretrained(args.model_name,
                                                                 from_tf=args.from_tf,
                                                                 return_dict=False)
            else:
                raise Exception(f"Unkown model type:{args.model_type}")

            model = SentimentClassifier(self.num_labels, hugging_face_model, args.custom_model_dropout)

            if args.freze_base_model is True:
                raise Exception("This combination is not supported, i.e., the freeze base model is not implemented, must be added")

        else:
            if args.model_type == 'bert':
                model = BertForSequenceClassification.from_pretrained(args.model_name,
                                                                      num_labels=self.num_labels,
                                                                      from_tf=args.from_tf,
                                                                      return_dict=False)
            elif args.model_type == 'albert':
                model = AlbertForSequenceClassification.from_pretrained(args.model_name,
                                                                        num_labels=self.num_labels,
                                                                        from_tf=args.from_tf,
                                                                        return_dict=False)
            elif args.model_type == 'xlm':
                model = XLMForSequenceClassification.from_pretrained(args.model_name,
                                                                     num_labels=self.num_labels,
                                                                     from_tf=args.from_tf,
                                                                     return_dict=False)
            elif args.model_type == 'xlm-r':
                model = XLMRobertaForSequenceClassification.from_pretrained(args.model_name,
                                                                            num_labels=self.num_labels,
                                                                            from_tf=args.from_tf,
                                                                            return_dict=False)
            else:
                raise Exception(f"Unkown model type:{args.model_type}")

            if args.freze_base_model is True:
                logger.info("Freezing base model layers")
                for name, param in model.base_model.named_parameters():
                    print(name)
                    param.requires_grad = False

        # model_new = BertModel.from_pretrained(args.model_name)
        # model.save_pretrained(os.path.join(MODELS_DIR, 'pavlov-pokus'))
        # for param in model.base_model.parameters():
        #     print()
        #     param.requires_grad = False
        try:
            # pp = 0
            # for p in list(model.parameters()):
            #     nn = 1
            #     for s in list(p.size()):
            #         nn = nn * s
            #     pp += nn
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            pp = sum([np.prod(p.size()) for p in model_parameters])
            logger.info("Number of parameters for model:" + str(args.model_name) + " is:" + str(pp))
        except Exception as e:
            logger.error("Error during count number:" + str(e))

        return model


    def get_lr_scheduler(self, optimizer, total_steps):
        warm_up_steps = self.args.warm_up_steps
        scheduler_name = self.args.scheduler

        if warm_up_steps > 0:
            if warm_up_steps == 1:
                raise Exception("Warmup steps cannot be 1")
            if warm_up_steps < 1:
                warm_up_steps = warm_up_steps * total_steps
                warm_up_steps = math.ceil(warm_up_steps)

        logger.info("Number of warm up steps:" + str(warm_up_steps) + " out of:" + str(total_steps) + " original warmup steps:" + str(self.args.warm_up_steps))

        # https://huggingface.co/transformers/main_classes/optimizer_schedules.html#learning-rate-schedules-pytorch
        if scheduler_name == 'linear_wrp':
            # linearly decreasing
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warm_up_steps,
                num_training_steps=total_steps
            )
        elif scheduler_name == 'cosine_wrp':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warm_up_steps,
                num_training_steps=total_steps
            )
        elif scheduler_name == 'polynomial_wrp':
            scheduler = get_transformer_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warm_up_steps,
                num_training_steps=total_steps,
                power=2
            )
        elif scheduler_name == 'constant':
            scheduler = get_constant_schedule(optimizer)
        else:
            raise Exception(f"Unkonwn scheduler:{self.args.scheduler}")

        return scheduler

    def fine_tune(self, args):
        # load model
        model = self.load_model(args)

        if args.data_parallel is True:
            if torch.cuda.device_count() > 1:
                logger.info("Trying to apply data parallesism, number of used GPUs:" + str(torch.cuda.device_count()))
                # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
                model = nn.DataParallel(model)
            else:
                logger.info("Data parallelism is enabled but there is only GPUs:" + str(torch.cuda.device_count()))

        if args.enable_wandb is True:
            wandb_tmp_name = str(args.config_name)
            try:
                wandb.init(project="czert-transformers",name=wandb_tmp_name, config=vars(args),
                           dir=WANDB_DIR,
                           reinit=True)
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))
            # If we run multiple runs in one python script, there is a memory leak in wandb causing OOM on GPU after
            # some time
            # wandb.watch(model)

        # self.print_model_info(model, args)

        # move it to device
        model = model.to(self.device)

        # Run training
        optimizer = AdamW(model.parameters(), lr=self.args.lr, correct_bias=False)
        total_steps = len(self.train_data_loader) * self.epoch_num

        scheduler = self.get_lr_scheduler(optimizer, total_steps)

        loss_fn = nn.CrossEntropyLoss().to(self.device)

        t0 = time.time()
        # if in full_mode we need to pass self instance
        if args.full_mode:
            tuner = self
        else:
            tuner = None

        history = run_training(self.epoch_num, model, self.train_data_loader, self.dev_data_loader,
                               loss_fn, optimizer, scheduler, self.use_custom_model, self.device, self.dev_size,
                               args.print_stat_frequency, args.enable_wandb, args.data_parallel, args, tuner)

        train_time = time.time() - t0
        logger.info(f'Total time for training:{format_time(train_time)}')

        # run it only if it is not in full mode
        if args.full_mode is False:
            self.perform_train_eval(model, args, train_time, history, args.epoch_num)

        print(70*"---")
        # print_gpu_info()
        # del model, optimizer, scheduler
        # gc.collect()
        # # Clean memory
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        # print("--------")
        # print_gpu_info()
        print(70 * "---")

    def perform_train_eval(self, model, args, train_time, model_history, curr_epoch):
        # test_acc, _ = eval_model(
        #     model,
        #     self.test_data_loader,
        #     loss_fn,
        #     self.device,
        #     self.test_size,
        #     self.use_custom_model
        # )
        #
        # logger.info(f"Accuracy test:{test_acc.item()}")


        y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
            model,
            self.test_data_loader,
            self.use_custom_model,
            self.device
        )

        clas_report = classification_report(y_test, y_pred, target_names=self.dataset_loader.get_class_names())
        print(clas_report)
        logger.info(clas_report)

        f1, accuracy, precision, recall = evaluate_predictions(y_pred, y_test)

        dataset_name = args.dataset_name
        if dataset_name == 'combined':
            tmp = '-'.join(args.combined_datasets)
            dataset_name = dataset_name + '-' + tmp

        result_string, only_results = get_table_result_string(
            f'{dataset_name}\tTransformer train test:{args.model_name} {args} curr_epoch:{str(curr_epoch)}',
            f1, 0, accuracy, 0, precision, 0, recall, 0, train_time)

        if args.enable_wandb is True:
            try:
                wandb.run.summary['f1'] = f1
                wandb.run.summary['accuracy'] = accuracy
                wandb.run.summary['precision'] = precision
                wandb.run.summary['recall'] = recall
                wandb.run.summary['class_report'] = str(clas_report)
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))


        result_string = "\n-----------Test Results------------\n\t" + result_string

        if self.use_only_train_data is False:
            _, y_dev_pred, _, y_dev = get_predictions(
                model,
                self.dev_data_loader,
                self.use_custom_model,
                self.device
            )
            f1_dev, accuracy_dev, precision_dev, recall_dev = evaluate_predictions(y_dev_pred, y_dev)

            result_string_dev, only_results_dev = get_table_result_string("Dev results", f1_dev, 0, accuracy_dev, 0, precision_dev, 0,
                                                        recall_dev,
                                                        0, 0)
            if args.enable_wandb is True:
                try:
                    wandb.run.summary['f1_dev'] = f1_dev
                    wandb.run.summary['accuracy_dev'] = accuracy_dev
                    wandb.run.summary['precision_dev'] = precision_dev
                    wandb.run.summary['recall_dev'] = recall_dev
                except Exception as e:
                    logger.error("Error WANDB with exception e:" + str(e))



            only_results +="\t" + only_results_dev
            result_string += "\n-----------Dev Results------------\n" + result_string_dev

        if args.dataset_name == 'imdb-csfd':
            _, y_dev_pred_czech, _, y_dev_czech = get_predictions(
                model,
                self.dev_data_czech_loader,
                self.use_custom_model,
                self.device
            )

            f1_dev_czech, accuracy_dev_czech, precision_dev_czech, recall_dev_czech = evaluate_predictions(y_dev_pred_czech, y_dev_czech)

            result_string_dev_czech, only_results_dev_czech = get_table_result_string("Dev results czech", f1_dev_czech, 0, accuracy_dev_czech, 0,
                                                                          precision_dev_czech, 0,
                                                                          recall_dev_czech,
                                                                          0, 0)
            print(70 * '*-')
            print("Test CZECH Test F1:{:.4f}".format(f1))
            print("Test CZECH Test accuracy:{:.4f}".format(accuracy))
            print("Test CZECH Test precision:{:.4f}".format(precision))
            print("Test CZECH Test recall:{:.4f}".format(recall))
            print(70 * '*-')

            print(70 * '*-')
            print("DEV CZECH  F1:{:.4f}".format(f1_dev_czech))
            print("DEV CZECH  accuracy:{:.4f}".format(accuracy_dev_czech))
            print("DEV CZECH  precision:{:.4f}".format(precision_dev_czech))
            print("DEV CZECH  recall:{:.4f}".format(recall_dev_czech))
            print(70 * '*-')


            try:
                print(70 * '*-')
                print("DEV English  F1:{:.4f}".format(f1_dev))
                print("DEV English  accuracy:{:.4f}".format(accuracy_dev))
                print("DEV English  precision:{:.4f}".format(precision_dev))
                print("DEV English  recall:{:.4f}".format(recall_dev))
                print(70 * '*-')
            except Exception:
                logger.info("No dev data for English")


            if args.enable_wandb is True:
                try:
                    wandb.run.summary['f1_dev_czech'] = f1_dev_czech
                    wandb.run.summary['accuracy_dev_czech'] = accuracy_dev_czech
                    wandb.run.summary['precision_dev_czech'] = precision_dev_czech
                    wandb.run.summary['recall_dev_czech'] = recall_dev_czech
                except Exception as e:
                    logger.error("Error WANDB with exception e:" + str(e))

            only_results += "\t" + only_results_dev_czech
            result_string += "\n-----------CZECH Dev Results------------\n" + result_string_dev_czech

        if args.dataset_name == 'csfd-imdb':
            _, y_dev_pred_eng, _, y_dev_eng = get_predictions(
                model,
                self.dev_data_eng_loader,
                self.use_custom_model,
                self.device
            )

            f1_dev_eng, accuracy_dev_eng, precision_dev_eng, recall_dev_eng = evaluate_predictions(y_dev_pred_eng, y_dev_eng)

            result_string_dev_english, only_results_dev_english = get_table_result_string("Dev results english", f1_dev_eng, 0, accuracy_dev_eng, 0,
                                                                          precision_dev_eng, 0,
                                                                          recall_dev_eng,
                                                                          0, 0)
            print(70 * '*-')
            print("Test ENGLISH Test F1:{:.4f}".format(f1))
            print("Test ENGLISH Test accuracy:{:.4f}".format(accuracy))
            print("Test ENGLISH Test precision:{:.4f}".format(precision))
            print("Test ENGLISH Test recall:{:.4f}".format(recall))
            print(70 * '*-')

            print(70 * '*-')
            print("DEV ENGLISH Test F1:{:.4f}".format(f1_dev_eng))
            print("DEV ENGLISH Test accuracy:{:.4f}".format(accuracy_dev_eng))
            print("DEV ENGLISH Test precision:{:.4f}".format(precision_dev_eng))
            print("DEV ENGLISH Test recall:{:.4f}".format(recall_dev_eng))
            print(70 * '*-')


            try:
                print(70 * '*-')
                print("DEV Czech  F1:{:.4f}".format(f1_dev))
                print("DEV Czech  accuracy:{:.4f}".format(accuracy_dev))
                print("DEV Czech  precision:{:.4f}".format(precision_dev))
                print("DEV Czech  recall:{:.4f}".format(recall_dev))
                print(70 * '*-')
            except Exception:
                logger.info("No dev data for English")


            if args.enable_wandb is True:
                try:
                    wandb.run.summary['f1_dev_eng'] = f1_dev_eng
                    wandb.run.summary['accuracy_dev_eng'] = accuracy_dev_eng
                    wandb.run.summary['precision_dev_eng'] = precision_dev_eng
                    wandb.run.summary['recall_dev_eng'] = recall_dev_eng
                except Exception as e:
                    logger.error("Error WANDB with exception e:" + str(e))

            only_results += "\t" + only_results_dev_english
            result_string += "\n-----------English Dev Results------------\n" + result_string_dev_english


        print("\n\n\n-----------Save results------------\n" + str(only_results) + "\n\n\n")
        results_file = args.result_file
        if args.full_mode is True:
            results_file = results_file[curr_epoch]

        with open(results_file, "a") as f:
            f.write(only_results + "\n")

        if args.enable_wandb is True:
            try:
                wandb.run.summary['results_string'] = only_results
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))

        print(result_string)
        logger.info(result_string)

        if args.enable_wandb is True:
            try:
                wandb.run.summary['result_string_head'] = result_string
                wandb.join()
            except Exception as e:
                logger.error("Error WANDB with exception e:" + str(e))

        save_model_transformer(model, self.tokenizer, vars(self.args), self.args.model_save_dir, train_time,
                               accuracy, f1, precision, recall, result_string, model_history, curr_epoch)


    def evaluate_fine_tuned_model(self, args):
        logger.info("Loading model...")
        model = self.load_model(args)

        # move it to device
        logger.info("Moving model to device:" + str(self.device))
        model = model.to(self.device)
        logger.info("Starting evaluation...")



        # loss_fn = nn.CrossEntropyLoss().to(self.device)

        # t0 = time.time()

        # test_acc, _ = eval_model(
        #     model,
        #     self.test_data_loader,
        #     loss_fn,
        #     self.device,
        #     self.test_size,
        #     self.use_custom_model
        # )

        # logger.info(f"Accuracy test:{test_acc.item()}")
        # train_time = time.time() - t0
        # logger.info(f'Total time for evaluation:{format_time(train_time)}')

        y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
            model,
            self.test_data_loader,
            self.use_custom_model,
            self.device,
            batch_size=self.batch_size,
            print_progress=True
        )

        clas_report = classification_report(y_test, y_pred, target_names=self.dataset_loader.get_class_names())
        print(clas_report)

        f1, accuracy, precision, recall = evaluate_predictions(y_pred, y_test)

        print(70 * '*-')
        print("Test F1:{:.4f}".format(f1))
        print("Test accuracy:{:.4f}".format(accuracy))
        print("Test precision:{:.4f}".format(precision))
        print("Test recall:{:.4f}".format(recall))
        print(70*'*-')

        if self.use_only_train_data is False:
            _, y_dev_pred, _, y_dev = get_predictions(
                model,
                self.dev_data_loader,
                self.use_custom_model,
                self.device,
                print_progress=True
            )
            f1_dev, accuracy_dev, precision_dev, recall_dev = evaluate_predictions(y_dev_pred, y_dev)

            print("Dev F1:{:.4f}".format(f1_dev))
            print("Dev accuracy:{:.4f}".format(accuracy_dev))
            print("Dev precision:{:.4f}".format(precision_dev))
            print("Dev recall:{:.4f}".format(recall_dev))
            print(70 * '*-')


    def print_model_info(self, model, args):
        if args.use_custom_model:
            transformer_model = model.hugging_face_model
        else:
            transformer_model = model

        params = list(transformer_model.named_parameters())
        logger.info(70 * "-")
        logger.info('The model has {:} different named parameters.\n'.format(len(params)))
        logger.info("Printing model layers")
        for p in params:
            logger.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        logger.info(70 * "-")

    def print_dataset_info(self, args):
        dataset_df = self.dataset_loader.load_entire_dataset()

        # just print some example
        sentence = dataset_df['text'][150]
        ids = self.tokenizer.encode(sentence, max_length=args.max_seq_len, pad_to_max_length=True)

        logger.info(f' Sentence: {sentence}')
        logger.info(f'   Tokens: {self.tokenizer.convert_ids_to_tokens(ids)}')
        logger.info(f'Token IDs: {ids}')

        if self.args.draw_dataset_stats is True:
            logger.info(f"Saving dataset tokens histogram for tokenizer:{self.args.tokenizer_type}")
            # See distribution of text len
            token_lens = []

            count_i = 0
            for txt in dataset_df.text:
                tokens = self.tokenizer.encode(txt)
                token_lens.append(len(tokens))
                count_i = count_i + 1
                if count_i % 1000 == 0 and count_i > 0:
                    logger.info("Processed:" + str(count_i))

            max_len = max(token_lens)
            avg_len = np.mean(token_lens)
            cnt = Counter(token_lens)
            # sort by key
            cnt = sorted(cnt.items())
            print("Sentence len - Counts")


            dataset_name = args.dataset_name
            if dataset_name == 'combined':
                tmp = '-'.join(args.combined_datasets)
                dataset_name = dataset_name + '-' + tmp

            model_name = args.model_name
            model_name = model_name.replace('/','-')
            tokenizer = args.tokenizer_type
            prefix = dataset_name + '_' + model_name + '-' + tokenizer + '-'
            histogram_file = os.path.join(self.dataset_loader.get_dataset_dir(), prefix + 'histogram.txt')

            with open(histogram_file, mode='w', encoding='utf-8') as f:
                f.write("Average len:{:.4f}".format(avg_len) + '\n')
                f.write("Max len:" +str(max_len) + '\n')
                f.write('length - count'+ '\n')
                for (length, count) in cnt:
                    # print()
                    f.write(str(length) + ' - ' + str(count) + '\n')


            logger.info(f"Max tokens len:{max_len}")
            logger.info(f"Avg tokens len:{avg_len}")

            tokens_histogram_path = os.path.join(self.dataset_loader.get_dataset_dir(), prefix + 'tokens_histogram.png')
            logger.info(f"Tokens histogram image saved to:{tokens_histogram_path}")
            # fig, ax = plt.subplots()


            # based on https://github.com/mwaskom/seaborn/issues/479#issuecomment-333304727
            plt.figure()  # it resets the plot

            # Plotting hist without kde
            ax = sns.distplot(token_lens, kde=False, color='blue')

            # Creating another Y axis
            second_ax = ax.twinx()

            # Plotting kde without hist on the second Y axis
            sns.distplot(token_lens, ax=second_ax, kde=True, hist=False)
            second_ax.set_yticks([])



            # Removing Y ticks from the second axis
            # ax.set(xlabel=f"Token count, max len:{max_len}", ylabel='Frequency')
            ax.set(xlabel=f"Subword Token Counts", ylabel='Frequency')
            # plt.xlabel(f"Token count, max len:{max_len}")
            # plt.ylabel("Frequency")
            # plt.show()

            x_max = 1024
            plt.xticks(np.arange(0, x_max + 1, 100))
            plt.xlim([0, x_max])

            sns.despine(top=True, right=True, left=False, bottom=False)

            plt.savefig(tokens_histogram_path, dpi=400)
            plt.savefig(tokens_histogram_path + ".pdf")
            plt.figure()

            print()



def fine_tune_torch(args):
    logger.info("Fine Tuning with Pytorch")
    tuner = TorchTuner(args)
    if args.draw_dataset_stats is True:
        logger.info("Dataset stats saved")
    elif args.eval is True:
        tuner.evaluate_fine_tuned_model(args)
    else:
        tuner.fine_tune(args)

    print("Deleting tuner")
    del tuner
    # gc.collect()
    # Clean memory
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    print("--------")
    # print_gpu_info()
    print(70 * "---")
    print("Tuner deleted")




def print_gpu_info():
    try:
        logger.info(f"GPU first device name{torch.cuda.get_device_name(0)}")
        # t = torch.cuda.get_device_properties(0).total_memory
        # c = torch.cuda.memory_cached(0)
        # a = torch.cuda.memory_allocated(0)
        # f = c - a  # free inside cache
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        t = info.total
        f = info.free
        a = info.used
        logger.info(f'GPU 0 Memory total    : {int(t / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory free     : {int(f / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory used     : {int(a / (1024.0 * 1024.0))} MiB')

        logger.info(f"GPU first device name{torch.cuda.get_device_name(1)}")
        # t = torch.cuda.get_device_properties(0).total_memory
        # c = torch.cuda.memory_cached(0)
        # a = torch.cuda.memory_allocated(0)
        # f = c - a  # free inside cache
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(1)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        t = info.total
        f = info.free
        a = info.used
        logger.info(f'GPU 0 Memory total    : {int(t / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory free     : {int(f / (1024.0 * 1024.0))} MiB')
        logger.info(f'GPU 0 Memory used     : {int(a / (1024.0 * 1024.0))} MiB')
        # print(f'cached   : {c/(1024.0*1024.0)}')
    except Exception as e:
        logger.info("Exception during:" + str(e))

