import argparse
import logging
import os
import random
from time import time
import tensorflow as tf
import sys

import numpy as np
import wandb

from config import DATASET_NAMES, LOGGING_FORMAT, LOGGING_DATE_FORMAT, LSTM_TRAINED_MODELS, RANDOM_SEED, RESULTS_DIR, \
    WANDB_DIR, LOG_DIR
from src.polarity.baseline.utils import dataset_get_x_y, get_table_result_string, get_sum_info, evaluate_baseline_model
from src.polarity.data.loader import DATASET_LOADERS
from src.polarity.lstm_baseline.embeddings.EmbeddingsVectorizer import EmbeddingsVectorizer
from src.polarity.lstm_baseline.nn_config import OPTIMIZERS_CHOICES, TOKENIZER_CHOICES, LR_SCHEDULER_CHOICES
from src.polarity.lstm_baseline.nn_model import ATTENTIONS_TYPES, ATTENTIONS_ACTIVATIONS, MODELS_GETTER_MAP, \
    draw_histogram_stats, build_and_run
from src.polarity.lstm_baseline.nn_utils import generate_file_name, generate_file_name_transformer, save_model
from src.polarity.preprocessing.Data_Cleaner import text_processor_normalize, Data_Cleaner
from src.utils import disable_tensorflow_gpus

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)



def main_param():
    parser = build_parser()

    args = parser.parse_args()
    if args.silent is True:
        logging.root.setLevel(level=logging.ERROR)
    else:
        logging.root.setLevel(level=logging.INFO)

    run_single(args)

def run_single(args):
    result_file = generate_file_name(vars(args))
    result_file = result_file + ".results"
    result_file = os.path.join(RESULTS_DIR, result_file)

    num_repeat = args.num_repeat

    t0 = time()
    for run in range(1, (num_repeat + 1)):
        run_t0 = time()
        print("*=" * 70)
        print("Repeat:" + str(run))

        main_exp(result_file)

        repeat_time = time() - run_t0
        print("Time for run:" + str(run) + " is:" + str(int(repeat_time)))

    print("==" * 70)
    total_time = time() - t0
    print("Total time for evaluation:" + str(int(total_time)) + " sec")
    print("==" * 70)

def main_exp(result_file):

    parser = build_parser()
    args = parser.parse_args()

    args = init_loging(args, parser, result_file, generating_fce=generate_file_name, set_format=False)

    logger.info(f"Running baseline with the following parameters:{args}")
    logger.info("-------------------------")
    main(args)





def main(args):
    if args.use_cpu is True:
        disable_tensorflow_gpus()

    if args.enable_wandb is True:
        try:
            wandb.init(project="XXX", name=args.config_name, config=vars(args), reinit=True,
                       dir=WANDB_DIR)
        except Exception as e:
            logger.error("Error WANDB with exception e:" + str(e))

    if args.set_seed is True:
        os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

    # data loading stuff
    data_loader = DATASET_LOADERS[args.dataset_name](args.max_train_data, args.binary)
    print_dataset_info(args, data_loader)

    if args.use_only_train_data:
        data_train = data_loader.get_train_dev_data()
        X_dev, y_dev = None, None
    else:
        data_train = data_loader.get_train_data()
        data_dev = data_loader.get_dev_data()
        X_dev, y_dev = dataset_get_x_y(data_dev, 'text', 'label')

    data_test = data_loader.get_test_data()

    clas_names = data_loader.get_class_names()
    X_train, y_train = dataset_get_x_y(data_train, 'text', 'label')
    X_test, y_test = dataset_get_x_y(data_test, 'text', 'label')

    print("Train data:" + str(get_sum_info(X_train, data_train['label_text'].astype(str).values.tolist(), clas_names)))
    if args.use_only_train_data is False:
        print("Dev data:" + str(get_sum_info(X_dev, data_dev['label_text'].astype(str).values.tolist(), clas_names)))
    print("Test data:" + str(get_sum_info(X_test, data_test['label_text'].astype(str).values.tolist(), clas_names)))

    if args.draw_histogram_stats is True:
        draw_histogram_stats(data_loader, args)
        return

    # build and train model
    f1_dev, accuracy_dev, precision_dev, recall_dev, model, tensorboard_log_dir, train_time, callback_list, \
    vectorizer, param_dict, cached_x_vectors_test_path = build_and_run(X_train, y_train, X_dev, y_dev,
                                                                       data_loader.get_class_num(), args)

    # if wandb is enabled and there are dev data
    if args.enable_wandb is True and args.use_only_train_data is False:
        try:
            wandb.run.summary['f1_dev'] = f1_dev
            wandb.run.summary['accuracy_dev'] = accuracy_dev
            wandb.run.summary['precision_dev'] = precision_dev
            wandb.run.summary['recall_dev'] = recall_dev
        except Exception as e:
            logger.error("Error WANDB with exception e:" + str(e))

    # extract vectors for test data
    x_test_vector = vectorizer.vectorize(X_test, cache_file_x=cached_x_vectors_test_path)

    # evaluate on test data
    f1, accuracy, precision, recall, _, _ = \
        evaluate_baseline_model(model, x_test_vector, y_test,
                                average='macro', grid=False,
                                categorical_output=True, prob_output=True)

    if args.enable_wandb is True:
        try:
            wandb.run.summary['f1'] = f1
            wandb.run.summary['accuracy'] = accuracy
            wandb.run.summary['precision'] = precision
            wandb.run.summary['recall'] = recall
        except Exception as e:
            logger.error("Error WANDB with exception e:" + str(e))

    dataset_name = args.dataset_name
    if dataset_name == 'combined':
        tmp = '-'.join(args.combined_datasets)
        dataset_name = dataset_name + '-' + tmp

        # get result string for table
    result_string, only_results = get_table_result_string(f'{dataset_name}\tNN train test:{args.model_name} {args}',
                                                          f1, 0, accuracy, 0, precision, 0, recall, 0, train_time)

    if args.use_only_train_data is False:
        # get result string for dev data
        result_string_dev, only_results_dev = get_table_result_string("Dev results", f1_dev, 0, accuracy_dev, 0,
                                                                      precision_dev, 0, recall_dev, 0, 0)

        only_results += "\t" + only_results_dev

    result_string = "\n-----------Test Results------------\n" + result_string
    if args.use_only_train_data is False:
        result_string += "\n-----------Dev Results------------\n" + result_string_dev

    print(result_string)
    # logger.info(result_string)

    print("\n\n\n-----------Save results------------\n" + str(only_results) + "\n\n\n")
    results_file = args.result_file
    with open(results_file, "a") as f:
        f.write(only_results + "\n")

    if args.enable_wandb is True:
        try:
            wandb.run.summary['results_string'] = only_results
        except Exception as e:
            logger.error("Error WANDB with exception e:" + str(e))

    print(70 * '=')
    print("Test results")
    multiply = 100
    print("f1 score: %.3f%%" % (f1 * multiply))
    print("accuracy: %.3f%%" % (accuracy * multiply))
    print("precision: %.3f%%" % (precision * multiply))
    print("recall: %.3f%%" % (recall * multiply))
    print(70 * '=')

    # save model
    dict_args = vars(args)
    save_model(model, dict_args, dict_args['embeddings_file'], tensorboard_log_dir, train_time,
               accuracy, f1, precision, recall, callback_list, LSTM_TRAINED_MODELS, param_dict,
               result_string)

    wandb.finish()


def print_dataset_info(args, dataset_loader):
    dataset_df = dataset_loader.load_entire_dataset()

    # just print some example
    sentence = dataset_df['text'][150]

    use_data_cleaner = args.use_data_cleaner
    data_cleaner = None
    if use_data_cleaner is True:
        data_cleaner = Data_Cleaner(text_processor=text_processor_normalize, lower_case=False,
                                    elong_words=True, elong_punct=True)

    vectorizer = EmbeddingsVectorizer(word_map=None,
                                      tokenizer_name=args.tokenizer,
                                      use_stemmer=args.use_stemmer,
                                      data_cleaner=data_cleaner,
                                      we_matrix=None,
                                      language=None,
                                      max_length=None)

    tokens = vectorizer.tokenize_to_list(sentence)

    logger.info(f' Sentence: {sentence}')
    logger.info(f'   Tokens: {tokens}')


def init_loging(args, parser, result_file, generating_fce=generate_file_name_transformer, set_format=True):

    config_name = generating_fce(vars(args))
    if args.full_mode is True:
        # select the first epoch config name
        file_name = os.path.join(LOG_DIR, config_name[1] + '-full_mode.log')
    else:
        file_name = os.path.join(LOG_DIR, config_name + '.log')
    parser.add_argument("--config_name",
                        default=config_name)
    parser.add_argument("--result_file",
                        default=result_file)
    args = parser.parse_args()

    if set_format:
        # just to reset logging settings
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(format=LOGGING_FORMAT,
                            datefmt=LOGGING_DATE_FORMAT,
                            filename=file_name)

        formatter = logging.Formatter(fmt=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)


        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logging.root.setLevel(level=logging.INFO)
        if args.silent is True:
            # logging.root.setLevel(level=logging.ERROR)
            console_handler.setLevel(level=logging.ERROR)
        else:
            # logging.root.setLevel(level=logging.INFO)
            console_handler.setLevel(level=logging.INFO)

        logging.getLogger().addHandler(console_handler)

    return args


def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Required parameters
    parser.add_argument("--dataset_name",
                        required=True,
                        choices=DATASET_NAMES,
                        help="The dataset that will be used, they correspond to names of folders")

    parser.add_argument("--model_name",
                        required=True,
                        choices=list(MODELS_GETTER_MAP.keys()),
                        help="Name of the model that will be used")

    # Embeddings params
    parser.add_argument("--embeddings_file",
                        required=True,
                        help='File name of the embeddings')

    parser.add_argument("--embeddings_size",
                        default=300,
                        type=int,
                        required=True,
                        help="Dimension of word embeddings")

    # Optional parameters
    parser.add_argument("--max_words",
                        default=None,
                        type=int,
                        help="Maximum words that will be loaded from word embeddings")

    parser.add_argument("--trainable_word_embeddings",
                        default=False,
                        action='store_true',
                        help="If set, the word embeddings layer is updated during training")

    # Preprocessing options
    parser.add_argument("--tokenizer",
                        default='white_space',
                        choices=TOKENIZER_CHOICES,
                        help="Possible tokenizers that are used for tokenization")

    parser.add_argument("--use_stemmer",
                        default=False,
                        action='store_true',
                        help='If set program will use stemmer in attempt to find word in embeddings for unknown words')

    parser.add_argument("--use_data_cleaner",
                        default=False,
                        action='store_true',
                        help='If set program will use data cleaner before tokenization')

    # Model options
    parser.add_argument("--set_seed",
                        default=False,
                        action='store_true',
                        help='If set the program will set seed value for all random initialization and '
                             'it should produce reproducible results')

    parser.add_argument("--max_seq_len",
                        default=48,
                        type=int,
                        help="Maximum sequence length of tokens  used as an input for the model")

    parser.add_argument("--batch_size",
                        default=256,
                        type=int,
                        help="Batch size")

    parser.add_argument("--epoch_count",
                        default=1,
                        type=int,
                        help="Number of epochs for fine tuning")

    parser.add_argument("--lr",
                        default=0.005,
                        type=float,
                        help="(Initial) learning rate, if --lr_scheduler is used, this is used as its initial learning rate")

    parser.add_argument("--lr_scheduler_name",
                        default="none",
                        choices=LR_SCHEDULER_CHOICES,
                        help="Learning rate scheduler to be used, if none option then only the learning rate from"
                             "--lr parameter is used")

    parser.add_argument("--warm_up_steps",
                        default=0,
                        type=float,
                        help="Number of warmup steps, the --lr_scheduler_name parameter must be set to other than "
                             "none, to be applied warmup, if less than 1 than it is used as percents/fraction of the total"
                             " number of steps,")

    parser.add_argument("--optimizer",
                        default='Adam',
                        choices=OPTIMIZERS_CHOICES,
                        help="Optimizer one of:" + str(OPTIMIZERS_CHOICES))

    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay parameter, is only applied when AdamW optimzer is used")

    parser.add_argument("--l2_rnn_reg",
                        default=0.0,
                        type=float,
                        help="Value for l2 reguralization applied to a RNN units")

    parser.add_argument("--l2_reg",
                        default=0.0,
                        type=float,
                        help="Value for l2 reguralization for the last Dense layer with softmax, a good initial value can be 0.01,"
                             " watch out, probably it should not be used with the --weight_decay paramater, since"
                             " it would introduce another reguralization")

    parser.add_argument("--we_noise",
                        default=0.0,
                        type=float,
                        help="Gausian noise, that is added as a layer after the embedding layer")

    parser.add_argument("--use_cpu",
                        default=False,
                        action='store_true',
                        help="If set, the program will always run on CPU")

    parser.add_argument("--use_early_stopping",
                        default=False,
                        action='store_true',
                        help="If set, the early stopping is applied")

    parser.add_argument("--use_masking",
                        default=True,
                        action='store_true',
                        help="If set, words are masked in embeddings layer")

    parser.add_argument("--dropout_words",
                        default=0.0,
                        type=float,
                        help="Dropout after word embeddings")

    parser.add_argument("--dropout_rnn",
                        default=0.2,
                        type=float,
                        help="Dropout after the rnn layer")

    parser.add_argument("--dropout_rnn_recurrent",
                        default=0.0,
                        type=float,
                        help="Dropout between reccurrent units")

    parser.add_argument("--rnn_cells",
                        default=128,
                        type=int,
                        help="Number of units in RNN layer")

    parser.add_argument("--bidirectional",
                        default=True,
                        action='store_true',
                        help="If set, rnn layers are bidirectional")

    parser.add_argument("--final_layer_size",
                        default=0,
                        type=int,
                        help="Size of final Dense layer, if set to 0 no layer is added")

    parser.add_argument("--dropout_final",
                        default=0.5,
                        type=float,
                        help="Dropout after rnn layer")

    parser.add_argument("--use_attention",
                        action='store_true',
                        default=False,
                        help="Whether to use attention")

    parser.add_argument("--use_batch_norm",
                        action='store_true',
                        default=False,
                        help="Whether to use batch normalization after each layer")

    parser.add_argument("--dropout_attention",
                        default=0.0,
                        type=float,
                        help="Dropout after attention layer")

    parser.add_argument("--attention_type",
                        default='additive',
                        type=str,
                        choices=ATTENTIONS_TYPES,
                        help="Type of used attention")

    parser.add_argument("--attention_activation",
                        default='none',
                        choices=ATTENTIONS_ACTIVATIONS.keys(),
                        help='Activation of attention score')

    # Other parameters
    parser.add_argument("--enable_wandb",
                        default=False,
                        action='store_true',
                        help="If set, the program will use wandb for logging, otherwise not")

    parser.add_argument("--silent",
                        default=False,
                        action='store_true',
                        help="If used, logging is set to ERROR level, otherwise INFO is used")

    parser.add_argument("--binary",
                        default=False,
                        action='store_true',
                        help="If used the polarity task is treated as binary classification, i.e., positve/negative"
                             " The neutral examples are dropped")

    parser.add_argument("--draw_histogram_stats",
                        action="store_true",
                        help="If specified the statistics about the given datasets are printed and saved to tmp folder"
                             " The default tokenizer of embedding vectorizer is used, training does not continue after this")

    parser.add_argument("--use_only_train_data",
                        default=False,
                        action='store_true',
                        help="If set, the program will use training and develoopment data for training, i.e. it will use"
                             "train + dev for training, no validation is done during training")

    # TODO eval
    parser.add_argument("--eval",
                        default=False,
                        action='store_true',
                        help="If used, the evaluation on a given dataset and model is performed,"
                             "if used all other param \"--model_name\" must be a path to the model folder, --tokenizer_type must be set")

    parser.add_argument("--num_repeat",
                        default=1,
                        type=int,
                        help='Number of repeats of the experiment')

    # TODO
    parser.add_argument("--full_mode",
                        default=False,
                        action='store_true',
                        help="If set, the program will evaluate the model after each epoch and write results into the result files")

    # TODO
    parser.add_argument("--max_train_data",
                        default=-1,
                        type=float,
                        help="Amount of data that will be used for training, "
                             "if (-inf, 0> than it is ignored, "
                             "if (0,1> than percentage of the value is used as training data, "
                             "if (1, inf) absolute number of training examples is used as training data")


    return parser


if __name__ == '__main__':
    main_param()