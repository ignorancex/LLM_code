import argparse
import logging
import os
import sys

from config import POLARITY_DIR, DATASET_NAMES, BASE_PATH, LOGGING_FORMAT, LOGGING_DATE_FORMAT, \
    LOG_DIR, RESULTS_DIR, TRANSFORMERS_TRAINED_MODELS, TRAINED_MODELS_DIR
from src.polarity.lstm_baseline.nn_utils import generate_file_name_transformer, get_actual_time
from src.polarity.pytorch.finetuning_torch import fine_tune_torch, SCHEDULERS

logging.basicConfig(format=LOGGING_FORMAT,
                    datefmt=LOGGING_DATE_FORMAT)
logger = logging.getLogger(__name__)


def main_tmp(args):
    if args.library == 'pt':
        fine_tune_torch(args)
    else:
        raise Exception(f"Unknown paramter{args.library}")

# https://github.com/ThilinaRajapakse/simpletransformers/issues/515
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def build_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    # Required parameters
    parser.add_argument("--dataset_name",
                        required=True,
                        choices=DATASET_NAMES,
                        help="The dataset that will be used, they correspond to names of folders")

    parser.add_argument("--model_name",
                        required=True,
                        type=str,
                        help="Name of model from hugging face or path to directory with the model")

    # Non required parameters
    parser.add_argument("--eval",
                        default=False,
                        action='store_true',
                        help="If used, the evaluation on a given dataset and model is performed,"
                             "if used all other param \"--model_name\" must be a path to the model folder, --tokenizer_type must be set")

    parser.add_argument("--joint_eval",
                        default=False,
                        action='store_true',
                        help='If used, the evaluation on a given folder is performed, parameter --joint_eval_dir must be specified, and each directory in'
                               'that dir must be a model that is going to be evaluated')

    parser.add_argument("--joint_eval_dir",
                        default=None,
                        type=str,
                        help="The given parameter must be a directory where each directory is one model name, the argument is ignored"
                             "if the --joint_eval parameter is not set")

    parser.add_argument("--model_save_dir",
                        default=TRANSFORMERS_TRAINED_MODELS,
                        type=str,
                        help="Folder where the finetuned model will be saved, default is trained_models/transformers")

    parser.add_argument("--library",
                        default='pt',
                        choices=['pt'],
                        help="Library used for fine-tuning, default is pytorch - pt, tensorflow - tf")

    parser.add_argument("--data_dir",
                        default=POLARITY_DIR,
                        type=str,
                        help=f"The dir with data for polarity, default is set to:{POLARITY_DIR}")

    parser.add_argument("--silent",
                        default=False,
                        action='store_true',
                        help="If used, logging is set to ERROR level, otherwise INFO is used")

    parser.add_argument("--binary",
                        default=False,
                        action='store_true',
                        help="If used the polarity task is treated as binary classification, i.e., positve/negative"
                             " The neutral examples are dropped")

    parser.add_argument("--max_seq_len",
                        default=64,
                        type=int,
                        help="Maximum sequence length of tokens  used as an input for the model")

    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Batch size")

    parser.add_argument("--epoch_num",
                        default=5,
                        type=int,
                        help="Number of epochs for fine tuning")

    parser.add_argument("--lr",
                        default=2e-6,
                        type=float,
                        help="Learning rate")

    parser.add_argument("--scheduler",
                        default='linear_wrp',
                        choices=SCHEDULERS,
                        type=str,
                        help="Schedulre used for scheduling learning rate,"
                             " see https://huggingface.co/transformers/main_classes/optimizer_schedules.html#learning-rate-schedules-pytorch")

    parser.add_argument("--warm_up_steps",
                        default=0,
                        type=float,
                        help="Number of warmup steps, if less than 1 than it is used as percents/fraction of the total"
                             " number of steps, cannot be set to one")

    parser.add_argument("--use_cpu",
                        default=False,
                        action='store_true',
                        help="If set, the program will always run on CPU")

    parser.add_argument("--from_tf",
                        default=False,
                        action='store_true',
                        help="If set, the program will try to load the tensorflow model into pytorch model, in that case"
                             " all GPUs for tensorflow are disabled")
    # Watch the cross-lingual datasets
    parser.add_argument("--use_only_train_data",
                        default=False,
                        action='store_true',
                        help="If set, the program will use training and develoopment data for training, i.e. it will use"
                             "train + dev for training, no validation is done during training")

    parser.add_argument("--model_type",
                        default='bert',
                        choices=['bert', 'albert', 'xlm', 'xlm-r'],
                        help="Type of model that will be loaded")

    parser.add_argument("--tokenizer_type",
                        default='berttokenizer',
                        choices=['berttokenizer', 'berttokenizerfast', 'xlmtokenizer', 'xlm-r-tokenizer', 'berttokenizerfast-cased'],
                        help="Type of tokenizer that will be used, the tokenizer config must be in"
                             " the same folder as the model, specified by parameter model_name")

    parser.add_argument("--use_custom_model",
                        default=False,
                        action="store_true",
                        help="If set, the program will use custom last layer instead of XXXForSequenceClassification class"
                             " from hugging face")

    parser.add_argument("--custom_model_dropout",
                        default=0.1,
                        type=float,
                        help="Only relevant if --use_custom_model is used, set the dropout for custom model")

    parser.add_argument("--print_stat_frequency",
                        default=25,
                        type=int,
                        help="Specify the frequency of printing train info, i.e. after how many batches will be the "
                             "info printed")

    parser.add_argument("--draw_dataset_stats",
                        action="store_true",
                        help="If specified the statistics about the given datasets are printed and saved to dataset folder"
                             "don't forget to specify the correct tokenizer, it can be slower because it loads the entire dataset"
                             " and it tokenizes it, The fine-tuning is not run with this parameter")

    parser.add_argument("--use_random_seed",
                        default=False,
                        action='store_true',
                        help="If set, the program will NOT set a seed value for all random sources, "
                             "if set the results should NOT be same across runs with the same configuration.")

    parser.add_argument("--enable_wandb",
                        default=False,
                        action='store_true',
                        help="If set, the program will use wandb for logging, otherwise not")

    parser.add_argument("--data_parallel",
                        default=False,
                        action='store_true',
                        help='If set, the program will run on all avaialble GPUs')

    parser.add_argument("--num_repeat",
                        default=1,
                        type=int,
                        help="Specify the number, of how many times will be the experiment repeated")

    parser.add_argument("--freze_base_model",
                        default=False,
                        action='store_true',
                        help="If set, the program will freeze all parametrs from base model (except the last linear layer) "
                             "It does NOT WORK for custom model, must be implemented")

    parser.add_argument("--full_mode",
                        default=False,
                        action='store_true',
                        help="If set, the program will evaluate the model after each epoch and write results into the result files")

    parser.add_argument("--max_train_data",
                        default=-1,
                        type=float,
                        help="Amount of data that will be used for training, "
                             "if (-inf, 0> than it is ignored, "
                             "if (0,1> than percentage of the value is used as training data, "
                             "if (1, inf) absolute number of training examples is used as training data")

    return parser


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

def main_exp(result_file):
    parser = build_parser()
    args = parser.parse_args()

    args = init_loging(args, parser, result_file)

    logger.info(f"Running fine-tuning with the following parameters:{args}")
    logger.info("-------------------------")
    main_tmp(args)

def main():
    parser = build_parser()
    args = parser.parse_args()
    result_file = generate_file_name_transformer(vars(args))

    if args.full_mode is True:
        for key, val in result_file.items():
            tmp_val = val + ".results"
            tmp_val = os.path.join(RESULTS_DIR, tmp_val)
            result_file[key] = tmp_val
    else:
        result_file = result_file + ".results"
        result_file = os.path.join(RESULTS_DIR, result_file)

    main_exp(result_file)

if __name__ == '__main__':
    main()