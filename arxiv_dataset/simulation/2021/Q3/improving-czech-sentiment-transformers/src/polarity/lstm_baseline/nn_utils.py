import datetime
import json
import logging
import math
import os
import pickle
from contextlib import redirect_stdout
from distutils.dir_util import copy_tree
from pathlib import Path

import tensorflow_addons as tfa
from nltk import WhitespaceTokenizer, word_tokenize, ToktokTokenizer
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import PolynomialDecay, \
    CosineDecay, ExponentialDecay

# from eval import get_stats_string, get_excel_format
from src.polarity.lstm_baseline.lr_schedulers.schedule import WarmUpDecay

logger = logging.getLogger(__name__)


def get_actual_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H.%M_%S.%f")


def load_model_keras(model_path):
    return load_model(model_path)


def append_f1_to_file_name(name_file, f1):
    name_file = name_file + "_F1-%.4f" % (f1)
    return name_file


def generate_file_name_transformer(config):
    """
    If the full_mode is True, it returns dictionary otherwise it returns only one file

    :param config:
    :return:
    """
    time = get_actual_time()
    epochs = config['epoch_num']
    binary = config['binary']
    batch_size = config['batch_size']
    model_name = Path(config['model_name']).name
    full_mode = config['full_mode']
    max_train_data = config['max_train_data']
    if max_train_data > 1:
        max_train_data = int(max_train_data)

    dataset_name = config['dataset_name']
    if dataset_name == 'combined':
        tmp = '-'.join(config['combined_datasets'])
        dataset_name = dataset_name + '-' + tmp


    if config['eval'] is True:
        model_name = model_name[:30]

    num_iter = 1
    if full_mode is True:
        num_iter = epochs

    name_files = {}
    name_file = None
    for i in range(1, num_iter + 1):
        # if we are in full mode we change the epochs
        if full_mode is True:
            epochs = i

        name_file = model_name + "_" \
                    + dataset_name \
                    + "_BS-" + str(batch_size) \
                    + "_EC-" + str(epochs) \
                    + "_LR-%.7f" % (config['lr']) \
                    + "_LEN-" + str(config['max_seq_len']) \
                    + "_SCH-" + str(config['scheduler']) \
                    + "_TRN-" + str(config['use_only_train_data']) \
                    + "_MXT-" + str(max_train_data) \
                    + "_FRZ-" + str(config['freze_base_model']) \
                    + "_BIN-" + str(binary) \
                    + "_F-" + str(full_mode)

        name_file += "_" + time
        name_file = name_file.replace('.', '-')
        name_files[i] = name_file

    if full_mode is False:
        name_files = name_file

    return name_files


def generate_file_name(config, f1=None, epochs=None):
    embeddings_name = config['embeddings_file']
    time = get_actual_time()
    if epochs is None:
        epochs = config['epoch_count']
    binary = config['binary']
    batch_size = config['batch_size']
    model_name = Path(config['model_name']).name
    if config['eval'] is True:
        model_name = model_name[:30]

    dataset_name = config['dataset_name']
    if dataset_name == 'combined':
        tmp = '-'.join(config['combined_datasets'])
        dataset_name = dataset_name + '-' + tmp

    name_file = model_name + "_" \
                + dataset_name \
                + "_BS-" + str(batch_size) \
                + "_EC-" + str(epochs) \
                + "_LEN-" + str(config['max_seq_len']) \
                + "_LR-%.7f" % (config['lr']) \
                + "_SCH-" + str(config['lr_scheduler_name']) \
                + "_WR-" + str(config['warm_up_steps']) \
                + "_Opt-" + config['optimizer'] \
                + "_CPU-" + str(config['use_cpu']) \
                + "_TRAIN-" + str(config['use_only_train_data']) \
                + "_BIN-" + str(binary) \
                + "_WD-%.5f" % config['weight_decay'] \
                + "_TWE-" + str(config['trainable_word_embeddings']) \
                + "_MW-" + str(config['max_words']) \
                + "_DR-" + str(config['dropout_rnn']) \
                + "_DF-" + str(config['dropout_final'])


    name_file += "-" + embeddings_name + "_EDim" + str(config['embeddings_size'])
    name_file += "_" + time
    name_file = name_file.replace('.', '-')
    if f1 is not None:
        name_file = name_file + "_F1-%.4f" % (f1)

    return name_file


def build_param_string(param_dict):
    string_param = 'Model parameters:\n'
    string_param += '--------------------\n'
    for i, v in param_dict.items():
        if i == 'we_matrix':
            continue
        string_param += str(i) + ": " + str(v) + '\n'

    return string_param

def save_model_transformer(model, tokenizer, config, save_dir, train_test_time,
                           accuracy, macro_f1, precision, recall,
                           result_string, history, curr_epoch, tensorboard_log_dir=None):
    model_name = Path(config['model_name']).name
    name_folder = os.path.join(save_dir, model_name)
    if os.path.exists(name_folder) is not True:
        os.makedirs(name_folder)

        # generate file name
    # name_file = generate_file_name_transformer(config, f1=macro_f1)
    if config['full_mode'] is True:
        name_file = config['config_name'][curr_epoch]
    else:
        name_file = config['config_name']
    name_file = append_f1_to_file_name(name_file, macro_f1)
    print("File name:", name_file)

    # visaulize_training(history, os.path.join(name_folder, name_file + '.png'))

    if tensorboard_log_dir is not None:
        # copy tensorboard log file to new directory
        copy_files(tensorboard_log_dir, name_folder)

    # This is config from command line
    # dump used config
    with open(os.path.join(name_folder, name_file + ".config"), 'w') as outfile:
        json.dump(config, outfile, indent=4)

        # dump results
    print_str = get_stats_string(accuracy, macro_f1, precision, recall)

    with open(os.path.join(name_folder, name_file + ".txt"), 'a') as f:
        f.write(print_str)
        f.write('\n')
        # print results in excel format
        f.write(result_string + '\n')
        f.write('\n')
        f.write("train and test time: " + "{0:.2f}s".format(train_test_time) + '\n')
        f.write("Used Parameters:")
        f.write(build_param_string(config))

    try:
        # save trained model, inspired by http://mccormickml.com/2019/07/22/BERT-fine-tuning/
        if model is not None:
            model_save_dir = os.path.join(name_folder, name_file)

            if os.path.exists(model_save_dir) is not True:
                os.makedirs(model_save_dir)

            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(model_save_dir)
            tokenizer.save_pretrained(model_save_dir)

            print("Model saved...")

    except Exception:
        logger.error("Failed to save model", exc_info=True)

    return name_file


#
def save_model(model, config, embeddings_name, tensorboard_log_dir, train_test_time,
               accuracy, macro_f1, precision, recall, callback_list, save_dir,
               param_dict, result_string):
    name_folder = os.path.join(save_dir, config['model_name'])

    if os.path.exists(name_folder) is not True:
        os.makedirs(name_folder)

    epochs = None
    if config['use_early_stopping'] is True:
        es = callback_list[1]
        epochs = es.stopped_epoch

    # generate file name
    name_file = generate_file_name(config, f1=macro_f1, epochs=epochs)
    print("File name:", name_file)

    # copy tensorboard log file to new directory
    copy_files(tensorboard_log_dir, name_folder)

    # save trained model
    model_file_name = name_file + ".h5"
    model.save(os.path.join(name_folder, model_file_name))

    # assing file name
    config['model_file_name'] = model_file_name

    # This is config from command line
    # dump used config
    with open(os.path.join(name_folder, name_file + ".config"), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # This is final config used for model
    # dump used config
    with open(os.path.join(name_folder, name_file + "-real.config"), 'w') as outfile:
        json.dump(param_dict, outfile, indent=4)

    # dump model summary
    with open(os.path.join(name_folder, name_file + ".txt"), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    # dump results
    print_str = get_stats_string(accuracy, macro_f1, precision, recall)

    with open(os.path.join(name_folder, name_file + ".txt"), 'a') as f:
        f.write(print_str)
        f.write('\n')
        # print results in excel format
        f.write(result_string + '\n')
        f.write('\n')
        f.write("train and test time: " + "{0:.2f}s".format(train_test_time) + '\n')
        f.write("Used Parameters:")
        f.write(build_param_string(param_dict))

    # assing file name
    config['model_file_name'] = 'NONE'

    print("Model saved...")
    return name_file


def copy_files(source_folder_path, dst_folder, override=True):
    # source folder with the current log
    source_folder_path = os.path.abspath(source_folder_path)

    # name of folder with logs
    log_file_dir = os.path.basename(os.path.normpath(source_folder_path))

    # list of files in source folder
    files = os.listdir(source_folder_path)

    dst_folder = os.path.join(dst_folder, log_file_dir)

    if os.path.exists(dst_folder) is not True:
        os.mkdir(dst_folder)

    copy_tree(source_folder_path, dst_folder)

    # for file in files:
    #     tmp_src = os.path.join(source_folder_path,file)
    #     tmp_dst = os.path.join(dst_folder,file)
    #     tmp_dst = os.path.splitext(tmp_dst)[0]
    #     tmp_dst = tmp_dst + '.evt'
    #
    #     # if file exists but should not be override, continue
    #     if os.path.exists(tmp_dst) and override is False:
    #         continue
    #
    #     # if file exists and should be override remove it
    #     elif os.path.exists(tmp_dst) and override is True:
    #         os.remove(tmp_dst)
    #     # pokud doje k chybe je to tim ze delka cesty ve windows muze byt maximalne 259/269 znaku
    #     # gpedit.msc
    #     # https://www.tenforums.com/tutorials/79976-open-local-group-policy-editor-windows-10-a.html
    #     # https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing
    #     # zapnout long file mode
    #     shutil.copy(tmp_src,tmp_dst)


def save_data_pickle(data, path):
    # save data to a file
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=4)
    # print("Data successfully saved")


# loading binary data from the given path
def load_data_pickle(path):
    try:
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
    except:
        print('Error reading data')
        print('One more attemp')
        with open(path, 'rb') as fp:
            data = pickle.load(fp)

    return data


def get_stats_string(accuracy, macro_f1, precision, recall):
    string = '----Average----' + '\n' \
             + 'accuracy: %2.4f ' % (accuracy) + '\n' \
             + 'f1 macro score: %2.4f ' % (macro_f1) + '\n' \
             + 'precision: %2.4f ' % (precision) + '\n' \
             + 'recall: %2.4f ' % (recall) + '\n'

    return string


def get_tokenizer(tokenizer_name, lang='czech'):
    if tokenizer_name == 'white_space':
        tokenizer = WhitespaceTokenizer()

    elif tokenizer_name == 'toktok':
        tokenizer = ToktokTokenizer()

    elif tokenizer_name == 'word_tokenizer':
        tokenizer = WordTokenizerMock(lang)
    else:
        raise Exception('Not valid tokenizer:' + tokenizer_name)

    return tokenizer



class WordTokenizerMock:
    def __init__(self, language):
        self.language = language

    def tokenize(self, text):
        words = [x for x in word_tokenize(text, language=self.language, preserve_line=True) if len(x) >= 1]
        return words

def compute_number_of_steps(data_size, batch_size, epochs):
    steps_per_epoch = data_size / batch_size
    steps_per_epoch = math.ceil(steps_per_epoch)
    steps_total = steps_per_epoch * epochs
    return steps_total


def get_learning_rate(initial_lr, lr_scheduler_name, expected_steps, minimum_lr=0.0000001, warmup_steps=0):
    if lr_scheduler_name == 'none' or lr_scheduler_name is None:
        return initial_lr
    else:
        scheduler = None
        if lr_scheduler_name == 'exp':
            scheduler = ExponentialDecay(initial_lr, expected_steps, decay_rate=0.05)
        elif lr_scheduler_name == 'poly':
            scheduler = PolynomialDecay(initial_lr, decay_steps=expected_steps, end_learning_rate=minimum_lr)
        elif lr_scheduler_name == 'cosine':
            scheduler = CosineDecay(initial_lr, expected_steps)
        elif lr_scheduler_name == 'step':
            raise NotImplementedError("Step Decay not implemented")
            # scheduler = StepDecay(initial_lr, 0.5, 1250)
        else:
            raise Exception("Unknown learning rate scheduler name:" + str(lr_scheduler_name))

        if (scheduler is not None) and warmup_steps > 0:
            if warmup_steps == 1:
                raise Exception("Warmup steps cannot be 1")
            if warmup_steps < 1:
                warmup_steps = warmup_steps * expected_steps
                warmup_steps = math.ceil(warmup_steps)
            scheduler = WarmUpDecay(initial_lr, scheduler, warmup_steps)

        return scheduler



def get_optimizer(optimizer_name, lr, weight_decay=0.0):
    """
    # Learning rate decay, can be implemented with tf.keras.optimizers.schedules.LearningRateSchedule


    :param optimizer_name:
    :param lr:  floating point value, or a schedule that is a
                `tf.keras.optimizers.schedules.LearningRateSchedule`,
    :return:
    """

    if optimizer_name == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr, amsgrad=False)

    elif optimizer_name == 'AdamW':
        optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)

    elif optimizer_name == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=lr)

    elif optimizer_name == 'Adagrad':
        optimizer = keras.optimizers.Adagrad(learning_rate=lr)

    elif optimizer_name == 'Adadelta':
        optimizer = keras.optimizers.Adadelta(learning_rate=lr)

    elif optimizer_name == 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=lr)
    else:
        raise Exception('Not valid optimizer:' + optimizer_name)

    return optimizer
