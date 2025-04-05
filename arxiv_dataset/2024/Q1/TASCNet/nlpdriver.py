## Driver command
# python nlpdriver.py --model-name vgg16 --data-path datasets/movie_reviews_training_data.pkl

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import warnings
import logging
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import callbacks, models, optimizers

from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity
from utils.utils import *
from utils import wrapper

warnings.filterwarnings("ignore")

# set seed
SEED = 0
tf.random.set_random_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model-name",
    type=str,
    required=True,
    default="vgg16",
    help="label for the model to create the results folder",
)
parser.add_argument(
    "--model-path",
    type=str,
    default="models/weights_cnn_moviereviews_sentiment_autotune_continued-reduced-0.99-1000epochs.h5",
    help="path to the model's .h5 file",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    default="CalTech101",
    help="label for the dataset to create the results folder"
)
parser.add_argument(
    "--data-path", type=str, default="datasets/movie_reviews_training_data.pkl"
)
parser.add_argument("--weighted-avg", action="store_true")
parser.add_argument("--train-epochs", type=int, default=50)
parser.add_argument("--optim-epochs", type=int, default=50)
parser.add_argument("--prune-per", type=int, default=5)
parser.add_argument("--model-type", type=str, default="", required=False, help="pass `resnet` to make sure equal number of filters to be pruned in each iteration")

args = parser.parse_args()

BATCH_SIZE = 32
DATA_SET = args.dataset

TRAINING_EPOCHS = args.train_epochs
OPTIMIZATION_EPOCHS = args.optim_epochs
PRUNING_PERCENTAGE = args.prune_per

with open(args.data_path, mode="rb") as f:
    data = pickle.load(f)

train_data = data["training"]
val_data = data["validation"]
class_weights = dict(Counter(train_data[1]))

MODEL = args.model_name
weighted_avg_str = "_weighted_avg" if args.weighted_avg else ""

MODEL_FOLDER = os.path.join(
    "Results",
    f"Pruned_Models_{MODEL}_{DATA_SET}{weighted_avg_str}_{args.prune_per}per",
)
os.makedirs(MODEL_FOLDER, exist_ok=True)
LOGFILE = os.path.join(
    MODEL_FOLDER, f"Prune_{MODEL}_{DATA_SET}{weighted_avg_str}_{args.prune_per}per.csv"
)

os.system(f"rm -r {MODEL_FOLDER}/*")

logging.debug(f"\n\nPruning {MODEL} with {DATA_SET}\n\n")


logging.debug("==========================================")


class InitModel:
    def __init__(self, epochs, train=True):
        self.epochs = epochs
        self.history = 0
        self.weight_list_per_epoch = None
        self.model = self.build_model()
        if train:
            self.model, self.history, self.weight_list_per_epoch = self.train(
                self.model
            )

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
        model = models.load_model(args.model_path)
        return model

    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)
        return X_train, X_test

    def train(self, model):
        # training parameters
        learning_rate = 0.001

        # optimization details
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizers.Adam(lr=0.1),
            metrics=["accuracy"],
        )

        gw = Get_Weights()
        model.summary()
        # training process in a for loop with learning rate drop every 25 epoches.
        logging.debug("\n\nTraining on CalTech: 1st time\n\n")
        lr_reducer = ReduceLROnPlateau(
            monitor="val_acc",
            factor=0.99,
            cooldown=0,
            patience=1,
            min_lr=0.5e-15,
            verbose=1,
        )
        history = model.fit(
            *train_data,
            validation_data=val_data,
            epochs=self.epochs,
            callbacks=[lr_reducer, gw],
            class_weight=class_weights,
            batch_size=BATCH_SIZE,
        )

        return model, history, gw.weight_list


def train(model, epochs):
    """
    Arguments:
        model:model to be trained
        epochs:number of epochs to be trained
    Return:
        model:trained/fine-tuned Model,
        history: accuracies and losses (keras history)
        weight_list_per_epoch = all weight tensors per epochs in a list
    """
    # training parameters

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(lr=0.1),
        metrics=["accuracy"],
    )

    gw = Get_Weights()
    model.summary()

    # training process in a for loop with learning rate drop every 25 epoches.
    lr_reducer = ReduceLROnPlateau(
        monitor="val_acc",
        factor=0.99,
        cooldown=0,
        patience=1,
        min_lr=0.5e-15,
        verbose=1,
    )
    history = model.fit(
        *train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[lr_reducer, gw],
        class_weight=class_weights,
        batch_size=BATCH_SIZE,
    )

    return model, history, gw.weight_list


def optimize(model, weight_list_per_epoch, epochs, percentage):
    def lr_scheduler(epoch):
        return learning_rate * (0.5 ** (epoch // lr_drop))

    reduce_lr = callbacks.LearningRateScheduler(lr_scheduler)
    gw = Get_Weights()

    # get filter pairs
    _, flattened_filters_per_epoch = my_get_l1_norms_filters_per_epoch(
        weight_list_per_epoch
    )
    distance_matrix_list = my_get_distance_matrix_list(flattened_filters_per_epoch)
    episodes_for_all_layers = my_get_episodes_for_all_layers(
        distance_matrix_list, percentage
    )
    all_conv_layers = my_get_all_conv_layers(model)

    filter_pos_episodes = dict(zip(all_conv_layers, episodes_for_all_layers))

    # wrap model
    wrapped = wrapper.OptimizerWrapper(model, filter_pos_episodes=filter_pos_episodes)

    # train model
    wrapped.compile(
        optimizer=optimizers.Adam(0.1), loss="binary_crossentropy", metrics=["accuracy"]
    )

    lr_reducer = ReduceLROnPlateau(
        monitor="val_acc",
        factor=0.99,
        cooldown=0,
        patience=1,
        min_lr=0.5e-15,
        verbose=1,
    )
    history = wrapped.fit(
        *train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[lr_reducer, gw],
        class_weight=class_weights,
        batch_size=BATCH_SIZE,
    )

    return wrapped.model, history


# this dictionary is to log the parameters and is later converted into a dataframe.
log_dict = dict()
log_dict["train_loss"] = []
log_dict["train_acc"] = []
log_dict["val_loss"] = []
log_dict["val_acc"] = []
log_dict["total_params"] = []
log_dict["total_params_trainable"] = []
log_dict["total_flops"] = []
log_dict["total_flops_trainable"] = []


my_vgg = InitModel(epochs=TRAINING_EPOCHS, train=True)
model, history, weight_list_per_epoch = (
    my_vgg.model,
    my_vgg.history,
    my_vgg.weight_list_per_epoch,
)
learning_rate = 0.001
lr_decay = 1e-6
lr_drop = 20


def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


reduce_lr = callbacks.LearningRateScheduler(lr_scheduler)

logging.debug(history.history.keys())

try:
    best_acc_index = history.history["val_accuracy"].index(
        max(history.history["val_accuracy"])
    )
    log_dict["train_loss"].append(history.history["loss"][best_acc_index])
    log_dict["train_acc"].append(history.history["accuracy"][best_acc_index])
    log_dict["val_loss"].append(history.history["val_loss"][best_acc_index])
    log_dict["val_acc"].append(history.history["val_accuracy"][best_acc_index])
    validation_accuracy = max(history.history["val_accuracy"])
except:
    best_acc_index = history.history["val_acc"].index(max(history.history["val_acc"]))
    log_dict["train_loss"].append(history.history["loss"][best_acc_index])
    log_dict["train_acc"].append(history.history["acc"][best_acc_index])
    log_dict["val_loss"].append(history.history["val_loss"][best_acc_index])
    log_dict["val_acc"].append(history.history["val_acc"][best_acc_index])
    validation_accuracy = max(history.history["val_acc"])

a, b = count_model_params_flops(model)
c, d = count_model_params_flops(model, layers=my_get_all_conv_layers(model))
log_dict["total_params"].append(a)
log_dict["total_flops"].append(b)
log_dict["total_params_trainable"].append(c)
log_dict["total_flops_trainable"].append(d)

log_df = pd.DataFrame(log_dict)
log_df.to_csv(LOGFILE)

# stop pruning if the accuracy drops by 5% from maximum accuracy ever obtained.
logging.debug("Initial Validation Accuracy = {}".format(validation_accuracy))
max_val_acc = validation_accuracy
count = 0

while (validation_accuracy - max_val_acc >= -0.02) and my_get_all_conv_layers(model):
    logging.debug("ITERATION {} ".format(count + 1))
    if max_val_acc < validation_accuracy:
        max_val_acc = validation_accuracy

    if count < 1:
        logging.debug("OPTIMIZATION")
        model, _ = optimize(
            model, weight_list_per_epoch, OPTIMIZATION_EPOCHS, PRUNING_PERCENTAGE
        )
        model = my_delete_filters(
            model,
            weight_list_per_epoch,
            PRUNING_PERCENTAGE,
            weighted_avg=args.weighted_avg,
            model_type=args.model_type,
        )
        logging.debug("FINETUNING")
        model, history, weight_list_per_epoch = train(model, TRAINING_EPOCHS)
        logging.debug("\n\n after training", len(weight_list_per_epoch))

    elif count <= 3:
        logging.debug("OPTIMIZATION")
        model, _ = optimize(
            model, weight_list_per_epoch, OPTIMIZATION_EPOCHS, PRUNING_PERCENTAGE
        )
        model = my_delete_filters(
            model,
            weight_list_per_epoch,
            PRUNING_PERCENTAGE,
            weighted_avg=args.weighted_avg,
            model_type=args.model_type,
        )
        logging.debug("FINETUNING")
        model, history, weight_list_per_epoch = train(model, TRAINING_EPOCHS)
        logging.debug("\n\n after training", len(weight_list_per_epoch))
    else:
        logging.debug("OPTIMIZATION")
        model, _ = optimize(
            model, weight_list_per_epoch, OPTIMIZATION_EPOCHS, PRUNING_PERCENTAGE
        )
        model = my_delete_filters(
            model,
            weight_list_per_epoch,
            PRUNING_PERCENTAGE,
            weighted_avg=args.weighted_avg,
            model_type=args.model_type,
        )
        logging.debug("FINETUNING")
        model, history, weight_list_per_epoch = train(model, TRAINING_EPOCHS)

    PRUNED_MODEL_FILE = f"{MODEL_FOLDER}/{log_df.shape[0]}_{MODEL}_{DATA_SET}.h5"
    model.save(PRUNED_MODEL_FILE)

    a, b = count_model_params_flops(model)
    c, d = count_model_params_flops(model, layers=my_get_all_conv_layers(model))

    try:
        validation_accuracy = max(history.history["val_acc"])
        best_acc_index = history.history["val_acc"].index(
            max(history.history["val_acc"])
        )
        log_dict["train_loss"].append(history.history["loss"][best_acc_index])
        log_dict["train_acc"].append(history.history["acc"][best_acc_index])
        log_dict["val_loss"].append(history.history["val_loss"][best_acc_index])
        log_dict["val_acc"].append(history.history["val_acc"][best_acc_index])
    except:
        validation_accuracy = max(history.history["val_accuracy"])
        best_acc_index = history.history["val_accuracy"].index(
            max(history.history["val_accuracy"])
        )
        log_dict["train_loss"].append(history.history["loss"][best_acc_index])
        log_dict["train_acc"].append(history.history["accuracy"][best_acc_index])
        log_dict["val_loss"].append(history.history["val_loss"][best_acc_index])
        log_dict["val_acc"].append(history.history["val_accuracy"][best_acc_index])

    log_dict["total_params"].append(a)
    log_dict["total_flops"].append(b)
    log_dict["total_params_trainable"].append(c)
    log_dict["total_flops_trainable"].append(d)

    log_df = pd.DataFrame(log_dict)
    log_df.to_csv(LOGFILE)
    logging.debug(
        "VALIDATION ACCURACY AFTER {} ITERATIONS = {}".format(
            count + 1, validation_accuracy
        )
    )
    count += 1

log_df = pd.DataFrame(log_dict)
log_df.to_csv(LOGFILE)
