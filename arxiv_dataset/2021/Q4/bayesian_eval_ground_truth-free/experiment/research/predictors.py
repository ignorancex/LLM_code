"""Trains the predictors on different formats of the data, in kfold CV.

Saves results in the given output directory as follows:
    output_dir/dataset_id/model_id/random_seed/datetime/#_fold_cv/eval_fold_#/

    datetime directory is optional, and used for indicating the order of runs.
    '#' is used to indicate the variable number in its respective spot, so
    the number of K folds in '#_fold_cv/' and the focus fold number in
    'eval_fold_#'.

    And within can be the 'checkpoints/', 'pred.csv', 'summary.json', and model
    weights as HDF5.

    'checkpoints/' may contain weights.epoch.hdf5 or pred.epoch.hdf5
"""
from copy import deepcopy
import csv
from datetime import datetime
import json
import logging
import os
import random
from time import perf_counter, process_time

import h5py
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold, StratifiedKFold

from psych_metric import datasets
#from psych_metric.datasets import data_handler
#from experiment.crowd_layer import .... crowd layer classes etc.

import experiment.io


class CheckpointValidaitonOutput(keras.callbacks.Callback):
    """Saves the validaiton output and target pairs to a csv file for every
    period.

    Attributes
    ----------
    filepath : str
    period : int
        Interval (number of epochs) between checkpoints.
    delimiter : str
        The character to use as the delmiter of the csv files.
    """
    def __init__(self, filepath, period=1, delimiter=','):
        super(CheckpointValidaitonOutput, self).__init__()
        self.filepath = filepath
        self.period = period
        self.delimiter = delimiter

        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        """Save the targets and model outputs of the most recent epoch."""
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0

            with h5py.File(f'{self.filepath}.{epoch + 1:02d}.hdf5', 'w') as h5f:
                h5f['pred'] = self.validation_data[0]

            """
            with open(f'{self.filepath}.{epoch:02d}.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=self.delimiter)
                #writer.writerow(['targets', 'outputs'])
                writer.writerow(['outputs'])

                for i in range(len(self.validation_data[1])):
                    writer.writerow([self.validation_data[0]])
            """


def load_dataset(dataset, *args, **kwargs):
    """Temporary mask of dataset.data_handler.load_dataset() that only handles
    LabelMe, FacialBeauty, and then FirstImpressions and ChaLearn.
    """
    if dataset.lower() == 'facial_beauty' or dataset.lower() == 'facialbeauty':
        return datasets.FacialBeauty(dataset, *args, **kwargs)

    # First Impressions
    #elif dataset_exists(dataset, 'first_impressions'):
    #    return datasets.FirtImpressions(dataset, *args, **kwargs)

    # CrowdLayer
    elif dataset.lower() == 'labelme':
        return datasets.CrowdLayer(dataset, *args, **kwargs)
    else:
        raise NotImplementedError(' '.join([
            'This is a tmp mask of dataset.data_handler.load_dataset as not',
            'all the datasets are deemed necessary for the code using this',
            'script.',
        ]))


def load_prep_data(dataset_id, data_config, label_src, parts='labelme'):
    """Load and prep dataset."""
    # TODO this should probably all be handled mostly in the dataset class itself. Specifically the label encoding and binarization.

    #dataset = data_handler.load_dataset(dataset_id, **data_config)
    dataset = load_dataset(dataset_id, **data_config)
    if dataset_id == 'LabelMe' and parts == 'labelme':
        images, labels = dataset.load_images(os.path.join(
            dataset.data_dir,
            dataset.dataset,
            'labelme_vgg16_encoded.h5'
        ))
    else:
        images, labels = dataset.load_images(
            majority_vote=True,
            img_shape=(224, 224),
        )

    # select the label source for this run
    if label_src == 'annotations':
        labels = dataset.df

        # TODO handle proper binarizing of annotations labels.
        #raise NotImplementedError
        return images, labels
    elif label_src == 'frequency':
        # TODO perhaps should be handled in dataset class load_images function.
        # TODO -1 is hardcoded expected Missing value for annotator. Good for only LabelMe
        # TODO avoid the usage of pd.SparseDataFrame as it contains critical bugs in latest pandas releases (its depracted)
        if dataset_id == 'LabelMe':
            labels = pd.DataFrame(dataset.df.values).replace(-1, np.NaN).dropna(
                'columns',
                'all',
            ).apply(
                lambda x: x.value_counts(True),
                axis=1,
            ).fillna(0)
        else:
            labels = pd.DataFrame(dataset.df.values).apply(
                lambda x: x.value_counts(True),
                axis=1,
            ).fillna(0)


        return images, labels.values, labels.columns.tolist()

    elif label_src == 'majority_vote' or label_src == 'ground_truth':
        if isinstance(labels, pd.SparseDataFrame):
            # NOTE assumes labels are ints and w/in [0,255]
            labels = labels[label_src].values.values.astype('uint8')
        elif isinstance(labels, pd.SparseSeries):
            # NOTE assumes labels are ints and w/in [0,255]
            labels = labels.values.values.astype('uint8')
        elif isinstance(labels, pd.DataFrame):
            labels = labels[label_src]

        # Binarize the label data
        label_bin = LabelBinarizer()
        label_bin.fit(labels)
        y_data = label_bin.transform(labels).astype('float32', copy=False)

        return images, y_data, label_bin.classes_.tolist()
    else:
        raise ValueError(
            'expected `label_src` to be "majority_vote", "ground_truth", or '
            + f'"annotations", but recieved {label_src}',
        )


def run_experiment(
    output_dir,
    label_src,
    dataset_id,
    data_config,
    model_config,
    kfold_cv_args,
    focus_fold=None,
    random_seeds=None,
):
    """Runs the kfold cross validation  experiment with same model and data,
    just different seeds.

    Parameters
    ----------
    output_dir : str
        The output directory to save the results of the model.
    label_src : str
        The source of the labels to train the supervised models.
    dataset_id : str
        The dataset identifier of the dataset to load.
    data_config : dict
        The arguments related to the datasets to load and use..
    model_config : dict
        The arguments related to the model initialization, training, and testing.
    kfold_cv_args : dict
        The arguments related to the kfold cross validation.
    focus_fold : int, optional
        Specifies the fold of the data to focus for running a single train-eval
        session of the model on the data. If given, K fold cross validation
        does not execute, only that single data partition is run. Defaults to
        None.
    random_seeds : int | list of ints, optional
        The random seed(s) used to initialize the random number generator for
        the k fold cross validation data splitting and for the intial
        initialization of the models for each training session.
    """
    if label_src == 'majority_vote' or label_src == 'ground_truth' or label_src == 'frequency':
        images, labels, bin_classes = load_prep_data(
            dataset_id,
            data_config,
            label_src,
            model_config['parts'],
        )
    else:
        # NOTE annotatoins not implemented yet.
        raise NotImplementedError("annotations label src not implemented.")
        images, labels = load_prep_data(
            dataset_id,
            data_config,
            label_src,
            model_config['parts'],
        )
        bin_classes = labels.columns.tolist()

    summary = {
        dataset_id: data_config,
        'model_config': model_config,
        'kfold_cv_args': kfold_cv_args,
        'label_binarizer': bin_classes
    }

    if isinstance(random_seeds, int):
        random_seeds = [random_seeds]

    if random_seeds:
        output_dir = os.path.join(
            output_dir,
            dataset_id,
            model_config['model_id'],
        )

        for i, r in enumerate(random_seeds):
            kfold_cv_args.pop('random_seed', None)

            logging.info(f'{i + 1}/{len(random_seeds)} Random Seeds: {r}')

            r_output_dir = os.path.join(
                output_dir,
                str(r),
                datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
            )

            kfold_cv(
                model_config,
                images,
                labels,
                r_output_dir,
                summary=summary,
                random_seed=r,
                **kfold_cv_args,
            )
    else:
        # create identifying directory path for saving results.
        output_dir = os.path.join(
            output_dir,
            dataset_id,
            model_config['model_id'],
            str(kfold_cv_args['random_seed']),
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        )

        # TODO decide how to handle output_dir

        if focus_fold:
            # TODO run single train, test, maybe put in kfold_cv? idk.
            # TODO add focus fold implementation
            raise NotImplementedError('`focus_fold` for loading and testing a specific fold given `k` folds and data is not yet implemented.')
        else:
            kfold_cv(
                model_config,
                images,
                labels,
                output_dir,
                summary=summary,
                **kfold_cv_args,
            )


def kfold_cv(
    model_config,
    features,
    labels,
    output_dir,
    summary,
    kfolds=5,
    random_seed=None,
    save_pred=True,
    save_model=True,
    stratified=None,
    train_focus_fold=False,
    shuffle=True,
    repeat=None,
    period=0,
    period_save_pred=False,
    #focus_fold=None,
    early_stopping=None,
):
    """Performs kfold cross validation on the model and saves the results.

    Parameters
    ----------
    model :
        custom class or something needs: fit/train, eval, save
    data :
        The data to be split into folds and used in cross validation.
    k : int, optional
        The number of folds.
    random_seed : int, optional
        The seed used for initializing the random number generators.
    save_pred : bool, optional
        returns the predictions in addition to the rest of the output.
    save_model : bool, optional
        returns the model in addition to the rest of the output.
    stratified : bool | array-like, optional
        By default the data splitting is unstratified (default is None). If
        True, the data is stratified when split to preserve class balance of
        original dataset. If a 1-D array-like object of same length as data,
        then it is treated as the strata of the data for splitting.
    train_focus_fold : bool, optional
        If False (default), the single focus fold will be the current
        iteration's test set while the rest are used for training the model,
        otherwise the focus fold is the only fold used for training and the
        rest are used for testing.
    shuffle: bool, optional
        Whether or not to shuffle the data prior to splitting for KFold CV.
    repeat: int, optional
        Number of times to repeat the K fold CV.
    period: int, optional
        Number of epochs to save the model checkpoints. Defaults to 0, meaning
        no checkpoints will be saved. The checkpoints save model weights only.
    period_save_pred : bool, optional
        If True and model checkpoint exists, then save the validation input and
        output.
    """
    if not random_seed:
        random_seed = random.randint(0, 2**32 - 1)

        # TODO use same seed for initializing the model everytime or different seeds?
        raise NotImplementedError
        # shuffle param indicates if they want data shuffling or not, ie. seed to be made or not for shuffling only.

    # Create the kfolds directory for this experiment.
    output_dir_kfolds = os.path.join(output_dir, f'{kfolds}_fold_cv')
    os.makedirs(output_dir_kfolds, exist_ok=True)

    # Data index splitting
    if stratified:
        fold_indices = StratifiedKFold(kfolds, shuffle, random_seed).split(features, labels)
    else:
        fold_indices = KFold(kfolds, shuffle, random_seed).split(features)

    for i, (other_folds, focus_fold) in enumerate(fold_indices):
        if random_seed:
            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)

        kfold_summary = deepcopy(summary)
        kfold_summary['kfold_cv_args'].update({
            'random_seed': random_seed,
            'focus_fold': i + 1,
        })

        output_dir_eval_fold = os.path.join(output_dir_kfolds, f'eval_fold_{i+1}')
        os.makedirs(output_dir_eval_fold, exist_ok=True)

        # Set the correct train and test indices
        if train_focus_fold:
            train_idx = focus_fold
            test_idx = other_folds
        else:
            train_idx = other_folds
            test_idx = focus_fold

        # Create callbacks for the model.
        if period > 0:
            checkpoint_dir = os.path.join(output_dir_eval_fold, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

            callbacks = [keras.callbacks.ModelCheckpoint(
                os.path.join(checkpoint_dir, 'weights.{epoch:02d}.hdf5'),
                save_weights_only=True,
                period=period,
            )]

            if period_save_pred:
                callbacks.append(CheckpointValidaitonOutput(
                    os.path.join(checkpoint_dir, 'pred'),
                    period=period,
                ))
        elif early_stopping:
            callbacks = [keras.callbacks.EarlyStopping(**early_stopping)]
        else:
            callbacks = None

        logging.info(f'{i + 1}/{kfolds} fold cross validation: Training')

        if period <= 0 and early_stopping:
            model, init_times, train_times = prepare_model(
                model_config,
                features[train_idx],
                labels[train_idx],
                output_dir_eval_fold,
                callbacks=callbacks,
                validation_data=(features[test_idx], labels[test_idx]),
            )

            kfold_summary['early_stopped_epoch'] = callbacks[-1].stopped_epoch
            logging.info('Early stopping at epoch: %d', callbacks[-1].stopped_epoch)

        elif period_save_pred:
            # TODO properly handle filling in Y True for validation data.
            model, init_times, train_times = prepare_model(
                model_config,
                features[train_idx],
                labels[train_idx],
                output_dir_eval_fold,
                callbacks=callbacks,
                validation_data=(features[test_idx], np.empty((len(test_idx), len(labels[0])))),
            )
        else:
            model, init_times, train_times = prepare_model(
                model_config,
                features[train_idx],
                labels[train_idx],
                output_dir_eval_fold,
                callbacks=callbacks,
            )

        logging.info(f'{i + 1}/{kfolds} fold cross validation: Testing')

        start_perf_time = perf_counter()
        start_process_time = process_time()

        if 'test' in model_config:
            pred = model.predict(features[test_idx], **model_config['test'])
        else:
            pred = model.predict(features[test_idx])

        test_process = process_time() - start_process_time
        test_perf = perf_counter() - start_perf_time

        if save_pred:
            np.savetxt(
                os.path.join(output_dir_eval_fold, 'pred.csv'),
                pred,
                delimiter=',',
            )

        kfold_summary['runtimes'] = {
            'init': init_times,
            'train': train_times,
            'test': {
                'process': test_process,
                'perf': test_perf,
            },
        }

        # save summary
        experiment.io.save_json(
            os.path.join(output_dir_eval_fold, 'summary.json'),
            kfold_summary,
        )


def prepare_model(
    model_config,
    features,
    labels,
    output_dir=None,
    callbacks=None,
    validation_data=None,
):
    """Prepares the model by initializing it and training it.

    Parameters
    ----------
    model_config : dict
        The arguments related to the model initialization, training, and testing.
    features : array-like
        The feature data used as input to the model for training.
    labels : array-like
        The labels associated with the samples in features for training.
    output_dir : str
        The output directory to save the results of the model.
    callbacks : list
        Callbacks to be used by Keras when training the model.
    validation_data : , optional


    Returns
    -------
    tuple(keras.models.Model, float, float)
        The trained keras Model and the runtimes for initializing and training
        that model.
    """
    if 'filepath' in model_config:
        # TODO could remove from here and put into calling code when loading is possible
        return keras.models.load_model(model_config['filepath']), None, None

    start_perf_time = perf_counter()
    start_process_time = process_time()

    model = load_model(
        model_config['model_id'],
        parts=model_config['parts'],
        **model_config['init'],
    )

    init_process = process_time() - start_process_time
    init_perf = perf_counter() - start_perf_time

    # TODO callbacks exist: make one that keeps track of runtimes for checkpoint models.

    start_perf_time = perf_counter()
    start_process_time = process_time()

    if 'train' in model_config:
        if validation_data:
            # NOTE validaiton_data will obviously add prediction time to training. Use checkpoints to handle appropriate time training.
            model.fit(
                features,
                labels,
                callbacks=callbacks,
                validation_data=validation_data,
                **model_config['train'],
            )
        else:
            model.fit(
                features,
                labels,
                callbacks=callbacks,
                **model_config['train'],
            )
    else:
        if validation_data:
            model.fit(
                features,
                labels,
                validation_data=validation_data,
                callbacks=callbacks,
            )
        else:
            model.fit(features, labels, callbacks=callbacks)

    train_process = process_time() - start_process_time
    train_perf = perf_counter() - start_perf_time

    if isinstance(output_dir, str) and os.path.isdir(output_dir):
        model.save(os.path.join(output_dir, f'{model_config["model_id"]}.h5'))

    init_times = {'process': init_process, 'perf': init_perf}
    train_times = {'process': train_process, 'perf': train_perf}

    return model, init_times, train_times


def load_model(
    model_id,
    crowd_layer=False,
    parts='labelme',
    weights_file=False,
    kl_div=False,
    **kwargs,
):
    """Either initializes the model or loads the model from file.

    Parameters
    ----------
    model_id : str
        The model identifier of the model to load.
    crowd_layer : bool
        Uses crowd layer in model if True, standard model output otherwise.
    parts : str
        indicates the part of the model to be loaded, either the DNN only, the
        classifier at the end only, or the full model. Useful for preprocessing
        and training of frozen DNN encoded samples.
    weights_file : str
        Filepath to the model weights to be loaded after creating the model.
    kl_div : bool
        Whether to use KL divergence as loss or not.
    kwargs : dict
        The remaining key word arguements to use in loading the model.

    Returns
    -------
    The model either initialized or  loaded from file.
    """

    # TODO add loading of model and model weights from file for each.

    if model_id.lower() == 'vgg16':
        model = vgg16_model(crowd_layer=crowd_layer, parts=parts, **kwargs)

        if crowd_layer:
            # TODO model.compile('adam', CrowdLayer...)
            raise NotImplementedError
        elif kl_div:
            print("\nKL DIV!\n")
            model.compile('adam', 'kullback_leibler_divergence')
        else:
            model.compile('adam', 'categorical_crossentropy')
    if model_id.lower() == 'resnext50':
        model = resnext50_model(crowd_layer=crowd_layer, **kwargs)

        if crowd_layer:
            # TODO model.compile('adam', CrowdLayer...)
            raise NotImplementedError
        elif kl_div:
            model.compile('adam', 'kullback_leibler_divergence')
        else:
            model.compile('adam', 'categorical_crossentropy')

    if isinstance(weights_file, str) and os.path.isfile(weights_file):
        model.load_weights(weights_file)

    return model


def vgg16_model(
    input_shape=(256, 256, 3),
    num_labels=8,
    crowd_layer=False,
    parts='labelme',
):
    """VGG16 model used in Crowd Layer paper on LabelMe dataset.

    Parameters
    ----------
    input_shape : tuple(ints)
        Dimensions of an image
    num_label : int
        The number of labels to be classified.
    crowd_layer : bool
        Uses crowd layer in model if True, standard model output otherwise.
    parts : str
        indicates the part of the model to be loaded, either the DNN only, the
        classifier at the end only, or the full model. Useful for preprocessing
        and training of frozen DNN encoded samples.

    Returns
    -------
        The initialized VGG16 model.
    """
    if parts == 'full' or parts == 'vgg16':
        input_layer = keras.layers.Input(shape=input_shape, dtype='float32')

        # create model and freeze them model.
        vgg16 = keras.applications.vgg16.VGG16(False, input_tensor=input_layer)
        for layer in vgg16.layers:
            layer.trainable = False
        x = vgg16.layers[-1].output

        if parts == 'vgg16':
            return keras.models.Model(inputs=input_layer, outputs=x)
    elif parts.lower() == 'labelme':
        input_layer = keras.layers.Input(shape=(8, 8, 512), dtype='float32')
        x = input_layer
    else:
        raise ValueError(
            '`parts`: expected "full", "vgg16", or "labelme", but recieved '
            + f'`{parts}`.'
        )

    # Add the layers specified in Crowd Layer paper.
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(num_labels, activation='softmax')(x)

    if crowd_layer:
        # TODO: Crowd Layer for the VGG16 model.
        raise NotImplementedError

    return keras.models.Model(inputs=input_layer, outputs=x)


def resnext50_model(
    input_shape=(224, 224, 3),
    num_labels=5,
    crowd_layer=False,
):
    """Creates the ResNeXt50 model.

    Parameters
    ----------
    input_shape : tuple(ints)
        Dimensions of an image
    num_label : int
        The number of labels to be classified.
    crowd_layer : bool
        Uses crowd layer in model if True, standard model output otherwise.

    Returns
    -------
        The initialized ResNeXt50 model from Keras.
    """
    input_layer = keras.layers.Input(shape=input_shape, dtype='float32')

    # create model and freeze them model.
    # NOTE this requires re-adding ResNeXt codelines in merge:
    # https://github.com/keras-team/keras/pull/11203/files
    resnext50 = keras.applications.resnext.ResNeXt50(input_tensor=input_layer)
    x = resnext50.layers[-2].output

    # TODO need to do a thing to make the model for the FB dataset...
    # ie. output layers. Match the pytorch implementation.
    x = keras.layers.Dense(num_labels, activation='softmax')(x)

    if crowd_layer:
        # TODO
        raise NotImplementedError

    return keras.models.Model(inputs=input_layer, outputs=x)


if __name__ == '__main__':
    #args, data_config, model_config, kfold_cv_args, random_seeds = experiment.io.parse_args()
    #args, random_seeds = experiment.io.parse_args()
    args = experiment.io.parse_args()

    # Undo need for parse_args to return random_seeds
    # TODO fix this mess here and its usage in predictors.py`
    #if args.random_seeds and len(args.random_seeds) == 1:
    #    args.kfold_cv.random_seed = args.random_seeds[0]
    #    random_seeds = None
    #else:
    #    random_seeds = args.random_seeds

    #kfold_cv_dict = vars(kfold_cv)
    #kfold_cv_dict['random_seed'] = [args.random_seeds] if (args.random_seeds, int) else args.random_seeds

    # TODO implement testing of a specific model and data partition from summary.json
    # TODO then implement that on wide scale for all 'checkpoints' missing predictions.

    if args.which_gpu:
        # TODO does not work atm...
        raise NotImplementedError('Selecting specific GPU not implemented.')
    else:
        run_experiment(
            args.output_dir,
            args.label_src,
            args.dataset_id,
            vars(args.data),
            vars(args.model),
            vars(args.kfold_cv),
            focus_fold=args.focus_fold,
            random_seeds=args.random_seeds,
        )
