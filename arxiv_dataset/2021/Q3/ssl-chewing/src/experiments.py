"""
Experiment based on the SimCLR idea [1]
- LOSO experiment
- dataset: wu2
- input: windows
- ground-truth: chewing vs non-chewing or eating vs non-eating

b) Training
   - model: combination of [1] and [2]
b) Evaluate on wu2 LOSO

[1] https://github.com/google-research/simclr
[2] https://ieeexplore.ieee.org/abstract/document/8037060/
"""

import pickle
from pathlib import Path
from typing import List, NoReturn

import tensorflow as tf
from absl import flags, app
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.optimizers import Adam

import globalconfig as g_conf
from dataset.augmentors import BaseAugmentor, NoiseAugmentor, RandomAmplifierAugmentor
from dataset.commons import SampleType, GroundTruthType
from dataset.labeltransform import CategoricalLabelTransform, NumpyCastLabelTransform
from dataset.utilities import suggest_patience
from dataset.wu2 import wu2metadata
from dataset.wu2.audiowindowsubset import AudioWindowSubsetBuilder
from dataset.wu2.audiowindowsubsetmetadata import AudioWindowSubsetMetadata
from dataset.wu2.loaders import AudioWindowLoaderConf
from dataset.wu2.wu2metadata import WU2Metadata
from evaluation.postprocessing import chewing2meals
from evaluation.result import PredictionType, Result, ClassificationType
from model import papapanagiotou2017convolutional as papapa2017
from model.simclrprojectionhead import nonlinear_projection_head, linear_projection_head
from optimizer.larsoptimizer import LARSOptimizer
from simclr.contrastiveloss import create_contrastive_loss
from simclr.dualaugmentor import DualAugmentor
from simclr.duallabeltransform import DualLabelTransform
from simclr.simclrhelper import SimCLRHelper
from simclr.warmupandcosinedecay import WarmUpAndCosineDecay
from utilities.datetimeutils import get_str_timestamp
from utilities.listutils import select_by_index
from utilities.printutils import print_with_header, pretty_lopo_header
from utilities.typingutils import is_typed_list
from utilities.vartools import print_label_histogram

#
# Program flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('tensorflow_memory_growth', False, 'Enable tensorflow memory growth')
flags.DEFINE_float('temperature', 1.0, 'Temperature value of contrastive loss for task-agnostic training')
flags.DEFINE_string('projection_head', 'linear', 'Type of projection head (linear or non-linear)')
flags.DEFINE_integer('g_keep', 0, 'Number of layers of the projection head to retain for the task-specific model', 0)
flags.DEFINE_integer('ta_epochs', 50, 'Number of epochs to train the task-agnostic model', 0)
flags.DEFINE_integer('ts_epochs', 50, 'Number of epochs to train the task-specific model', 0)
flags.DEFINE_string('load_ta_model', None, 'The name (timestamp) of the task-agnostic model to load for parts 2 and 3')
flags.DEFINE_boolean('part3', False, 'Run part 3 (instead of parts 1 and 2')
flags.DEFINE_integer('keras_verbosity', 1, 'Verbosity level for keras', 0, 2)

#
# ---   Global configuration   ---
g_conf.set_experiment_name(__file__, True)
g_timestamp = get_str_timestamp()


#
# ---   Terminology   ---
#
# The entire dataset is split into 2 partitions:
#   - devel: it is used to train the feature extractor, and also train and evaluate a classifier (using LOPO)
#   - test:  it is used to evaluate the trained feature extractor and classifier to a completely "unseen" dataset
#
# The devel set is further partitioned for the LOPO experiment


def part_0():
    #
    # ---   Common options   ---

    # Audio version and window specification
    fs: float = 2000.0
    loader_conf: AudioWindowLoaderConf = AudioWindowLoaderConf("2khz_hpf20hz", fs, 5.0, 1.0)

    # Type of ground that will be used to evaluation classification tasks
    ground_truth_type: GroundTruthType = GroundTruthType.MEAL

    # Partition (i.e. subject) indices that will be used for the final testing
    test_partition_idxs: List[int] = [0, 1, 2, 3]
    validation_partition_idxs: List[int] = [4, 5]

    #
    # ---   SimCLR models   ---

    # Define inputs (TODO check if a single input layer can be used for both models)
    ta_input: Input = Input((loader_conf.wsize, 1), None, "ta_input")
    ts_input: Input = Input((loader_conf.wsize, 1), None, "ts_input")

    # Define layers
    f: List[Layer] = papapa2017.feature_extraction_layers_5sec()
    if FLAGS.projection_head == 'linear':
        g: List[Layer] = linear_projection_head(128)
    elif FLAGS.projection_head == 'non-linear':
        g: List[Layer] = nonlinear_projection_head(128, 512)  # 512 is an arbitrary guess of f.output_shape
    else:
        raise ValueError('Unknown projection_head value: ' + FLAGS.projection_head)
    h: List[Layer] = papapa2017.classification_layers(False)

    # create models and print summaries
    models: SimCLRHelper = SimCLRHelper(ta_input, ts_input, f, g, h, FLAGS.g_keep)

    # models.ta_model.summary()
    # models.ts_model.summary()

    #
    # ---   Initialization   ---
    flags_file: Path = g_conf.get_results_path() / ('flags ' + g_timestamp + '.txt')
    if flags_file.exists():
        flags_file.unlink()
    FLAGS.append_flags_into_file(flags_file)

    return test_partition_idxs, validation_partition_idxs, loader_conf, models, ground_truth_type


#
# ---   Part 1   ---
def part_1(test_partition_idxs: List[int], loader_conf: AudioWindowLoaderConf, models: SimCLRHelper,
           ground_truth_type: GroundTruthType) -> NoReturn:
    """
    Experiment part 1: Train the task-agnostic (and projection head) layers on devel set.
    """
    assert is_typed_list(test_partition_idxs, int)
    assert isinstance(loader_conf, AudioWindowLoaderConf)
    assert isinstance(models, SimCLRHelper)
    assert isinstance(ground_truth_type, GroundTruthType)

    batch_size: int = 256
    epochs: int = FLAGS.ta_epochs

    md: WU2Metadata = WU2Metadata(SampleType.WINDOW)

    # Define the two augmentors that will be applied in parallel
    augmentor1: BaseAugmentor = RandomAmplifierAugmentor(.5, 2.0)
    augmentor2: BaseAugmentor = NoiseAugmentor(NoiseAugmentor.NoiseType.UNIFORM, -0.005, 0.005)

    # Create the dataset subset
    builder = AudioWindowSubsetBuilder(md, batch_size, ground_truth_type, loader_conf,
                                       DualAugmentor(augmentor1, augmentor2), DualLabelTransform())
    devel_set, _, _ = builder.split(select_by_index(md.partitions, test_partition_idxs))
    devel_set.shuffle()  # Shuffle once now to avoid batches in chronological order during the first epoch

    # NOTE warmup_epochs=round(0.05 * epochs) in the paper
    warmup_epochs = max(5, round(0.1 * epochs))
    optimizer = LARSOptimizer(
        WarmUpAndCosineDecay(0.3, 'linear', warmup_epochs, epochs, devel_set.__len__(), batch_size),
        weight_decay=1e-4)

    loss = create_contrastive_loss(temperature=FLAGS.temperature)

    callbacks = [TensorBoard(log_dir=str(g_conf.get_tensorboard_logdir() / g_timestamp / 'task-agnostic'),
                             update_freq='batch')]

    # models.load_weights()  # Optionally, load weights from saved model
    models.set_ta_mode()
    models.ta_model.compile(optimizer, loss)
    models.ta_model.fit(devel_set, epochs=epochs, verbose=FLAGS.keras_verbosity, callbacks=callbacks)
    models.save_weights(g_conf.get_models_path() / g_timestamp)


#
# ---   Part 2   ---
def part_2(test_partition_idxs: List[int], loader_conf: AudioWindowLoaderConf, models: SimCLRHelper,
           ground_truth_type: GroundTruthType) -> NoReturn:
    """
    Part 2: Train the classifier in LOSO mode over the devel set using the pre-trained task-agnostic layers.
    """
    assert is_typed_list(test_partition_idxs, int)
    assert isinstance(loader_conf, AudioWindowLoaderConf)
    assert isinstance(models, SimCLRHelper)
    assert isinstance(ground_truth_type, GroundTruthType)

    # Configuration
    batch_size: int = 64
    epochs: int = FLAGS.ts_epochs
    noof_validation_users: int = 2

    md: WU2Metadata = WU2Metadata(SampleType.WINDOW)

    # Create the dataset
    _, devel_md = wu2metadata.split(md, test_partition_idxs)
    subset_md = AudioWindowSubsetMetadata(devel_md, loader_conf)

    # Create the model
    models.ts_model.compile(Adam(), BinaryCrossentropy(), [BinaryAccuracy(), Precision(), Recall()])

    # Create result viewers
    prediction_type = PredictionType.PROBABILITY
    results = Result(subset_md.get_noof_samples(), ClassificationType.BINARY, 2, prediction_type)

    # Leave-one-partition-out loop
    for test_idx, test_partition in enumerate(devel_md.partitions):
        print_with_header('Test ' + pretty_lopo_header(devel_md.partitions, test_idx), 'LOPO loop')

        # Create the dataset subsets
        builder = AudioWindowSubsetBuilder(devel_md, batch_size, ground_truth_type, loader_conf,
                                           label_transform=NumpyCastLabelTransform(float))
        train_set, test_set, validation_set = builder.lopo_split(test_idx, noof_validation_users)
        train_set.shuffle()  # Shuffle once now to avoid batches in chronological order during the first epoch

        print_label_histogram(test_set.get_y())

        # Load original weights: pretrained f() and g(), randomly initialized h()
        load_g_timestamp: str
        if FLAGS.load_ta_model is None:
            load_g_timestamp = g_timestamp
        else:
            load_g_timestamp = FLAGS.load_ta_model
        models.load_weights(g_conf.get_models_path() / load_g_timestamp)
        models.set_ts_mode()
        model_name: str = 'ts_lopo_' + str(test_idx) + '_best_model.h5'

        # patience = suggest_patience(epochs)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=suggest_patience(epochs), restore_best_weights=True),
            ModelCheckpoint(filepath=str(g_conf.get_models_path() / g_timestamp / model_name), save_best_only=True),
            TensorBoard(
                log_dir=str(g_conf.get_tensorboard_logdir() / g_timestamp / ('task-specific LOPO ' + str(test_idx))))]

        models.ts_model.fit(train_set, epochs=epochs, verbose=FLAGS.keras_verbosity, validation_data=validation_set,
                            callbacks=callbacks)

        print(models.ts_model.evaluate(test_set, verbose=FLAGS.keras_verbosity, return_dict=True))

        y_true = test_set.get_y()
        y_pred = models.ts_model.predict(test_set, verbose=FLAGS.keras_verbosity)
        results.append(subset_md.get_global_idxs(test_idx), y_true, y_pred, test_idx)

    results_file: Path = g_conf.get_results_path() / ('ts_lopo_results ' + g_timestamp + '.pickle')
    if results_file.exists():
        results_file.unlink()
    with open(str(results_file), 'wb') as file:
        pickle.dump(results, file)

    for report in results.get_classification_reports():
        print(report)

    results._y_pred = chewing2meals(results._y_pred, loader_conf.wstep_sec)
    for report in results.get_classification_reports():
        print(report)


#
# ---   Part 3   ---
def part_3(test_partition_idxs: List[int], validation_partition_idxs: List[int], loader_conf: AudioWindowLoaderConf,
           models: SimCLRHelper, ground_truth_type: GroundTruthType) -> NoReturn:
    """
    Part 3: - Load the task-agnostic (and projection head) layers on devel set.
            - Train the classifier in the devel set.
            - Evaluate on the test set.
    """
    assert is_typed_list(test_partition_idxs, int)
    assert isinstance(loader_conf, AudioWindowLoaderConf)
    assert isinstance(models, SimCLRHelper)
    assert isinstance(ground_truth_type, GroundTruthType)

    # Configuration
    batch_size: int = 64
    epochs: int = FLAGS.ts_epochs

    md: WU2Metadata = WU2Metadata(SampleType.WINDOW)

    # Create the dataset
    test_set_md, _ = wu2metadata.split(md, test_partition_idxs)
    subset_md = AudioWindowSubsetMetadata(test_set_md, loader_conf)

    # Create the model
    models.ts_model.compile(Adam(), BinaryCrossentropy(), [BinaryAccuracy(), Precision(), Recall()])

    # Create result viewers
    prediction_type = PredictionType.PROBABILITY
    results = Result(subset_md.get_noof_samples(), ClassificationType.BINARY, 2, prediction_type)

    # Create the dataset subsets
    builder = AudioWindowSubsetBuilder(md, batch_size, ground_truth_type, loader_conf,
                                       label_transform=NumpyCastLabelTransform(float))
    train_set, test_set, validation_set = builder.split(
        select_by_index(md.partitions, test_partition_idxs),
        select_by_index(md.partitions, validation_partition_idxs))
    train_set.shuffle()  # Shuffle once now to avoid batches in chronological order during the first epoch

    print_label_histogram(test_set.get_y())

    if FLAGS.load_ta_model is None:
        # Do not load anything, but enable training of the entire model
        models.set_ta_mode()
    else:
        # Load original weights: pretrained f() and g(), randomly initialized h(), and only allow h() to be trained
        models.load_weights(g_conf.get_models_path() / FLAGS.load_ta_model)
        models.set_ts_mode()

    model_name: str = 'ts_best_model.h5'

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=suggest_patience(epochs), restore_best_weights=True),
        ModelCheckpoint(filepath=str(g_conf.get_models_path() / g_timestamp / model_name), save_best_only=True),
        TensorBoard(log_dir=str(g_conf.get_tensorboard_logdir() / g_timestamp / 'task-specific'))]

    models.ts_model.fit(train_set, epochs=epochs, verbose=FLAGS.keras_verbosity, validation_data=validation_set,
                        callbacks=callbacks)

    for test_idx, test_partition in enumerate(test_set_md.partitions):
        builder2 = AudioWindowSubsetBuilder(test_set_md, batch_size, ground_truth_type, loader_conf,
                                            label_transform=NumpyCastLabelTransform(float))
        _, lopo_test_set, _ = builder2.lopo_split(test_idx)

        y_true = lopo_test_set.get_y()
        y_pred = models.ts_model.predict(lopo_test_set, verbose=FLAGS.keras_verbosity)
        results.append(subset_md.get_global_idxs(test_idx), y_true, y_pred, test_idx)

    results_file: Path = g_conf.get_results_path() / ('ts_results ' + g_timestamp + '.pickle')
    if results_file.exists():
        results_file.unlink()
    with open(str(results_file), 'wb') as file:
        pickle.dump(results, file)

    for report in results.get_classification_reports():
        print(report)

    results._y_pred = chewing2meals(results._y_pred, loader_conf.wstep_sec)
    for report in results.get_classification_reports():
        print(report)


#
# ---   Main   ---
def main(args):
    del args

    if FLAGS.tensorflow_memory_growth:
        for dev in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(dev, True)

    test_partition_idxs, validation_partition_idxs, loader_conf, models, ground_truth_type = part_0()

    if not FLAGS.part3:
        # Only run part 1 if not using an existing trained model
        if FLAGS.load_ta_model is None:
            part_1(test_partition_idxs, loader_conf, models, ground_truth_type)
        part_2(test_partition_idxs, loader_conf, models, ground_truth_type)
    else:
        part_3(test_partition_idxs, validation_partition_idxs, loader_conf, models, ground_truth_type)


if __name__ == "__main__":
    app.run(main)
