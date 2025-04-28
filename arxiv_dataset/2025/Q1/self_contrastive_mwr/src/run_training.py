import os
import argparse
import datetime
import random
from typing import List, Tuple

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


from callbacks.callbacks import PerformancePlotCallback
from loader.data_tf_generator import (TensorFlowDataGenerator, TensorFlowDataGlobalGenerator,
                                      TensorFlowDataRegionalGenerator, TensorFlowDataLocalGenerator,
                                      TensorFlowDataJointGenerator)
from models import (base_model, global_model, regional_model, local_model,
                    joint_model)
from models.custom_layers import Identity
from losses.contrastive_losses import contrastive_loss, npairs_loss
from analysis import classification



def parse_arguments() -> Tuple[str, bool, str]:
    """
    Parse command line arguments.

    Returns:
        Tuple[str, bool, str]: A tuple containing the model type, a boolean indicating if contrastive loss is used,
        and the type of contrastive loss.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='base', help='Type of model')
    parser.add_argument('--contrastive_type', type=str, default='none', help='Type of contrastive loss')
    args = parser.parse_args()

    contrastive_type = args.contrastive_type
    is_contrastive = contrastive_type != 'none'
    model_type = args.model_type

    return model_type, is_contrastive, contrastive_type


def gpu_memory_growth() -> None:
    """
    This function enables GPU memory growth for TensorFlow by setting the `memory_growth`
    flag to `True` for each available GPU.

    Returns:
        None
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def set_global_seed(seed: int) -> None:
    """
    Set the global seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    

def get_callbacks(checkpoint_directory: str, folder_name: str, data: Tuple[np.ndarray, np.ndarray],
                  debug: bool = False) -> List[tf.keras.callbacks.Callback]:
    """
    Returns a list of callbacks for training a Keras model.

    Args:
        checkpoint_directory (str): The directory to save the model checkpoints.
        folder_name (str): The name of the folder to store the logs and checkpoints.
        data (Tuple[np.ndarray, np.ndarray]): A tuple containing the input data and target data.
        debug (bool, optional): If True, additional debugging callbacks will be added. Defaults to False.

    Returns:
        List[tf.keras.callbacks.Callback]: A list of Keras callbacks.
    """
    callback_list = []
    
    metric_monitor = 'val_loss'
    mode = 'min'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_directory,
        save_weights_only=False,
        monitor=metric_monitor,
        mode=mode,
        save_best_only=True)
    callback_list.append(model_checkpoint_callback)
    
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor=metric_monitor,
        min_delta=0,
        patience=50,
        mode=mode,
        restore_best_weights=True)
    callback_list.append(early_stopping_callback)
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.9,
        min_delta=1e-5,
        patience=5,
        mode='min',
        min_lr=1e-9)
    callback_list.append(reduce_lr)
    
    log_dir = os.path.join('..', '_logs', 'fit', folder_name) + '/'
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback_list.append(tensorboard_callback)
    #hparams_callback = hp.KerasCallback(log_dir, hparams)
    
    if debug:
        plots_callback = PerformancePlotCallback(x_test=data[0], y_test=data[1],
                                                 model_name=folder_name)
        callback_list.append(plots_callback)
    
    return callback_list


def getLayerIndexByName(model: tf.keras.Model, layername: str) -> int:
    """
    Get the index of a layer in a Keras model by its name.

    Args:
        model (tf.keras.Model): The Keras model.
        layername (str): The name of the layer.

    Returns:
        int: The index of the layer in the model.
    """
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx


def get_contrastive_loss(contrastive_type: str) -> tfa.losses:
    """
    Returns the contrastive loss function based on the specified contrastive type.

    Args:
        contrastive_type (str): The type of contrastive loss to use.

    Returns:
        tfa.losses: The contrastive loss function.

    Raises:
        ValueError: If an invalid contrastive type is provided.
    """
    if contrastive_type == 'triplethard':
        return tfa.losses.TripletHardLoss(margin=0.1)
    elif contrastive_type == 'tripletsemihard':
        return tfa.losses.TripletSemiHardLoss(margin=0.1)
    elif contrastive_type == 'contrastive':
        return contrastive_loss(margin=0.1)
    elif contrastive_type == 'npairs':
        return npairs_loss()
    else:
        raise ValueError(f"Invalid contrastive type: {contrastive_type}")
    

def load_weights_paths(is_contrastive: bool, contrastive_type: str) -> Tuple[str, str, str]:
    """
    Load the paths of the weights files based on the contrastive type.

    Args:
        is_contrastive (bool): A boolean indicating whether the model is contrastive or not.
        contrastive_type (str): The type of contrastive learning.

    Returns:
        Tuple[str, str, str]: A tuple containing the paths of the local, regional, and global weights files.
    """
    if not is_contrastive:
        local_weights_path = os.path.join('..', '_checkpoint', 'local', 'model.h5')
        regional_weights_path = os.path.join('..', '_checkpoint', 'regional', 'model.h5')
        global_weights_path = os.path.join('..', '_checkpoint', 'global', 'model.h5')
    elif contrastive_type == 'triplethard':
        local_weights_path = os.path.join('..', '_checkpoint', 'local_triplethard', 'model.h5')
        regional_weights_path = os.path.join('..', '_checkpoint', 'regional_triplethard', 'model.h5')
        global_weights_path = os.path.join('..', '_checkpoint', 'global_triplethard', 'model.h5')
    elif contrastive_type == 'tripletsemihard':
        local_weights_path = os.path.join('..', '_checkpoint', 'local_tripletsemihard', 'model.h5')
        regional_weights_path = os.path.join('..', '_checkpoint', 'regional_tripletsemihard', 'model.h5')
        global_weights_path = os.path.join('..', '_checkpoint', 'global_tripletsemihard', 'model.h5')
    elif contrastive_type == 'contrastive':
        local_weights_path = os.path.join('..', '_checkpoint', 'local_contrastive', 'model.h5')
        regional_weights_path = os.path.join('..', '_checkpoint', 'regional_contrastive', 'model.h5')
        global_weights_path = os.path.join('..', '_checkpoint', 'global_contrastive', 'model.h5')
    elif contrastive_type == 'npairs':
        local_weights_path = os.path.join('..', '_checkpoint', 'local_npairs', 'model.h5')
        regional_weights_path = os.path.join('..', '_checkpoint', 'regional_npairs', 'model.h5')
        global_weights_path = os.path.join('..', '_checkpoint', 'global_npairs', 'model.h5')
    
    return local_weights_path, regional_weights_path, global_weights_path


def get_generators(model_type: str) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, TensorFlowDataGenerator]:
    """
    Get the data generators for training, validation, and testing.

    Args:
        model_type (str): The type of model to use for generating the data.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, TensorFlowDataGenerator]: A tuple containing the training,
        validation, and testing datasets, as well as the TensorFlowDataGenerator object used for generating the data.
    """
    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2
    batch_size = 4
    seed = 52
    data_path = os.path.join('..', 'data', 'dataset.csv')
    
    if model_type == 'base':
        tensorflow_generator = TensorFlowDataGenerator()
    elif model_type == 'global':
        tensorflow_generator = TensorFlowDataGlobalGenerator()
    elif model_type == 'regional':
        tensorflow_generator = TensorFlowDataRegionalGenerator()
    elif model_type == 'local':
        tensorflow_generator = TensorFlowDataLocalGenerator() 
    elif model_type == 'joint':
        tensorflow_generator = TensorFlowDataJointGenerator()
        
    generators = tensorflow_generator.get_generators(data_path,
                                                     train_ratio=train_ratio,
                                                     validation_ratio=validation_ratio,
                                                     test_ratio=test_ratio,
                                                     batch_size=batch_size,
                                                     contrastive=is_contrastive,
                                                     seed=seed)
    
    return generators


def get_model(model_type: str, is_contrastive: bool, contrastive_type: str,
              data_generator: TensorFlowDataGenerator) -> tf.keras.Model:
    """
    Returns a TensorFlow Keras model based on the specified model type, contrastive type, and data generator.

    Args:
        model_type (str): The type of model to create. Possible values: 'base', 'global', 'regional', 'local', 'joint'.
        is_contrastive (bool): Indicates whether the model should include contrastive loss.
        contrastive_type (str): The type of contrastive loss to use.
        data_generator (TensorFlowDataGenerator): The data generator object used for training.

    Returns:
        tf.keras.Model: The created TensorFlow Keras model.
    """
    class_sum = np.sum(data_generator.y_train.to_numpy().flatten())
    class_total = data_generator.y_train.to_numpy().flatten().shape[0]
    class_weight = 1 - class_sum / class_total
    
    # Default configurations
    loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True,
                                                   alpha=class_weight,
                                                   gamma=0.0,
                                                   from_logits=False,
                                                   label_smoothing=0.01)
    loss_weights = None
    if model_type == 'joint':
        optimizer = keras.optimizers.Adam(learning_rate=1e-7)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    
    metric = model_type
    load_weights = True # Only applicable to 'joint' model
    freeze_weights = False  # Only applicable to 'joint' model
    local_weights_path, regional_weights_path, global_weights_path = load_weights_paths(is_contrastive, contrastive_type)
          
    if model_type == 'base':
        model = base_model.get_model(data_generator.X_size, num_classes=1,
                                        is_contrastive=is_contrastive)
        
        if is_contrastive:
            loss = {'base_output': loss,
                    'base_contrastive': get_contrastive_loss(contrastive_type)}
            loss_weights = {'base_output': 1.0,
                            'base_contrastive': 0.1}
            
    elif model_type == 'global':
        model = global_model.get_model(data_generator.X_size, num_classes=1,
                                        is_contrastive=is_contrastive)
        
        if is_contrastive:
            loss = {'global_output': loss,
                    'global_contrastive': get_contrastive_loss(contrastive_type)}
            loss_weights = {'global_output': 1.0,
                            'global_contrastive': 0.1}
        
    elif model_type == 'regional':
        model = regional_model.get_model(data_generator.X_size, num_classes=1,
                                            is_contrastive=is_contrastive)
        
        if is_contrastive:
            loss = {'regional_output': loss,
                    'regional_contrastive': get_contrastive_loss(contrastive_type)}
            loss_weights = {'regional_output': 1.0,
                            'regional_contrastive': 0.1}
        
    elif model_type == 'local':
        model = local_model.get_model(data_generator.X_size, data_generator.X_input_size,
                                        num_classes=1, is_contrastive=is_contrastive)
        if is_contrastive:
            loss = {'local_output': loss,
                    'local_contrastive': get_contrastive_loss(contrastive_type)}
            loss_weights = {'local_output': 1.0,
                            'local_contrastive': 0.1}
        
    elif model_type == 'joint':
        model = joint_model.get_model(data_generator.X_sizes,
                                        data_generator.X_local_input_size, num_classes=1,
                                        is_contrastive=is_contrastive)
        loss = {'local_output': loss,
                'regional_output': loss,
                'global_output': loss,
                'joint_output': loss}
        loss_weights = {'local_output': 0.0,
                        'regional_output': 0.0,
                        'global_output': 0.0,
                        'joint_output': 1.0}
        
        if is_contrastive:
            loss['joint_local_contrastive'] = get_contrastive_loss(contrastive_type)
            loss['joint_regional_contrastive'] = get_contrastive_loss(contrastive_type)
            loss['joint_global_contrastive'] = get_contrastive_loss(contrastive_type)
            loss_weights['joint_local_contrastive'] = 0.1
            loss_weights['joint_regional_contrastive'] = 0.1
            loss_weights['joint_global_contrastive'] = 0.1
        
    if model_type == 'joint' and load_weights:
        tmp_model = tf.keras.models.load_model(local_weights_path,
                                                custom_objects={'Identity': Identity,
                                                                'batchwise_contrastive_loss': get_contrastive_loss(contrastive_type),
                                                                'batchwise_npairs_loss': get_contrastive_loss(contrastive_type)})
        idx = getLayerIndexByName(model, 'model_1')
        model.layers[idx].set_weights(tmp_model.get_weights())
        if freeze_weights:
            model.layers[idx].trainable = False
        
        tmp_model = tf.keras.models.load_model(regional_weights_path,
                                                custom_objects={'Identity': Identity,
                                                                'batchwise_contrastive_loss': get_contrastive_loss(contrastive_type),
                                                                'batchwise_npairs_loss': get_contrastive_loss(contrastive_type)})
        idx = getLayerIndexByName(model, 'model_3')
        model.layers[idx].set_weights(tmp_model.get_weights())
        if freeze_weights:
            model.layers[idx].trainable = False
        
        tmp_model = tf.keras.models.load_model(global_weights_path,
                                                custom_objects={'Identity': Identity,
                                                                'batchwise_contrastive_loss': get_contrastive_loss(contrastive_type),
                                                                'batchwise_npairs_loss': get_contrastive_loss(contrastive_type)})
        idx = getLayerIndexByName(model, 'model_5')
        model.layers[idx].set_weights(tmp_model.get_weights())
        if freeze_weights:
            model.layers[idx].trainable = False
    
    model.compile(optimizer=optimizer,
                    loss=loss,
                    loss_weights=loss_weights,
                    metrics={f'{metric}_output': tf.keras.metrics.BinaryAccuracy()})
    
    return model


def train_predict_model(model: tf.keras.Model, model_type: str,
                        train_generator: tf.data.Dataset, validation_generator: tf.data.Dataset,
                        test_generator: tf.data.Dataset) -> Tuple[str, np.ndarray]:
    """
    Trains the given model using the provided data generators and returns the folder name in which
    the weights of the model are saved and predicted values.

    Args:
        model (tf.keras.Model): The model to be trained.
        model_type (str): The type of the model.
        train_generator (tf.data.Dataset): The data generator for training data.
        validation_generator (tf.data.Dataset): The data generator for validation data.
        test_generator (tf.data.Dataset): The data generator for test data.

    Returns:
        Tuple[str, np.ndarray]: A tuple containing the folder name and predicted values.
    """
    prefix = model_type
    if is_contrastive:
        prefix += '_' + contrastive_type
    date_str = datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')
    folder_name = prefix + date_str
    checkpoint_path = os.path.join('..', '_checkpoint', folder_name)
    checkpoint_model_path = os.path.join(checkpoint_path, 'model.h5')
    
    epochs = 1000
    model.fit(x=train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=get_callbacks(checkpoint_model_path, folder_name,
                                        (validation_generator, data_generator.y_val),
                                        debug=False),
                verbose=1)
        
    y_val_pred = model.predict(validation_generator)
    if len(model.output_names) > 1:
        results = {name: output for name, output in zip(model.output_names, y_val_pred)}
        y_val_pred = results[f'{model_type}_output']

    y_test_pred = model.predict(test_generator)
    if len(model.output_names) > 1:
        results = {name: output for name, output in zip(model.output_names, y_test_pred)}
        y_test_pred = results[f'{model_type}_output']

    return folder_name, y_val_pred, y_test_pred


def evaluate_results(folder_name: str, y_val_pred: np.ndarray, y_test_pred: np.ndarray,
                     data_generator: TensorFlowDataGenerator) -> None:
    """
    Evaluate the results of the model by generating classification metrics and plots.

    Args:
        folder_name (str): The folder name where the results are saved.
        y_val_pred (np.ndarray): The predicted values of the model on the validation set.
        y_test_pred (np.ndarray): The predicted values of the model on the test set.
        data_generator (TensorFlowDataGenerator): The data generator object used for training.

    Returns:
        None
    """
    development_phase = False
    threshold = 0.5
    output_folder = '_output_results'
    output_folder = os.path.join('..', output_folder, folder_name)
    
    if development_phase:
        y_target = data_generator.y_val
        y_pred = y_val_pred
    else:
        y_target = data_generator.y_test
        y_pred = y_test_pred

    classification.predictions_to_csv(target_values=y_target.to_numpy().flatten(),
                                      predicted_values=y_pred.flatten(),
                                      output_folder=output_folder)
    classification.classification_report(y_target, y_pred,
                                         threshold=threshold, output_folder=output_folder)
    classification.confusion_matrix(y_target, y_pred,
                                    threshold=threshold, output_folder=output_folder)
    classification.roc_curve(y_target, y_pred,
                             output_folder=output_folder)
    classification.pr_curve(y_target, y_pred,
                            output_folder=output_folder)
    classification.det_curve(y_target, y_pred,
                             output_folder=output_folder)


if __name__ == '__main__':
    keras.backend.clear_session()
    set_global_seed(513)
    gpu_memory_growth()
    
    model_type, is_contrastive, contrastive_type = parse_arguments()

    train_generator, validation_generator, test_generator, data_generator = get_generators(model_type)        
    model = get_model(model_type, is_contrastive, contrastive_type, data_generator)
    folder_name, y_val_pred, y_test_pred = train_predict_model(model, model_type, train_generator,
                                                               validation_generator, test_generator)
    evaluate_results(folder_name, y_test_pred, data_generator)
