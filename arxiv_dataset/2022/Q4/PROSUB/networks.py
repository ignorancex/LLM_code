import autokeras as ak
import tensorflow as tf
from keras_tuner.engine.hyperparameters import Choice, Fixed

from layers import (DenseBlockCustom, DownsamplingMultBlock, MultBlock,
                    RegressionHeadCustom, SigmoidBlock)


def NAS_automodel(
    loss,
    network_name="prosub",
    n_features=1344,
    n_outputs=1344,
    use_batchnorm=False,
    nas_fixed=False,
    seed=None,
    dropout_nas=(0.0,),
    num_layers_nas=(1, 2, 3),
    num_units_nas=(16, 32, 64, 128, 256, 512, 1024),
    num_units_init=(1063, 781, 781, 1063),
):
    """Scorer + Predictor/Network Network, End-To-End"""

    input_x = ak.Input(name="input_x", shape=(n_features,))  # input measurement x

    if len(dropout_nas) == 1:
        dropout = Fixed("dropout", value=dropout_nas[0])
    else:
        values = dropout_nas.copy()
        values.sort()
        dropout = Choice("dropout", values=values, default=dropout_nas[0])
    if len(num_layers_nas) == 1:
        num_layers = Fixed("num_layers", value=num_layers_nas[0])
    else:
        values = num_layers_nas.copy()
        values.sort()
        num_layers = Choice("num_layers", values=values, default=num_layers_nas[0])
    kernel_initializer = tf.keras.initializers.HeNormal(seed=seed)
    bias_initializer = tf.keras.initializers.HeNormal(seed=seed)

    x = input_x
    no_base_layers = 2
    sel_units = num_units_init[:2]
    for i, unit in enumerate(sel_units):
        num_units_choice = num_units_nas.copy() + [unit]
        num_units_choice.sort()  # needs to be ordered
        x = DenseBlockCustom(
            name="SDense_" + str(i),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation="relu",
            num_layers=num_layers,
            num_units=Choice("num_units", num_units_choice, default=unit),
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            nas_fixed=nas_fixed,
        )(x)

    x = DenseBlockCustom(
        name="SDense_" + str(no_base_layers),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        num_units=Fixed("num_units", n_features),
        num_layers=Fixed("num_layers", 1),
        use_batchnorm=False,
        dropout=Fixed("dropout", 0.0),
        activation=None,
    )(x)

    if network_name == "prosub":
        x = SigmoidBlock(multiplier=2)(x)

    x = DownsamplingMultBlock(
        name="downsampling_mult_layer",
        network_name=network_name,
        feature_dim=n_features,
    )(x)

    x = MultBlock(name="DownsamplingOp")([x, input_x])

    pred_units = num_units_init[2:4]
    for i, unit in enumerate(pred_units):
        num_units_choice = num_units_nas.copy() + [unit]
        num_units_choice.sort()
        x = DenseBlockCustom(
            name="PDense_" + str(i),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation="relu",
            num_layers=num_layers,
            num_units=Fixed("num_units", unit),
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            nas_fixed=nas_fixed,
        )(x)

    x = RegressionHeadCustom(
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        output_dim=n_outputs,
        loss=loss,
        dropout=0,
        name="output_1",
    )(x)

    return {"inputs": input_x, "outputs": x}
