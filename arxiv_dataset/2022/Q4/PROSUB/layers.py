import autokeras as ak
import tensorflow as tf
from autokeras.blocks import reduction
from autokeras.utils import utils
from keras_tuner.engine import hyperparameters
from tensorflow.keras import layers
from tensorflow.python.util import nest


class DownsamplingMultBlock(ak.Block):
    class DownsamplingMultLayerBase(tf.keras.layers.Layer):
        """Base layer for subsampling, holds variables for use across NAS steps"""

        def __init__(self, feature_dim, name=None, **kwargs):
            super().__init__(name=name)
            self.feature_dim = feature_dim

            cfg = dict(trainable=False, initializer=tf.keras.initializers.Ones())
            # If change order here, change in load_predictor function
            addw = self.add_weight
            self.m = addw(**cfg, shape=(1, feature_dim), dtype="float32", name="m")
            self.sigma = addw(
                **cfg, shape=(1, feature_dim), dtype="float32", name="sigma"
            )
            self.sigma_bar = addw(
                **cfg, shape=(1, feature_dim), dtype="float32", name="sigma_bar"
            )
            self.alpha_t = addw(**cfg, dtype="float32", name="alpha_t")
            self.t = addw(**cfg, dtype="int32", name="t")
            self.M = addw(**cfg, dtype="int32", name="M")
            self.M_target = addw(**cfg, dtype="int32", name="M_target")
            self.M_target_best_val = addw(
                **cfg, dtype="float32", name="M_target_best_val"
            )

        def assign(self, **kwargs):
            for key, val in kwargs.items():
                func = getattr(self, key)
                func.assign(val)

        def get(self, *args):
            ret = []
            for key in args:
                out = getattr(self, str(key)).numpy()
                ret.append(out)
            return tuple(ret)

    class DownsamplingMultLayerPROSUB(DownsamplingMultLayerBase):
        """Subsampling Layer for PROSUB"""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, inputs):
            sigma = tf.keras.backend.mean(inputs, axis=0, keepdims=True)
            self.sigma.assign(sigma)
            out = self.alpha_t * inputs + (1 - self.alpha_t) * self.sigma_bar
            out = out * self.m
            return out

    class DownsamplingMultLayerSardunet(DownsamplingMultLayerBase):
        """Subsampling Layer for SARDU-Net, implemented in tf/keras"""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, inputs):
            x = tf.keras.layers.Softmax(axis=1)(-inputs)
            x = tf.keras.backend.mean(x, axis=0, keepdims=True)
            _, max_idx = tf.nn.top_k(x, k=self.M, sorted=True)

            m = tf.tensor_scatter_nd_add(
                tf.zeros([self.feature_dim]),
                tf.transpose(max_idx),
                tf.ones([self.M]),
            )  # self.feature_dim,
            m = tf.expand_dims(m, axis=0)
            self.m.assign(m)

            x = x * m
            x = x / tf.math.reduce_sum(x)
            self.sigma_bar.assign(x)  # measurements are only normalized within selected
            return x

    def __init__(self, network_name, feature_dim, name=None, **kwargs):
        super().__init__()
        if network_name == "prosub":
            self.downsampling_mult_layer = self.DownsamplingMultLayerPROSUB(
                name=name, feature_dim=feature_dim, **kwargs
            )
        elif network_name == "sardunet-nas":
            self.downsampling_mult_layer = self.DownsamplingMultLayerSardunet(
                name=name, feature_dim=feature_dim, **kwargs
            )
        else:
            assert False, "network_name in {prosub|sardunet-nas}"

    def build(self, hp, inputs=None):
        out = self.downsampling_mult_layer(inputs[0])
        return out

    def assign(self, **kwargs):
        self.downsampling_mult_layer.assign(**kwargs)

    def get(self, *args):
        self.downsampling_mult_layer.get(*args)


class MultBlock(ak.Block):
    """Multiplies two tensors in autokeras"""

    class MultLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, inputs):
            return inputs[0] * inputs[1]

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.mult = self.MultLayer(name=name)

    def build(self, hp, inputs=None):
        inputs = tf.nest.flatten(inputs)
        return self.mult([inputs[0], inputs[1]])


class SigmoidBlock(ak.Block):
    """Sigmoid layer in autokeras"""

    class SigmoidLayer(tf.keras.layers.Layer):
        def __init__(self, multiplier=1):
            super().__init__()
            self.multiplier = multiplier

        def call(self, inputs):
            return tf.keras.activations.sigmoid(inputs) * self.multiplier

        def get_config(self):
            return {"multiplier": self.multiplier}

    def __init__(self, multiplier=1):
        super().__init__()
        self.sigmoid_mult = self.SigmoidLayer(multiplier=multiplier)

    def build(self, hp, inputs=None):
        inputs = tf.nest.flatten(inputs)
        out = self.sigmoid_mult(inputs[0])
        return out


def MSEWeighted(loss_affine):
    """MSE loss, after denormalizing data"""

    assert len(loss_affine) == 2

    def loss(y_true, y_pred):
        y_true_out = y_true * loss_affine[0] + loss_affine[1]
        y_pred_out = y_pred * loss_affine[0] + loss_affine[1]
        return tf.keras.backend.mean(tf.keras.backend.square(y_true_out - y_pred_out))

    return loss


class DenseBlockCustom(ak.DenseBlock):
    """Extension ak.DenseBlock, more activations, initializations, and no-nas option"""

    def __init__(
        self,
        activation=None,
        dropout_off=False,
        kernel_initializer="he_uniform",
        bias_initializer="he_uniform",
        nas_fixed=False,
        **kwargs
    ):
        self.nas_fixed = nas_fixed
        if nas_fixed:
            self.dummy = hyperparameters.Choice("dummy", list(range(1000)), default=1)
            for arg in ("num_layers", "num_units", "dropout"):
                if isinstance(kwargs[arg], hyperparameters.Choice):
                    default = kwargs[arg].default
                    kwargs[arg] = hyperparameters.Fixed(arg, default)
                    print(arg, "set to  hyperparameters.Fixed(arg,", str(default))

        super().__init__(**kwargs)
        assert activation in {None, "relu", "sigmoid"}
        self.activation = activation
        self.dropout_off = dropout_off
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def get_config(self):
        config = super().get_config()
        if self.nas_fixed:
            config.update(
                {
                    "dummy": hyperparameters.serialize(self.dummy),
                }
            )
        return config

    @classmethod
    def from_config(cls, config):
        config["num_layers"] = hyperparameters.deserialize(config["num_layers"])
        config["num_units"] = hyperparameters.deserialize(config["num_units"])
        config["dropout"] = hyperparameters.deserialize(config["dropout"])
        config["dummy"] = hyperparameters.deserialize(config["dummy"])
        return cls(**config)

    # Add get_config?
    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node
        output_node = reduction.Flatten().build(hp, output_node)

        if self.nas_fixed:
            utils.add_to_hp(self.dummy, hp)

        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Boolean("use_batchnorm", default=False)

        for i in range(utils.add_to_hp(self.num_layers, hp)):
            units = utils.add_to_hp(self.num_units, hp, "units_{i}".format(i=i))
            output_node = layers.Dense(
                units,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
            )(output_node)
            if use_batchnorm:
                output_node = layers.BatchNormalization()(output_node)
            if self.activation is "relu":
                output_node = tf.keras.layers.ReLU()(output_node)
            elif self.activation is "sigmoid":
                output_node = tf.keras.layers.Activation("sigmoid")(output_node)

            if hasattr(self.dropout, "value") and (self.dropout.value == 0):
                pass
            elif utils.add_to_hp(self.dropout, hp) > 0:
                output_node = layers.Dropout(utils.add_to_hp(self.dropout, hp))(
                    output_node
                )
        return output_node


class RegressionHeadCustom(ak.RegressionHead):
    """Extension of ak.RegressionHead, more activations, initializations, and no-nas option"""

    def __init__(
        self, kernel_initializer="he_uniform", bias_initializer="he_uniform", **kwargs
    ):

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        super().__init__(**kwargs)

    # def get_config(self):
    #    config = super().get_config()
    #    config.update({"output_dim": self.output_dim, "dropout": self.dropout})
    #    return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        if dropout > 0:
            output_node = layers.Dropout(dropout)(output_node)
        output_node = reduction.Flatten().build(hp, output_node)
        output_node = layers.Dense(
            self.shape[-1],
            name=self.name,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )(output_node)
        return output_node
