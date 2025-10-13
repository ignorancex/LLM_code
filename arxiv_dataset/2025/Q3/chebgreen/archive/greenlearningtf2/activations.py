from .backend import tf, np, config

class CoeffsInitializer(tf.keras.initializers.Initializer):
    def __init__(self) -> None:
        super().__init__()
    def __call__(self, shape, dtype = config(tf), **kwargs):
        if shape == [4,]:
            return tf.constant([1.1915, 1.5957, 0.5, 0.0218], dtype = dtype)
        elif shape == [3,]:
            return tf.constant([2.383, 0.0, 1.0], dtype = dtype)
    def get_config(self):
        config = super(CoeffsInitializer, self).get_config()
        return config
class Rational(tf.keras.layers.Layer):
    """
    Rational Activation function.
    f(x) = P(x) / Q(x)
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self, name = None, **kwargs):
        super(Rational, self).__init__(name = name, **kwargs)

    def build(self, input_shape):
        self.Pcoeffs = self.add_weight(name = 'Pcoeffs', shape = [4,], dtype = config(tf), trainable = True, initializer = CoeffsInitializer)
        self.Qcoeffs = self.add_weight(name = 'Qcoeffs', shape = [3,], dtype = config(tf), trainable = True, initializer = CoeffsInitializer)

    def call(self,
            inputs: tf.Tensor
            ) -> tf.Tensor:
        return tf.math.divide(tf.math.polyval(tf.unstack(self.Pcoeffs),inputs), tf.math.polyval(tf.unstack(self.Qcoeffs),inputs))

    def get_config(self):
        config = super(Rational, self).get_config()
        config.update({'Pcoeffs': self.get_weights()[0],
                  'Qcoeffs': self.get_weights()[1]})
        return config

def get_activation(identifier):
    """Return the activation function."""
    if isinstance(identifier, str):
        return {
                'elu': tf.nn.elu,
                'relu': tf.nn.relu,
                'selu': tf.nn.selu,
                'sigmoid': tf.nn.sigmoid,
                'sin': tf.sin,
                'tanh': tf.nn.tanh,
                'rational': Rational(),
                }[identifier]