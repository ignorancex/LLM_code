"""
A collection of various custom layers for inteded use with cymetric.

Typical usage example:

    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from cymetric.models.tfmodels import PhiFSModel
    >>> from cymetric.models.metrics import TotalLoss
    >>> from cymetric.models.tfhelper import prepare_tf_basis, train_model
    >>> from invariant_layers import HomogenousCanonicalization

    Load data

    >>> data = np.load('dataset.npz')
    >>> BASIS = prepare_tf_basis(np.load('basis.pickle', allow_pickle=True))

    Setup model including invariant layer

    >>> n_coords = data['X_train'].shape[1]
    >>> ambient_space = np.array([4])
    >>> nn = tf.keras.Sequential(
    ...     [            
    ...         tf.keras.layers.Input(shape=(n_coords)),
    ...         HomogenousCanonicalization(ambient_space),
    ...         tf.keras.layers.Dense(64, activation="gelu"),
    ...         tf.keras.layers.Dense(1),
    ...     ]
    ... )
    >>> model = PhiFSModel(nn, BASIS)

    next we can compile and train

    n_epochs = 50
    metrics = [
    ...    TotalLoss(),
    ... ]
    >>> opt = tf.keras.optimizers.Adam()
    >>> model, training_history = train_model(
    ...         model,
    ...         data,
    ...         optimizer=opt,
    ...         epochs=n_epochs,
    ...         batch_sizes=[64, 50000],
    ...         verbose=1,
    ...         custom_metrics=metrics,
    ...     )
"""

import tensorflow as tf
import numpy as np


class SpectralLayer(tf.keras.layers.Layer):
    """Custom layer that maps the input into the spectral basis."""

    def __init__(self, ambient):
        """Initializes the layer with the given ambient dimensions.

        Args:
            ambient (np.array([n_ambient], np.int)): List of ambient dimensions.
        """
        super(SpectralLayer, self).__init__()
        self.ambient = ambient

    def real_to_complex(self, inputs):
        """Helper function to convert cymetric input into complex form.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Input tensor of cymetric format.

        Returns:
            tf.tensor([batch_size, n_coords], tf.complex64):
            Input converted into complex form.
        """
        real_inputs, imag_inputs = tf.split(inputs, 2, axis=-1)
        complex_inputs = tf.complex(real_inputs, imag_inputs)
        return complex_inputs

    def call(self, inputs):
        """Maps input into spectral basis.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY.

        Returns:
            output (tf.tensor[batch_size, n_spectral_basis], tf.float32):
            Input map into the spectral basis.
            n_spetral_basis depends on the number of ambient spaces and their respective dimensions.
        """
        complex_inputs = self.real_to_complex(inputs)
        splits_by_ambient = tf.split(
            complex_inputs, axis=-1, num_or_size_splits=[x + 1 for x in self.ambient]
        )
        complex_outputs = []
        for i in range(len(splits_by_ambient)):
            ncoords = splits_by_ambient[i].shape[-1]
            zz_bar = tf.einsum(
                "ai,aj->aij", splits_by_ambient[i], tf.math.conj(splits_by_ambient[i])
            )
            z_abs = tf.math.reciprocal(
                tf.reduce_sum(
                    tf.math.multiply(
                        splits_by_ambient[i], tf.math.conj(splits_by_ambient[i])
                    ),
                    1,
                )
            )
            z_abs = tf.reshape(z_abs, [-1])
            zz_bar = tf.reshape(zz_bar, [-1, ncoords**2])
            complex_outputs.append(tf.einsum("bi, b -> bi", zz_bar, z_abs))
        complex_outputs = tf.concat(complex_outputs, axis=-1)
        outputs = tf.concat(
            [tf.math.real(complex_outputs), tf.math.imag(complex_outputs)], axis=-1
        )
        return outputs


class RootScalingCanonicalization(tf.keras.layers.Layer):
    """Custom layer that performs an m-root of unity scaling canonicalization.
    Namely given an input that consists of n complex coordinates then RootScalingCanonicalization
    with degree=m multiply each complex coordinate with a unique m-root of unity such that the arguments of
    all complex coordinates are in the first m-quadrant.
    .. math::
        RootScalingCanonicalization(z_0, z_1, \hdots, z_{n-1}) := (e^{\frac{i\2\pi r_0}{m}} z_0, \hdots, e^{\frac{i\2\pi r_{n-1}}{m}} z_{n-1} )
    .. math::
        \forall j: 0\leq r_j \leq n-1, 0\leq \arg(e^{\frac{i\2\pi r_j}{m}} z_j) \leq 2\pi/m

    Note: Both inputs and outputs are in real format:
    .. math::
        \left(\Re(z_0) \quad \dots \quad \Re(z_{n-1}) \quad \Im(z_0) \quad \dots \quad \Im(z_{n-1}) \right) \in \mathbb{R}^{2n}
    """

    def __init__(self, degree):
        """Initializes the layer with the given degree.

        Args:
            degree: The degree the roots of unity used in scaling.
        """
        super(RootScalingCanonicalization, self).__init__()
        self.degree = degree

    def real_to_complex(self, inputs):
        """Helper function to convert cymetric input into complex form.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Input tensor of cymetric format.

        Returns:
            tf.tensor([batch_size, n_coords], tf.complex64):
            Input converted into complex form.
        """
        real_inputs, imag_inputs = tf.split(inputs, 2, axis=-1)
        complex_inputs = tf.complex(real_inputs, imag_inputs)
        return complex_inputs

    def call(self, inputs):
        """Applies root scaling canonicalization on the input tensor.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY.

        Returns:
            output (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY where the complex arguments of all coordinates lies in the first m-quadrant.
        """
        complex_inputs = self.real_to_complex(inputs)
        angles = tf.math.angle(complex_inputs) * self.degree
        angles = tf.complex(tf.math.cos(angles), tf.math.sin(angles))
        norms = tf.cast(tf.math.abs(complex_inputs), dtype=tf.dtypes.complex64)
        complex_outputs = norms * angles
        outputs = tf.concat(
            [tf.math.real(complex_outputs), tf.math.imag(complex_outputs)], axis=-1
        )
        return outputs


class HomogenousCanonicalization(tf.keras.layers.Layer):
    """Custom layer that performs homogenous canonicalization.
    Namely given an input in C^n that consists of n complex coordinates then HomogenousCanonicalization
    multiply each complex coordinate with the recipricoal of the coordiante with the highest norm.
    .. math::
        HomogenousCanonicalization(z_0, z_1, \hdots, z_{n-1}) := (z_0/z_j, z_1/z_j, \hdots,  z_{n-1}/z_j )
    .. math::
        \forall i: |z_i| \leq |z_j|

    Note: Both inputs and outputs are in real format:
    .. math::
        \left(\Re(z_0) \quad \dots \quad \Re(z_{n-1}) \quad \Im(z_0) \quad \dots \quad \Im(z_{n-1}) \right) \in \mathbb{R}^{2n}
    """

    def __init__(self, ambient):
        """Initializes the layer with the given ambient dimensions.

        Args:
            ambient (np.array([n_ambient], np.int)): List of ambient dimensions.
        """
        super(HomogenousCanonicalization, self).__init__()
        self.ambient = ambient

    def real_to_complex(self, inputs):
        """Helper function to convert cymetric input into complex form.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Input tensor of cymetric format.

        Returns:
            tf.tensor([batch_size, n_coords], tf.complex64):
            Input converted into complex form.
        """
        real_inputs, imag_inputs = tf.split(inputs, 2, axis=-1)
        complex_inputs = tf.complex(real_inputs, imag_inputs)
        return complex_inputs

    def call(self, inputs):
        """Applies homogenous canonicalization on the input tensor.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY.

        Returns:
            output (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY, rescaled by the reciprical of the coordinates with the highest norm.
        """
        complex_inputs = self.real_to_complex(inputs)
        splits_by_ambient = tf.split(
            complex_inputs, axis=-1, num_or_size_splits=[x + 1 for x in self.ambient]
        )
        complex_outputs = []
        for i in range(len(splits_by_ambient)):
            argmax = tf.math.argmax(tf.abs(splits_by_ambient[i]), axis=-1)
            z_max = tf.gather(splits_by_ambient[i], argmax, axis=-1, batch_dims=1)
            z_max = tf.squeeze(tf.math.reciprocal(z_max))
            complex_outputs.append(
                tf.einsum("b, bi -> bi", z_max, splits_by_ambient[i])
            )
        complex_outputs = tf.concat(complex_outputs, axis=-1)
        outputs = tf.concat(
            [tf.math.real(complex_outputs), tf.math.imag(complex_outputs)], axis=-1
        )
        return outputs


class PermCanonicalization(tf.keras.layers.Layer):
    """Custom layer that performs permutation canonicalization.
    Namely given an input in C^n that consists of n complex coordinates then PermCanonicalization
    reorder the coordinates according to norm in a descending order.
    .. math::
        PermCanonicalization(z_0, z_1, \hdots, z_{n-1}) := (z_{j_0}, z_{j_1}, \hdots, z_{j_{n-1}})
    .. math::
        |z_{j_0}| \geq |z_{j_1}| \geq \hdots \geq |z_{j_{n-1}}|

    Note: Both inputs and outputs are in real format:
    .. math::
        \left(\Re(z_0) \quad \dots \quad \Re(z_{n-1}) \quad \Im(z_0) \quad \dots \quad \Im(z_{n-1}) \right) \in \mathbb{R}^{2n}
    """

    def __init__(self, ambient):
        """Initializes the layer with the given ambient dimensions.

        Args:
            ambient (np.array([n_ambient], np.int)): List of ambient dimensions.
        """
        super(PermCanonicalization, self).__init__()
        self.ambient = ambient

    def real_to_complex(self, inputs):
        """Helper function to convert cymetric input into complex form.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Input tensor of cymetric format.

        Returns:
            tf.tensor([batch_size, n_coords], tf.complex64):
            Input converted into complex form.
        """
        real_inputs, imag_inputs = tf.split(inputs, 2, axis=-1)
        complex_inputs = tf.complex(real_inputs, imag_inputs)
        return complex_inputs

    def call(self, inputs):
        """Apply pemrutation canonicalization on the inputs tensor.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY.

        Returns:
            output (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY, with cooridnates reordered descendingly according to norm.
        """
        complex_inputs = self.real_to_complex(inputs)
        splits_by_ambient = tf.split(
            complex_inputs, axis=-1, num_or_size_splits=[x + 1 for x in self.ambient]
        )
        complex_outputs = []
        norms = tf.math.abs(splits_by_ambient[0])
        idx_sort = tf.argsort(norms, axis=-1, direction="DESCENDING")
        for i in range(len(splits_by_ambient)):
            complex_outputs.append(
                tf.gather(splits_by_ambient[i], idx_sort, axis=-1, batch_dims=1)
            )
        complex_outputs = tf.concat(complex_outputs, axis=-1)
        outputs = tf.concat(
            [tf.math.real(complex_outputs), tf.math.imag(complex_outputs)], axis=-1
        )
        return outputs


class ShiftCanonicalization(tf.keras.layers.Layer):
    """Custom layer that performs shift canonicalization.
    Namely given an input in C^n that consists of n complex coordinates then ShiftCanonicalization
    shift the coordinates to the left such that the coordinate with the highest norm becomes the first coordinate.
    .. math::
        ShiftCanonicalization(z_0, z_1, \hdots, z_{n-1}) := (z_{j \mod n}, z_{j + 1 \mod n}, \hdots, z_{j + n - 1 \mod 1})
    .. math::
        \forall 0 \leq i < n: |z_{j}| \geq |z_{i}|

    Note: Both inputs and outputs are in real format:
    .. math::
        \left(\Re(z_0) \quad \dots \quad \Re(z_{n-1}) \quad \Im(z_0) \quad \dots \quad \Im(z_{n-1}) \right) \in \mathbb{R}^{2n}
    """

    def __init__(self, ambient):
        """Initializes the layer with the given ambient dimensions.

        Args:
            ambient (np.array([n_ambient], np.int)): List of ambient dimensions.
        """
        super(ShiftCanonicalization, self).__init__()
        self.ambient = ambient
    

    def real_to_complex(self, inputs):
        """Helper function to convert cymetric input into complex form.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Input tensor of cymetric format.

        Returns:
            tf.tensor([batch_size, n_coords], tf.complex64):
            Input converted into complex form.
        """
        real_inputs, imag_inputs = tf.split(inputs, 2, axis=-1)
        complex_inputs = tf.complex(real_inputs, imag_inputs)
        return complex_inputs


    def call(self, inputs):

        """Apply shift canonicalization on the inputs tensor.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY.

        Returns:
            output (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY, with cooridnates shifted to the left such that the coordinate with the highest norm sits first.
        """
        complex_inputs = self.real_to_complex(inputs)
        splits_by_ambient = tf.split(complex_inputs, axis=-1, num_or_size_splits= [x + 1 for x in self.ambient])
        complex_outputs=[]
        for i in range(len(splits_by_ambient)):
            norms = tf.math.abs(splits_by_ambient[i])
            max = tf.expand_dims(tf.argmax(norms, axis=-1), axis=-1)
            idx_sort = max + tf.range(norms.shape[-1], dtype=tf.dtypes.int64)
            idx_sort = tf.math.floormod(idx_sort, norms.shape[-1])
            complex_outputs.append(tf.gather(splits_by_ambient[i], idx_sort, axis=-1, batch_dims=1))
        complex_outputs = tf.concat(complex_outputs, axis=-1)
        outputs = tf.concat([tf.math.real(complex_outputs), tf.math.imag(complex_outputs)], axis=-1)
        return outputs

    
class FreeRootScalingCanonicalization(tf.keras.layers.Layer):

    """Custom layer that performs a free m-root of unity scaling canonicalization.
    Namely given an input that consists of n complex coordinates then FreeRootsaclingScalingCanonicalization
    with degree=m multiplies the first coordinate with m-root 'r' that maximizes the real part, and the multiples the rest
    of the coordinate with consecutive power of 'r'.
    .. math::
        FreeRootScalingCanonicalization(z_0, z_1, \hdots, z_{n-1}) := (r z_0, r^2 z_1 \hdots, r^n z_{n-1} )
    .. math::
        \forall i \neq 1: \Re(r z_0) \geq \Re(r^i z_0) \and r^m = 1. 

    Note: Both inputs and outputs are in real format:
    .. math::
        \left(\Re(z_0) \quad \dots \quad \Re(z_{n-1}) \quad \Im(z_0) \quad \dots \quad \Im(z_{n-1}) \right) \in \mathbb{R}^{2n}
    """
    def __init__(self, degree):
        """Initializes the layer with the given degree.

        Args:
            degree: The degree the roots of unity used in scaling.
        """
        super(FreeRootScalingCanonicalization, self).__init__()
        self.degree = degree

    def real_to_complex(self, inputs):
        """Helper function to convert cymetric input into complex form.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Input tensor of cymetric format.

        Returns:
            tf.tensor([batch_size, n_coords], tf.complex64):
            Input converted into complex form.
        """
        real_inputs, imag_inputs = tf.split(inputs, 2, axis=-1)
        complex_inputs = tf.complex(real_inputs, imag_inputs)
        return complex_inputs


    def call(self, inputs):

        """Applies root scaling canonicalization on the input tensor.

        Args:
            inputs (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY.

        Returns:
            output (tf.tensor[batch_size, 2*n_coords], tf.float32): Points on the CICY.
        """
        complex_inputs = self.real_to_complex(inputs)
        first_coord = complex_inputs[:,0]
        roots = tf.constant([np.e**( 2j * np.pi * k/float(self.degree)) for k in range(self.degree)], tf.dtypes.complex64)
        first_coord = tf.einsum('b, i -> bi', first_coord, roots)
        reals = tf.math.real(first_coord)
        index_max = tf.expand_dims(tf.argmax(reals, axis=-1), axis=-1)
        chosen_roots = tf.gather(roots, index_max)
        chosen_roots = tf.math.pow(chosen_roots, tf.cast(tf.range(1, self.degree +1), dtype=tf.dtypes.complex64))
        complex_outputs = complex_inputs * chosen_roots
        outputs = tf.concat([tf.math.real(complex_outputs), tf.math.imag(complex_outputs)], axis=-1)
        return outputs