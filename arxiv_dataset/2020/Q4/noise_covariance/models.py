"""Models: a small CNN and VGG 19."""

import tensorflow.compat.v1 as tf

MEANS = [p / 255.0 for p in [125.3, 123.0, 113.9]]
STDS = [p / 255.0 for p in [63.0, 62.1, 66.7]]
WEIGHT_DECAY = 1e-4


# Define conv layers, fully connected layers, and pooling layers.
def conv_layer(x, kernel_shape, bias_shape):
  """Convolutional-ReLU Layer."""
  weights = tf.get_variable(
      'weights',
      kernel_shape,
      initializer=tf.initializers.he_normal())
  tf.add_to_collection(weights, 'all_weights')
  biases = tf.get_variable(
      'biases', bias_shape, initializer=tf.constant_initializer())
  output = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
  return tf.nn.relu(output + biases)


def fc_layer(x, n_in, n_out, activation_fn=tf.nn.relu):
  """Fully Connected Layer."""
  weights = tf.get_variable(
      'weights', [n_in, n_out],
      initializer=tf.initializers.random_normal(mean=0.0, stddev=0.01))
  tf.add_to_collection(weights, 'all_weights')
  biases = tf.get_variable(
      'biases', [n_out], initializer=tf.constant_initializer())
  output = tf.nn.xw_plus_b(x, weights, biases)
  if activation_fn is not None:
    output = activation_fn(output)
  return output


def max_pool_layer(x, image_s):
  """2x2 Max pooling with stride 2."""
  image_s = int(image_s / 2)
  x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
  return x, image_s


# Define the models.
class CnnModel(object):
  """A small CNN model."""

  def __init__(self, n_labels):
    self.n_labels = n_labels
    self._build_model()

  def _build_model(self):
    """Build CNN model."""
    img_s = 32
    self.x_input = tf.placeholder(
        tf.float32, shape=[None, 32, 32, 3], name='image')
    self.y_input = tf.placeholder(tf.int64, shape=None, name='label')

    # standardize input data.
    x_input = self.x_input / 255.0
    x_input = (x_input - MEANS) / STDS

    cin = 3  # Channel In
    cout = 32  # Channel Out
    with tf.variable_scope('conv1'):
      conv1 = conv_layer(x_input, [3, 3, cin, cout], [img_s, img_s, cout])
    cin = cout
    cout = 64
    with tf.variable_scope('conv2'):
      conv2 = conv_layer(conv1, [3, 3, cin, cout], [img_s, img_s, cout])
      pool1, img_s = max_pool_layer(conv2, img_s)
    cin = cout
    cout = 128
    with tf.variable_scope('conv3'):
      conv3 = conv_layer(pool1, [3, 3, cin, cout], [img_s, img_s, cout])
    cin = cout
    with tf.variable_scope('conv4'):
      conv4 = conv_layer(conv3, [3, 3, cin, cout], [img_s, img_s, cout])
      pool2, img_s = max_pool_layer(conv4, img_s)
    cout = 256
    with tf.variable_scope('conv5'):
      conv5 = conv_layer(pool2, [3, 3, cin, cout], [img_s, img_s, cout])
    cin = cout
    with tf.variable_scope('conv6'):
      conv6 = conv_layer(conv5, [3, 3, cin, cout], [img_s, img_s, cout])
      pool3, img_s = max_pool_layer(conv6, img_s)

    with tf.variable_scope('fc1'):
      n_in = img_s * img_s * cout
      n_out = 1024
      pool3_1d = tf.reshape(pool3, [-1, n_in])
      fc1 = fc_layer(pool3_1d, n_in, n_out)
    with tf.variable_scope('fc2'):
      n_in = n_out
      n_out = 512
      fc2 = fc_layer(fc1, n_in, n_out)
    with tf.variable_scope('fc3'):
      n_in = n_out
      n_out = self.n_labels
      self.logits = fc_layer(fc2, n_in, n_out, activation_fn=None)

    with tf.variable_scope('weights_norm'):
      weights_norm = tf.reduce_sum(
          input_tensor=WEIGHT_DECAY * tf.stack(
              [tf.nn.l2_loss(i) for i in tf.get_collection('all_weights')]),
          name='weights_norm')
    tf.add_to_collection('losses', weights_norm)

    with tf.variable_scope('cross_entropy'):
      cross_entropy = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=self.y_input, logits=self.logits))
    tf.add_to_collection('losses', cross_entropy)

    self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    correct_prediction = tf.equal(self.y_input, tf.argmax(self.logits, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


class VggModel(object):
  """VGG 19 model."""

  def __init__(self, n_labels):
    self.n_labels = n_labels
    self._build_model()

  def _build_model(self):
    """Build VGG model."""
    img_s = 32
    self.x_input = tf.placeholder(
        tf.float32, shape=[None, 32, 32, 3], name='image')
    self.y_input = tf.placeholder(tf.int64, shape=None, name='label')

    # standardize input data.
    x_input = self.x_input / 255.0
    x_input = (x_input - MEANS) / STDS

    cin = 3  # Channel In
    cout = 64  # Channel Out
    with tf.variable_scope('conv1'):
      conv1 = conv_layer(x_input, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv2'):
      conv2 = conv_layer(conv1, [3, 3, cout, cout], [img_s, img_s, cout])
      pool1, img_s = max_pool_layer(conv2, img_s)

    cin = cout
    cout = 128
    with tf.variable_scope('conv3'):
      conv3 = conv_layer(pool1, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv4'):
      conv4 = conv_layer(conv3, [3, 3, cout, cout], [img_s, img_s, cout])
      pool2, img_s = max_pool_layer(conv4, img_s)

    cin = cout
    cout = 256
    with tf.variable_scope('conv5'):
      conv5 = conv_layer(pool2, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv6'):
      conv6 = conv_layer(conv5, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv7'):
      conv7 = conv_layer(conv6, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv8'):
      conv8 = conv_layer(conv7, [3, 3, cout, cout], [img_s, img_s, cout])
      pool3, img_s = max_pool_layer(conv8, img_s)

    cin = cout
    cout = 512
    with tf.variable_scope('conv9'):
      conv9 = conv_layer(pool3, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv10'):
      conv10 = conv_layer(conv9, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv11'):
      conv11 = conv_layer(conv10, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv12'):
      conv12 = conv_layer(conv11, [3, 3, cout, cout], [img_s, img_s, cout])
      pool4, img_s = max_pool_layer(conv12, img_s)

    cin = cout
    cout = 512
    with tf.variable_scope('conv13'):
      conv13 = conv_layer(pool4, [3, 3, cin, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv14'):
      conv14 = conv_layer(conv13, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv15'):
      conv15 = conv_layer(conv14, [3, 3, cout, cout], [img_s, img_s, cout])
    with tf.variable_scope('conv16'):
      conv16 = conv_layer(conv15, [3, 3, cout, cout], [img_s, img_s, cout])
      pool5, img_s = max_pool_layer(conv16, img_s)

    with tf.variable_scope('fc1'):
      n_in = img_s * img_s * cout
      n_out = 512
      pool5_1d = tf.reshape(pool5, [-1, n_in])
      fc1 = fc_layer(pool5_1d, n_in, n_out)
    with tf.variable_scope('fc2'):
      n_in = n_out
      n_out = 512
      fc2 = fc_layer(fc1, n_in, n_out)
    with tf.variable_scope('fc3'):
      n_in = n_out
      n_out = self.n_labels
      self.logits = fc_layer(fc2, n_in, n_out, activation_fn=None)

    with tf.variable_scope('weights_norm'):
      weights_norm = tf.reduce_sum(
          input_tensor=WEIGHT_DECAY * tf.stack(
              [tf.nn.l2_loss(i) for i in tf.get_collection('all_weights')]),
          name='weights_norm')
    tf.add_to_collection('losses', weights_norm)

    with tf.variable_scope('cross_entropy'):
      cross_entropy = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=self.y_input, logits=self.logits))
    tf.add_to_collection('losses', cross_entropy)

    self.total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    correct_prediction = tf.equal(self.y_input, tf.argmax(self.logits, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
