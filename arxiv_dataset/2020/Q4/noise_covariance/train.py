"""Noise covariance experiments on CIFAR10."""

import os

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from google3.third_party.deepmind.deepmind_research.noise_covariance import models

tf.app.flags.DEFINE_enum('dataset_name', 'cifar10', ['cifar10'],
                         'name of dataset.')
tf.app.flags.DEFINE_enum('matching', 'none',
                         ['none', 'lr_matching', 'batch_matching'],
                         'the matching scheme.')
tf.app.flags.DEFINE_enum('model_name', 'cnn', ['cnn', 'vgg'],
                         'name of the model.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size for training.')
tf.app.flags.DEFINE_integer('training_epochs', 1000,
                            'number of epochs for training.')
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          'learning rate for training.')
tf.app.flags.DEFINE_integer('eval_batch_size', 1000,
                            'batch size for evaluation.')
tf.app.flags.DEFINE_integer('tf_rand_seed', 321, 'random seed for tf.')
tf.app.flags.DEFINE_integer('np_rand_seed', 654, 'random seed for numpy.')
tf.app.flags.DEFINE_integer('num_checkpoint_epochs', 1,
                            'number of epochs for saving a checkpoint.')
tf.app.flags.DEFINE_integer('matching_batch_size', 128,
                            'the small batch size for batch_matching scheme.')
tf.app.flags.DEFINE_float('lr_matching_factor', 8.0,
                          'the matching factor for lr_matching scheme.')
tf.flags.DEFINE_string('checkpoint_dir', '/tmp/noise_covariance/',
                       'directory for saving checkpoints.')
FLAGS = tf.app.flags.FLAGS

# Dictionary for number of training data.
num_train_dict = dict(cifar10=50000)
# Dictionary for number of test data.
num_test_dict = dict(cifar10=10000)
# Dictionary for number of labels.
num_labels = dict(cifar10=10)


def get_data(dataset_name):
  """Return training and test data.

  Args:
    dataset_name: string, name of the dataset ('cifar10').
  Returns:
    train/test data.
  """

  [training_set, test_set], _ = tfds.load(
      name=dataset_name,
      split=[tfds.Split.TRAIN, tfds.Split.TEST],
      with_info=True,
      shuffle_files=False)
  training_set = tfds.as_numpy(training_set.batch(num_train_dict[dataset_name]))
  test_set = tfds.as_numpy(test_set.batch(num_test_dict[dataset_name]))
  for each in training_set:
    x_train = each['image']
    y_train = each['label']
  for each in test_set:
    x_test = each['image']
    y_test = each['label']

  return x_train, y_train, x_test, y_test


# if use matching, get the learning rate matching factor.
def get_matching_lr(
    matching, learning_rate, batch_size, lr_matching_factor=8.0,
    matching_batch_size=128):
  """Return learning rate of the matching part.

  Args:
    matching: matching method
    learning_rate: original learning rate
    batch_size: original batch size
    lr_matching_factor: used in learning rate matching
    matching_batch_size: used in batch matching

  Returns:
    Learning rate in the matching step.
  """
  if matching == 'lr_matching':
    return learning_rate * np.sqrt(0.5 * (lr_matching_factor - 1.0))
  elif matching == 'batch_matching':
    return learning_rate * np.sqrt(0.5 * (
        1 - float(matching_batch_size) / float(batch_size)))
  else:
    raise NotImplementedError


def random_flip(x):
  """Flip the input x horizontally with 50% probability."""
  if np.random.rand(1)[0] > 0.5:
    return np.fliplr(x)
  return x


def zero_pad_and_crop(img, amount=4):
  """Zero pad by `amount` zero pixels on each side then take a random crop.

  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.

  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
  padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2,
                         img.shape[2]))
  padded_img[amount:img.shape[0] + amount, amount:
             img.shape[1] + amount, :] = img
  top = np.random.randint(low=0, high=2 * amount)
  left = np.random.randint(low=0, high=2 * amount)
  new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
  return new_img


def data_augment(x):
  batch_size = x.shape[0]
  for i in range(batch_size):
    x[i, ...] = random_flip(zero_pad_and_crop(x[i, ...]))
  return x


def main(argv):
  del argv  # unused.
  tf.reset_default_graph()
  tf.disable_eager_execution()
  # set random seed.
  tf.set_random_seed(FLAGS.tf_rand_seed)
  np.random.seed(FLAGS.np_rand_seed)

  dataset_name = FLAGS.dataset_name
  n_labels = num_labels[dataset_name]
  num_train = num_train_dict[dataset_name]
  num_test = num_test_dict[dataset_name]

  matching = FLAGS.matching

  # load training and test data.
  print('--- Loading {} ---'.format(dataset_name))
  # print('loading dataset')
  x_train, y_train, x_test, y_test = get_data(dataset_name)

  # define the model.
  model_name = FLAGS.model_name
  if model_name == 'cnn':
    print('--- Building CNN model ---')
    model = models.CnnModel(n_labels)
  elif model_name == 'vgg':
    print('--- Building VGG model ---')
    model = models.VggModel(n_labels)
  else:
    raise NotImplementedError

  # num of batches to eval test data.
  eval_batch_size = FLAGS.eval_batch_size
  num_eval_batch = int(num_test / eval_batch_size)
  # num of batches to eval train data.
  num_eval_train_batch = int(num_train / eval_batch_size)
  batch_size = FLAGS.batch_size
  num_steps_per_epoch = int(num_train / batch_size)
  training_epochs = FLAGS.training_epochs

  # define training operations.
  global_step = tf.train.get_or_create_global_step()
  trainable_vars = tf.trainable_variables()
  grads = tf.gradients(model.total_loss, trainable_vars)
  learning_rate = FLAGS.learning_rate
  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
      model.total_loss, global_step=global_step)

  # define matching operations
  if matching != 'none':
    l_rate_match = get_matching_lr(matching, learning_rate, batch_size)
    trainable_vars_ph_0 = []
    trainable_vars_ph_1 = []
    for var in trainable_vars:
      trainable_vars_ph_0.append(tf.placeholder(tf.float32, shape=var.shape))
      trainable_vars_ph_1.append(tf.placeholder(tf.float32, shape=var.shape))
    matching_step = []
    for var, ph_0, ph_1 in zip(trainable_vars, trainable_vars_ph_0,
                               trainable_vars_ph_1):
      matching_step.append(tf.assign_add(var, l_rate_match * (ph_0 - ph_1)))

  # checkpoint saver
  saver = tf.train.Saver(max_to_keep=1)
  checkpoint_dir = FLAGS.checkpoint_dir

  # start training.
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # First, if checkpoint_dir does not exist, create a folder.
    if not tf.gfile.IsDirectory(checkpoint_dir):
      # checkpoint directory does not exist; create directory.
      tf.gfile.MakeDirs(checkpoint_dir)

    epoch = 0
    train_acc_record = []
    train_loss_record = []
    test_acc_record = []
    test_loss_record = []

    while epoch < training_epochs:
      # compute train/test acc/loss and log, add to records.
      # compute test acc and loss on whole dataset.
      test_acc = 0.0
      test_loss = 0.0
      for b_id in range(num_eval_batch):
        data_dict_eval = {
            model.x_input:
                x_test[b_id * eval_batch_size:(b_id + 1) * eval_batch_size,
                       ...],
            model.y_input:
                y_test[b_id * eval_batch_size:(b_id + 1) * eval_batch_size]
        }
        test_acc_batch, test_loss_batch = sess.run(
            [model.accuracy, model.total_loss], feed_dict=data_dict_eval)
        test_acc += test_acc_batch
        test_loss += test_loss_batch
      test_acc /= float(num_eval_batch)
      test_loss /= float(num_eval_batch)
      test_acc_record.append(test_acc)
      test_loss_record.append(test_loss)

      # compute train acc, loss, and full gradient on whole dataset.
      train_acc = 0.0
      train_loss = 0.0
      for b_id in range(num_eval_train_batch):
        data_dict_eval_train = {
            model.x_input:
                x_train[b_id * eval_batch_size:(b_id + 1) * eval_batch_size,
                        ...],
            model.y_input:
                y_train[b_id * eval_batch_size:(b_id + 1) * eval_batch_size]
        }
        train_acc_batch, train_loss_batch = sess.run(
            [model.accuracy, model.total_loss],
            feed_dict=data_dict_eval_train)
        train_acc += train_acc_batch
        train_loss += train_loss_batch
      train_acc /= float(num_eval_train_batch)
      train_loss /= float(num_eval_train_batch)
      train_acc_record.append(train_acc)
      train_loss_record.append(train_loss)
      print('--- epoch {}, train acc {:.4f}, test acc {:.4f} ---'.format(
          epoch, train_acc, test_acc))

      if (epoch + 1) % FLAGS.num_checkpoint_epochs == 0:
        # write a checkpoint.
        print('--- Saving checkpoint ---')
        # save checkpoints.
        saver.save(sess,
                   os.path.join(checkpoint_dir, 'model.ckpt'),
                   global_step=global_step)
        # save training and test records.
        print('--- Saving records ---')
        # save train/test records.
        np.save(checkpoint_dir + '/train_acc.npy', train_acc_record)
        np.save(checkpoint_dir + '/train_loss.npy', train_loss_record)
        np.save(checkpoint_dir + '/test_acc.npy', test_acc_record)
        np.save(checkpoint_dir + '/test_loss.npy', test_loss_record)

      # Actual training step
      for _ in range(num_steps_per_epoch):
        # sample a batch without replacement
        selected_index = np.random.choice(
            num_train, size=batch_size, replace=False)
        x_batch = x_train[selected_index, ...]
        x_aug = data_augment(x_batch)
        y_batch = y_train[selected_index]
        data_dict = {model.x_input: x_aug, model.y_input: y_batch}
        sess.run(train_step, feed_dict=data_dict)

        # If using matching schemes, do these after the gradient descent step.
        if matching == 'lr_matching':
          selected_index_0 = np.random.choice(num_train, size=batch_size,
                                              replace=True)
          selected_index_1 = np.random.choice(num_train, size=batch_size,
                                              replace=True)
        elif matching == 'batch_matching':
          selected_index_0 = np.random.choice(
              num_train, size=FLAGS.matching_batch_size, replace=True)
          selected_index_1 = np.random.choice(
              num_train, size=FLAGS.matching_batch_size, replace=True)
        if matching != 'none':
          x_batch_0 = x_train[selected_index_0, ...]
          x_batch_1 = x_train[selected_index_1, ...]
          x_aug_0 = data_augment(x_batch_0)
          x_aug_1 = data_augment(x_batch_1)
          y_batch_0 = y_train[selected_index_0]
          y_batch_1 = y_train[selected_index_1]
          data_dict_0 = {model.x_input: x_aug_0, model.y_input: y_batch_0}
          data_dict_1 = {model.x_input: x_aug_1, model.y_input: y_batch_1}

          # compute gradients and difference
          grads_val_0 = sess.run(grads, feed_dict=data_dict_0)
          grads_val_1 = sess.run(grads, feed_dict=data_dict_1)
          # add to variables
          grads_dict = {}
          for ph_0, ph_1, array_0, array_1 in zip(trainable_vars_ph_0,
                                                  trainable_vars_ph_1,
                                                  grads_val_0, grads_val_1):
            grads_dict[ph_0] = array_0
            grads_dict[ph_1] = array_1
          sess.run(matching_step, feed_dict=grads_dict)

      # This epoch finished, update the epoch counter.
      epoch += 1


if __name__ == '__main__':
  tf.app.run(main)
