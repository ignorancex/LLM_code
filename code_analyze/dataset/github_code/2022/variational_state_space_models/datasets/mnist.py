import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.video import moving_sequence
import collections

MnistSequence = collections.namedtuple('MnistSequence', ['img'])
MnistEntry = collections.namedtuple('MnistEntry', ['img'])

def load_still(train_batch_size, test_batch_size, width, height):
    train_ds, test_ds = tfds.load("mnist", split=['train', 'test'], as_supervised=True, shuffle_files=True)

    def reformat(x, _):
        image = tf.image.resize(x, (height, width))
        image = tf.cast(image, tf.float32)
        image = (image / 255)*2 - 1
        return MnistEntry(image)

    train_ds = train_ds.map(reformat).cache().repeat()
    test_ds = test_ds.map(reformat).cache().repeat()

    train_ds = train_ds.shuffle(10*train_batch_size, seed=52182673124).batch(train_batch_size)
    test_ds = test_ds.batch(test_batch_size)

    return iter(tfds.as_numpy(train_ds)), iter(tfds.as_numpy(test_ds))

def load_still_fashion(train_batch_size, test_batch_size, width, height):
    train_ds, test_ds = tfds.load("fashion_mnist", split=['train', 'test'], as_supervised=True, shuffle_files=True)

    def reformat(x, _):
        image = tf.cast(x, tf.float32)
        image = image / 255
        image = tf.image.resize(image, (height, width))
        return MnistEntry(image)

    train_ds = train_ds.map(reformat).cache().repeat()
    test_ds = test_ds.map(reformat).cache().repeat()

    train_ds = train_ds.shuffle(10*train_batch_size, seed=52182673124).batch(train_batch_size)
    test_ds = test_ds.batch(test_batch_size)

    return iter(tfds.as_numpy(train_ds)), iter(tfds.as_numpy(test_ds))

def load_traj(batch_size, T=5, N=None):
    test_ds = tfds.load('moving_mnist', split=('test'), shuffle_files=False)
    test_ds = test_ds.map(lambda s: MnistSequence(tf.cast(s['image_sequence'][:T], tf.float32)/255)).repeat().batch(batch_size)

    mnist_ds = tfds.load("mnist", split=tfds.Split.TRAIN, as_supervised=True, shuffle_files=False)
    mnist_ds = mnist_ds.cache().repeat().shuffle(1024, seed=548612312321)
    def map_fn(image, label):
        sequence = moving_sequence.image_as_moving_sequence(image, sequence_length=T)
        return sequence.image_sequence
    ds = mnist_ds.map(map_fn).batch(2).map(lambda x: MnistSequence(tf.cast(tf.reduce_max(x, axis=0), tf.float32)/255))
    if N is not None:
        ds = ds.take(N).repeat()
    ds = ds.shuffle(10*batch_size, seed=12586213132).batch(batch_size)
    return ds, tfds.as_numpy(test_ds))
