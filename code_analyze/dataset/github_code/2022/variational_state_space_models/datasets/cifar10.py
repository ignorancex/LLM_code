import collections
import tensorflow as tf
import tensorflow_datasets as tfds

CifarEntry = collections.namedtuple('CifarEntry', ['img'])

def load_still(train_batch_size, test_batch_size, width, height):
    train_ds, test_ds = tfds.load("cifar10", split=['train', 'test'], shuffle_files=True)

    def reformat(s):
        image = tf.cast(s['image'], tf.float32)
        image = tf.image.resize(image, (height, width))
        image = (image / 255)*2 - 1
        return CifarEntry(image)

    train_ds = train_ds.map(reformat).cache().repeat()
    test_ds = test_ds.map(reformat).cache().repeat()
    
    train_ds = train_ds.shuffle(10*train_batch_size).batch(train_batch_size)
    test_ds = test_ds.batch(test_batch_size)

    return iter(tfds.as_numpy(train_ds)), iter(tfds.as_numpy(test_ds))