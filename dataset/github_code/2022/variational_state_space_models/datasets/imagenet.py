import collections
import tensorflow as tf
import tensorflow_datasets as tfds

ImageNetEntry = collections.namedtuple('ImageNetEntry', ['img'])

def load_still(train_batch_size, test_batch_size, width, height):
    import resource
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
    train_ds, test_ds = tfds.load(f"imagenet_resized/{width}x{height}", split=['train', 'validation'], shuffle_files=True)

    def reformat(s):
        image = tf.cast(s['image'], tf.float32)
        image = image / 255
        return ImageNetEntry(image)

    train_ds = train_ds.map(reformat).repeat()
    test_ds = test_ds.map(reformat).repeat()
    
    train_ds = train_ds.shuffle(10*train_batch_size).batch(train_batch_size)
    test_ds = test_ds.batch(test_batch_size)

    return iter(tfds.as_numpy(train_ds)), iter(tfds.as_numpy(test_ds))