import tensorflow as tf
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], "GPU")


def load_data(dset_name: str, batch_size: int, seed: int, reshape=True):
    """
    Load the dataset and split it into training and testing sets.
    """
    tf.random.set_seed(seed)  # for shuffling
    (train_data, eval_data, test_data), ds_info = tfds.load(
        dset_name,
        split=["train[:80%]", "train[80%:]", "test"],  # 80% train, 20% validation
        as_supervised=True,
        with_info=True,
    )

    # Preprocess the data
    def preprocess(image, label):
        # put channel in the first dimension
        if len(image.shape) == 3:
            image = tf.transpose(image, perm=[2, 0, 1])
        if reshape:
            image = (
                tf.cast(
                    tf.reshape(image, shape=[tf.reduce_prod(image.shape)]), tf.float32
                )
            )
        else:
            image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image, label

    train_data = (
        train_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .shuffle(buffer_size=10000)
        .prefetch(tf.data.AUTOTUNE)
    )
    eval_data = (
        eval_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_data = (
        test_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_data, eval_data, test_data, ds_info
