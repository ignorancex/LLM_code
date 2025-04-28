import tensorflow as tf


class Upsilon(tf.keras.metrics.Metric):

    def __init__(self, name='upsilon', **kwargs):
        super(Upsilon, self).__init__(name=name, **kwargs)
        self.total_cm = self.add_weight(name='total_cm', shape=(2, 2), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrx(y_true, y_pred))
        return self.total_cm

    def reset_states(self):
        """
            This method is only here due to a bug in the
            tensorflow keras metric to reset states. It
            does not account for rank > 0 variables.
        """
        tf.keras.backend.set_value(self.total_cm, np.zeros((2, 2)))

    def confusion_matrx(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, 1)
        y_true = tf.argmax(y_true, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=2)
        return cm

    def result(self):
        cm_normed = self.total_cm/tf.math.reduce_sum(self.total_cm, 1)
        self.upsilon = tf.math.reduce_min(tf.linalg.tensor_diag_part(cm_normed))
        return self.upsilon
