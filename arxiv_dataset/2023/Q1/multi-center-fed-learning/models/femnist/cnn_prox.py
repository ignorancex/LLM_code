import tensorflow as tf

from model import Model
import numpy as np
from utils.model_utils import batch_data

IMAGE_SIZE = 28


class ClientProxModel(Model):
    def __init__(self, seed, lr, mu, num_classes):
        self.num_classes = num_classes
        self.mu = mu
        super(ClientProxModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        prox_term = tf.placeholder("float", [None, ]) 
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=248, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        
        l2_loss = self.mu * tf.reduce_sum(prox_term)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + l2_loss
        train_op = self.optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
#         grads_and_vars = self.optimizer.compute_gradients(loss)
        # TODO: Confirm that opt initialized once is ok?
#         train_op = self.optimizer.apply_gradients(
#             grads_and_vars,
#             global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        pred_ops = tf.argmax(input=logits, axis=1)
        return features, labels, train_op, eval_metric_ops, loss, pred_ops, prox_term

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

    def run_epoch(self, data, batch_size):
        ravel = lambda x,y: np.ravel(x) - np.ravel(y)
        sumall = lambda x, y: np.sum(ravel(x, y) ** 2)
        
        for batched_x, batched_y in batch_data(data, batch_size):
            
            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)
            
            global_w = self.gl_ws
            local_w = self.get_params()
            first = True
            tup_diff = []
            for local_v, gl_v in zip(local_w, global_w):
                if first:
                    first = False
                else:
                    tup_diff.append(sumall(local_v, gl_v))
            
            with self.graph.as_default():
                self.sess.run(self.train_op,
                    feed_dict={
                        self.prox_term: tup_diff,
                        self.features: input_data,
                        self.labels: target_data
                    })

                
    def fix_global_ws(self, ws):
        self.gl_ws = ws    