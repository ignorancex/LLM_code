import numpy as np
import os
import tensorflow as tf

from PIL import Image

from model import Model
from utils.model_utils import batch_data

IMAGE_SIZE = 84
IMAGES_DIR = os.path.join('..', 'data', 'celeba', 'data', 'raw', 'img_align_celeba')

# this class is only for fedprox, which 
# requires a prox_term to compute loss
class ClientProxModel(Model):
    def __init__(self, seed, lr, mu, num_classes):
        self.num_classes = num_classes
        self.mu = mu
        super(ClientProxModel, self).__init__(seed, lr)

    def create_model(self):
        prox_term = tf.placeholder("float", [None, ])
        input_ph = tf.placeholder(
            tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
        out = input_ph
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        logits = tf.layers.dense(out, self.num_classes)
        label_ph = tf.placeholder(tf.int64, shape=(None,))
        l2_loss = self.mu * tf.reduce_sum(prox_term)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_ph,
            logits=logits) + l2_loss
        predictions = tf.argmax(logits, axis=-1)
        minimize_op = self.optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(
            tf.equal(label_ph, tf.argmax(input=logits, axis=1)))
        pred_ops = tf.argmax(input=logits, axis=1)
        return input_ph, label_ph, minimize_op, eval_metric_ops, tf.math.reduce_mean(loss), pred_ops, prox_term

    def process_x(self, raw_x_batch):
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        return raw_y_batch

    def _load_image(self, img_name):
        img = Image.open(os.path.join(IMAGES_DIR, img_name))
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('RGB')
        return np.array(img)
    
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