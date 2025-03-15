'''
This code is due to Yutong Deng (@yutongD), Yingtong Dou (@YingtongDou) and UIC BDSC Lab
DGFraud (A Deep Graph-based Toolbox for Fraud Detection)
https://github.com/safe-graph/DGFraud

FdGars ('FdGars: Fraudster Detection via Graph Convolutional Networks in Online App Review System')

Parameters:
    nodes: total nodes number
    gcn_output1: the first gcn layer unit number
    gcn_output2: the second gcn layer unit number
    embedding: node feature dim
    encoding: nodes representation dim (predict class dim)
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../..')))
import tensorflow as tf
from base_models.models import GCN
from algorithms.base_algorithm import Algorithm
from utils import utils


class FdGars(Algorithm):

    def __init__(self,
                 session,
                 nodes,
                 class_size,
                 gcn_output1,
                 gcn_output2,
                 meta,
                 embedding,
                 encoding):
        self.nodes = nodes
        self.meta = meta
        self.class_size = class_size
        self.gcn_output1 = gcn_output1
        self.embedding = embedding
        self.encoding = encoding

        self.placeholders = {'a': tf.placeholder(tf.float32, [self.meta, self.nodes, self.nodes], 'adj'),
                             'x': tf.placeholder(tf.float32, [self.nodes, self.embedding], 'nxf'),
                             'batch_index': tf.placeholder(tf.int32, [None], 'index'),
                             't': tf.placeholder(tf.float32, [None, self.class_size], 'labels'),
                             'lr': tf.placeholder(tf.float32, [], 'learning_rate'),
                             'mom': tf.placeholder(tf.float32, [], 'momentum'),
                             'num_features_nonzero': tf.placeholder(tf.int32)}

        loss, probabilities = self.forward_propagation()
        self.loss, self.probabilities = loss, probabilities
        self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01),
                                                         tf.trainable_variables())

        self.pred = tf.one_hot(tf.argmax(self.probabilities, 1), class_size)
        print(self.pred.shape)
        self.correct_prediction = tf.equal(tf.argmax(self.probabilities, 1), tf.argmax(self.placeholders['t'], 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        print('Forward propagation finished.')

        self.sess = session
        self.optimizer = tf.train.AdamOptimizer(self.placeholders['lr'])
        gradients = self.optimizer.compute_gradients(self.loss + self.l2)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = self.optimizer.apply_gradients(capped_gradients)
        self.init = tf.global_variables_initializer()
        print('Backward propagation finished.')

    def forward_propagation(self):
        with tf.variable_scope('gcn'):
            gcn_emb = []
            for i in range(self.meta):
                gcn_out = tf.reshape(GCN(self.placeholders, self.gcn_output1, self.embedding,
                                         self.encoding, index=i).embedding(), [1, self.nodes * self.encoding])
                gcn_emb.append(gcn_out)
            gcn_emb = tf.concat(gcn_emb, 0)
            gcn_emb = tf.reshape(gcn_emb, [self.nodes, self.encoding])
            print('GCN embedding over!')

        with tf.variable_scope('classification'):
            batch_data = tf.matmul(tf.one_hot(self.placeholders['batch_index'], self.nodes), gcn_emb)
            logits = tf.nn.softmax(batch_data)
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.placeholders['t'], logits=logits)

        return loss, tf.nn.sigmoid(logits)

    def train(self, x, a, t, b, learning_rate=1e-2, momentum=0.9):
        feed_dict = utils.construct_feed_dict(x, a, t, b, learning_rate, momentum, self.placeholders)
        outs = self.sess.run(
            [self.train_op, self.loss, self.accuracy, self.pred, self.probabilities],
            feed_dict=feed_dict)
        loss = outs[1]
        acc = outs[2]
        pred = outs[3]
        prob = outs[4]
        return loss, acc, pred, prob

    def test(self, x, a, t, b, learning_rate=1e-2, momentum=0.9):
        feed_dict = utils.construct_feed_dict(x, a, t, b, learning_rate, momentum, self.placeholders)
        acc, pred, probabilities, tags = self.sess.run(
            [self.accuracy, self.pred, self.probabilities, self.correct_prediction],
            feed_dict=feed_dict)
        return acc, pred, probabilities, tags
