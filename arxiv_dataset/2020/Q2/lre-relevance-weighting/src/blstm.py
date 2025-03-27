from os.path import join

import tensorflow as tf
import numpy as np
import random
import datetime

import config.blstm_config as config
from utils.nn_utils import attention


print('Loading Data.')
train_data = np.load(join(config.generate_config['output_loc'], 'train_15s_data.npy'))
train_labels = np.load(join(config.generate_config['output_loc'], 'train_15s_labels.npy'))
dev_data_5s = np.load(join(config.generate_config['output_loc'], 'dev_5s_data.npy'))
dev_labels_5s = np.load(join(config.generate_config['output_loc'], 'dev_5s_labels.npy'))
dev_data_15s = np.load(join(config.generate_config['output_loc'], 'dev_15s_data.npy'))
dev_labels_10s = np.load(join(config.generate_config['output_loc'], 'dev_15s_labels.npy'))


def batch_iter(data, labels):
    data_size = len(data)
    for e in range(config.train_conf['num_epochs']):
        for b in range(data_size / config.train_conf['train_batch_size']):
            ln = random.sample(range(data_size), config.train_conf['train_batch_size'])
            batch = data[ln]
            batch_labels = labels[ln]
            yield batch, batch_labels


def hot_encode(a, n_classes):
    b = np.zeros((a.size, n_classes))
    b[np.arange(a.size), a] = 1
    return b


def model(x_, y_, kp):
    in_x = tf.transpose(x_, perm=[0, 2, 1])

    with tf.variable_scope('first_layer'):
        cell_fw = tf.contrib.rnn.LSTMCell(config.model_conf['blstm_layer1_units'], use_peepholes=True)
        cell_bw = tf.contrib.rnn.LSTMCell(config.model_conf['blstm_layer1_units'], use_peepholes=True)
        lstm_outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, in_x, dtype=tf.float32)
        lstm_outputs = tf.concat(lstm_outputs, 2)

    with tf.variable_scope('second_layer'):
        cell_fw1 = tf.contrib.rnn.LSTMCell(config.model_conf['blstm_layer2_units'], use_peepholes=True)
        cell_bw1 = tf.contrib.rnn.LSTMCell(config.model_conf['blstm_layer2_units'], use_peepholes=True)
        lstm_outputs, state1 = tf.nn.bidirectional_dynamic_rnn(cell_fw1, cell_bw1, lstm_outputs, dtype=tf.float32)
        lstm_outputs = tf.concat(lstm_outputs, 2)

    attn, alphas = attention(lstm_outputs, 2 * config.model_conf['blstm_layer2_units'])
    w_dense = tf.Variable(tf.truncated_normal([2 * config.model_conf['blstm_layer2_units'],
                                               config.model_conf['dense_dim']], stddev=0.1))
    b_dense = tf.Variable(tf.random_normal([config.model_conf['dense_dim']]))
    h_dense = tf.nn.relu(tf.add(tf.matmul(attn, w_dense), b_dense))
    h_dense_drop = tf.nn.dropout(h_dense, kp)

    W_out = tf.Variable(
        tf.truncated_normal([config.model_conf['dense_dim'], len(config.label_dict.keys())], stddev=0.1))
    b_out = tf.Variable(tf.random_normal([len(config.label_dict.keys())]))
    logits = tf.add(tf.matmul(h_dense_drop, W_out), b_out)
    pred = tf.nn.softmax(logits, name='predictions')
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_))
    correct_pred = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return loss, pred, acc, alphas


print('Creating Model')
x_in = tf.placeholder("float32", [None, 500, None])
y_in = tf.placeholder("float", [None, ])
keep_prob = tf.placeholder_with_default(1.0, shape=())
blstm_loss, predictions, accuracy, attn_weights = model(x_in, y_in, keep_prob)
global_step = tf.Variable(0, name="global_step", trainable=False)
learning_rate = tf.train.exponential_decay(config.model_conf['start_learning_rate'], global_step,
                                           config.model_conf['decay_steps'], config.model_conf['decay_rate'],
                                           staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
grads_and_vars = optimizer.compute_gradients(blstm_loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
summary_writer = tf.summary.FileWriter(config.summary_conf["checkpoint_dir"])

print("Training started.")
max_val_acc_5s = 0.0
max_val_acc_10s = 0.0
max_tot_acc = 0.0
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    max_test_acc = 0.0
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    summary = tf.Summary()
    i = 0
    for batch_x, batch_y in next(batch_iter(train_data, train_labels)):
        _, step, train_loss, train_acc = sess.run([train_op, global_step, blstm_loss, accuracy],
                                                  feed_dict={x_in: batch_x, y_in: batch_y, keep_prob: 0.8})
        summary.value.add(tag='Train accuracy mini batch', simple_value=float(train_acc))
        summary.value.add(tag='Train loss mini batch', simple_value=float(train_loss))
        time_str = datetime.datetime.now().isoformat()
        if i % config.model_conf['print_checkpoint'] == 0:
            print("{}: step {}, mini batch loss {:g}, mini batch acc {:g}".format(time_str, step, train_loss,
                                                                                  train_acc))
        if i % config.model_conf['save_checkpoint'] == 0:
            dev_acc_5s = sess.run([accuracy], feed_dict={x_in: dev_data_5s, y_in: dev_labels_5s})
            dev_acc_10s = sess.run([accuracy], feed_dict={x_in: dev_data_15s, y_in: dev_labels_10s})
            tot_dev_data = len(dev_data_5s) + len(dev_data_15s)
            tot_acc = (dev_acc_5s[0] * len(dev_data_5s) + dev_acc_10s[0] * len(dev_data_15s)) / tot_dev_data * 1.0
            print('Accuracy on dev data, 5s:' + str(dev_acc_5s) + ' 15s:' + str(dev_acc_10s) + 'tot:' + str(tot_acc))
            summary.value.add(tag='Validation 5s accuracy', simple_value=dev_acc_5s)
            summary.value.add(tag='Validation 15s accuracy', simple_value=dev_acc_10s)
            summary.value.add(tag='Validation net accuracy', simple_value=tot_acc)
            summary_writer.flush()
            if tot_acc > max_tot_acc:
                max_val_acc_5s = dev_acc_5s
                max_val_acc_10s = dev_acc_10s
                max_tot_acc = tot_acc
                best_model_path = join(config.summary_conf['model_directory'], config.summary_conf['model_name'])
                saver.save(sess, best_model_path, global_step=global_step, latest_filename='latest_model')
        i += 1
