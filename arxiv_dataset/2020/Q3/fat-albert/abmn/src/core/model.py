import re
import tensorflow as tf
initializer = "xavier"


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/', x.op.name)
    tf.compat.v1.summary.histogram(tensor_name + '/activations', x)
    tf.compat.v1.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


# Dropout operation
def dropout(x, training, rate):
    print("Dropout: " + str(training))
    res = tf.compat.v1.layers.dropout(x, rate=rate, training=training)
    return res


# Preparing Layer operation, projects embeddings to lower dimension
def prep_embedding(x, hidden_size):
    left = tf.compat.v1.layers.dense(x, hidden_size, activation=tf.nn.sigmoid, kernel_initializer=get_initializer(initializer),
                           bias_initializer=tf.compat.v1.constant_initializer(value=0))
    right = tf.compat.v1.layers.dense(x, hidden_size, activation=tf.nn.tanh, kernel_initializer=get_initializer(initializer),
                            bias_initializer=tf.compat.v1.constant_initializer(value=0))
    mult = tf.multiply(left, right)
    return mult


# Attention Layer operation, input x2 is weighted with input x1
def prep_attention(x1, x2, hidden_size):
    # original approach by Wang and Jiang, but sometimes leads to bad performance here
    # left = tf.layers.dense(x1, hidden_size)
    # m = tf.matmul(left, x2, transpose_b=True)
    m = tf.matmul(x1, x2, transpose_b=True)
    g = tf.nn.softmax(tf.transpose(a=m))
    h = tf.matmul(g, x1)
    return h


# SUBMULT comparison operation
def compare_submult(x1, x2, hidden_size):
    sub = tf.subtract(x1, x2)
    pow = tf.multiply(sub, sub)
    mult = tf.multiply(x1, x2)
    con = tf.concat([pow, mult], 1)
    nn = tf.compat.v1.layers.dense(con, hidden_size, activation=tf.nn.relu, kernel_initializer=get_initializer(initializer),
                         bias_initializer=tf.compat.v1.constant_initializer())
    return nn


# MULT comparison operation
def compare_mult(x1, x2):
    return tf.multiply(x1, x2)


# CNN layer operation, returns also attention weighted word sequences for visualization
# Warning: Only use padding=SAME at the moment, otherwise attention visualization will throw an error.
# param filter_visualization obsolete, aggregate all filters now
def cnn(x, filter_sizes, hidden_size, filter_visualization=3, padding='SAME'):
    """

    :param x: input of size [first_dim x num_words x 2*hidden_size]
        for example in the 1st stage we have:
        q: [1 x num_words x 2* hidden_size]
        a: [num_answers x num_words x 2* hidden_size]
        p: [num_sentences x num_words x 2* hidden_size]
    :param filter_sizes:
    :param hidden_size:
    :param filter_visualization:
    :param padding: SAME means the 2nd dimension of the output of the conv1d is the same as its input (num_words)
    :return: concatenated 1dconv outputs for each filter in filter_sizes
            con dimensions: [first_dim x hidden_size * num_filter_sizes]
    """
    merge = []
    attention_vis = []
    for filter_size in filter_sizes:
        # conv_branch: [first_dim x num_words x 1* hidden_size]
        conv_branch = tf.compat.v1.layers.conv1d(
            inputs=x,
            # use as many filters as the hidden size
            filters=hidden_size,
            kernel_size=[filter_size],
            use_bias=True,
            activation=tf.nn.relu,
            trainable=True,
            padding=padding,
            kernel_initializer=get_initializer(initializer),
            bias_initializer=tf.compat.v1.constant_initializer(),
            name='conv_' + str(filter_size)
        )
        attention_vis.append(conv_branch)
        # pool over the words to obtain: [first_dim x 1* hidden_size]
        pool_branch = tf.reduce_max(input_tensor=conv_branch, axis=1)
        merge.append(pool_branch)

    # num_filter_sizes * [first_dim x hidden_size] -> [first_dim x hidden_size * num_filter_sizes]
    con = tf.concat(merge, axis=1)

    attention_vis = tf.stack(attention_vis, axis=0)
    attention_vis = tf.reduce_mean(input_tensor=attention_vis, axis=0)

    return con, attention_vis


def lstm(inputs, hidden_size, mode='lstm'):
    """
    RNN part of the aggregation function
    :param inputs:
    :param hidden_size:
    :param mode: [lstm: unidirectional lstm, gru: unidirectional gru, bi: bidirectional lstm]
                 in the paper, we only report results with the unidirectional lstm (default setting here)
    :return:
    """
    # unidirectional lstm or gru
    if mode == 'lstm' or mode == 'gru':
        cell = get_cell(mode, hidden_size)
        output, _ = tf.compat.v1.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        output = tf.reduce_max(input_tensor=output, axis=1)
        print("MODE: Reduce unidirectional " + mode)

    # bidirectional lstm
    else:
        cell1 = get_cell('lstm', hidden_size)
        cell2 = get_cell('lstm', hidden_size)
        output, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell1, cell2, inputs, dtype=tf.float32)
        output_fw, output_bw = output
        if mode == "bi":
            output = tf.concat([output_fw, output_bw], 2)
            output = tf.reduce_max(input_tensor=output, axis=1)
            print("MODE: Reduce Bidirectional " + mode)

    return output


# Prediction layer, computes final scores for answer candidates
def softmax_prep(x, hidden_size):
    # [num_answers x Y] -> [num_answers x hidden_size]
    inner = tf.compat.v1.layers.dense(x, hidden_size, activation=tf.nn.tanh, kernel_initializer=get_initializer(initializer),
                            bias_initializer=tf.compat.v1.constant_initializer(0))
    # [num_answers x Y] -> [num_answers x 1]
    lin = tf.compat.v1.layers.dense(inner, 1, kernel_initializer=get_initializer(initializer),
                          bias_initializer=tf.compat.v1.constant_initializer(0))
    return lin


# return global step for model savers
def get_global_step():
    with tf.device('/cpu:0'):
        gs = tf.compat.v1.get_variable('global_step', initializer=tf.constant(0), dtype=tf.int32)
    return gs


# Parameter update operation with TensorBoard logging
def update_params(total_loss, global_step, opt_name, learning_rate):
    optimizer = get_optimizer(opt_name, learning_rate)
    grads = optimizer.compute_gradients(total_loss)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    for var in tf.compat.v1.trainable_variables():
        tf.compat.v1.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.compat.v1.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op


# loss function for hinge loss
def compute_hinge_loss_sample(data):
    logits, labels = data
    H = tf.reduce_max(input_tensor=logits * (1 - labels), axis=0)
    L = tf.nn.relu((1 - logits + H) * labels)
    final_loss = tf.reduce_mean(input_tensor=tf.reduce_max(input_tensor=L, axis=0))
    return final_loss


# loss function for cross entropy loss
def compute_entropy_loss_sample(data):
    logits, labels = data
    final_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(labels), logits=logits)
    return final_loss


def compute_batch_mean_loss(logits, labels, loss_func):
    if loss_func == "hinge":
        print("Loss: HINGE")
        loss = tf.map_fn(compute_hinge_loss_sample, elems=[logits, labels], dtype=tf.float32)
    elif loss_func == "entropy":
        print("Loss: ENTROPY")
        loss = tf.map_fn(compute_entropy_loss_sample, elems=[logits, labels], dtype=tf.float32)
    else:
        print("Loss: ENTROPY")
        loss = tf.map_fn(compute_entropy_loss_sample, elems=[logits, labels], dtype=tf.float32)

    # apply L2 regularization (only for weights, not for bias)
    vars = tf.compat.v1.trainable_variables()
    vars_f = [v for v in vars if 'embedding' not in v.name]
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars_f
                       if 'bias' not in v.name]) * 0.0001
    loss_mean = tf.reduce_mean(input_tensor=loss + lossL2, name='batch_loss')
    return loss_mean


# accuracy computation op
def compute_accuracies(logits, labels, dim):
    probabs = tf.nn.softmax(logits)
    l_cast = tf.cast(labels, dtype=tf.int64)
    correct_prediction = tf.equal(tf.argmax(input=probabs, axis=dim), tf.argmax(input=l_cast, axis=dim))
    accuracy = tf.cast(correct_prediction, tf.float32)
    return accuracy


# probability computation op with softmax
def compute_probabilities(logits):
    # print("logits %s" % str(logits))
    # logits = tf.Print(logits, [logits], message="logits to compute softmax")
    probs = tf.nn.softmax(logits)
    return probs


# casting all labels > 0 to 1 (needed only for Wikiqa with multiple correct answers)
def cast_labels(labels):
    zero = tf.cast(0.0, dtype=tf.float32)
    l_cast = tf.cast(labels, dtype=tf.float32)
    zeros = tf.zeros_like(labels)
    condition = tf.greater(l_cast, zero)
    res = tf.compat.v1.where(condition, tf.ones_like(labels), zeros)
    return res


# get weight initializer op
def get_initializer(name):
    if (name == "variance"):
        return tf.compat.v1.variance_scaling_initializer()
    elif (name == "normal"):
        return tf.compat.v1.random_normal_initializer(stddev=0.1)
    elif (name == "uniform"):
        return tf.compat.v1.random_uniform_initializer(minval=-0.1, maxval=0.1)
    elif (name == "truncated"):
        return tf.compat.v1.truncated_normal_initializer(stddev=0.1)
    elif (name == "xavier"):
        return tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
    else:
        return tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")


# get optimizer op
# (Adamax support has been removed after optimizer testing since implementation was not compatible to newer Tensorflow version)
def get_optimizer(name, lr):
    if (name == "adam"):
        print("optimizer: ADAM")
        return tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    elif (name == "sgd"):
        print("optimizer: SGD")
        return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=(lr))
    else:
        return tf.compat.v1.train.AdamOptimizer(learning_rate=lr)


def get_cell(mode, hidden_size):
    if (mode == "lstm"):
        return tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_size)
    elif (mode == "gru"):
        return tf.compat.v1.nn.rnn_cell.GRUCell(hidden_size)
    else:
        return tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_size)
