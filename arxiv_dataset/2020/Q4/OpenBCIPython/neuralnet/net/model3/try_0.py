from __future__ import division
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib import layers as tflayers
tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op

class Deviatev1:
    def __init__(self, project_location):
        self.project_location = project_location
        self.project_config = self.project_location + "/config/config.json"
        self.column_names = ['ch1', 'ch2', 'ch3']
        self.number_of_eatures = 5
        self.number_of_labels = 180
        self.num_epochs = 5000
        self.learning_rate =1e-8
        self.batch_size = 5
        self.keep_prob = 0.9
        self.hidden_size = 10
        self.num_layers_rnn = 5
        self.num_steps = 5
        self.time_steps = 1
        self.dnn_layer_size = 5
        self.model_params = {"learning_rate": self.learning_rate, "keep_prob": self.keep_prob
            , 'num_steps': self.num_steps, 'num_layers_rnn':self.num_layers_rnn, 'dnn_layer_size': self.dnn_layer_size
            , 'number_of_labels': self.number_of_labels }
        self.validation_metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_accuracy,
                    prediction_key="classes"),
            "precision":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_precision,
                    prediction_key="classes"),
            "recall":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.contrib.metrics.streaming_recall,
                    prediction_key="classes")
        }

        self.test_metrics = {
            "accuracy":
                tf.contrib.learn.MetricSpec(
                    metric_fn=tf.metrics.accuracy, prediction_key="classes"),
        }

    def import_data(self, angle_type):
        kinect__angles = pd.read_csv(
            self.project_location + "/build/dataset/train/result/reconstructed_bycept_kinect__angles_.csv",
            header=None, names=self.column_names).dropna()
        channel_signals = pd.read_csv(self.project_location
                                      + "/build/dataset/train/result/bycept_feature_vectors.csv").dropna()
        # kinect__angles = kinect__angles.applymap(lambda x: '%.2f' % x)
        y_vals = np.array(kinect__angles.ix[:, angle_type], dtype=np.int32)

        x_vals = np.array(channel_signals)
        train_presentation = 0.8
        test_presentation = 0.8

        train_indices = np.random.choice(len(x_vals), int(round(len(x_vals) * train_presentation)), replace=False)
        rest_of_data_set_x_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
        rest_of_data_set_x = x_vals[rest_of_data_set_x_indices]
        rest_of_data_set_y_indices = np.array(list(set(range(len(y_vals))) - set(train_indices)))
        rest_of_data_set_y = y_vals[rest_of_data_set_y_indices]
        test_indices = np.random.choice(len(rest_of_data_set_x),
                                        int(round(len(rest_of_data_set_x_indices) * test_presentation)),
                                        replace=False)
        validate_indices = np.array(list(set(range(len(rest_of_data_set_x_indices))) - set(test_indices)))
        self.train_x = self.rnn_data(x_vals[train_indices])
        self.train_y = self.rnn_data(y_vals[train_indices], labels=True)
        self.test_x = self.rnn_data(rest_of_data_set_x[test_indices])
        self.test_y = self.rnn_data(rest_of_data_set_y[test_indices], labels=True)
        self.validate_x = self.rnn_data(rest_of_data_set_x[validate_indices])
        self.validate_y = self.rnn_data(rest_of_data_set_y[validate_indices], labels=True)
        self.kinect_angles = np.array(kinect__angles.ix[:, 0])

    def rnn_data(self, data, labels=False):
        data = pd.DataFrame(data)
        rnn_df = []
        for i in range(data.shape[0] - self.num_steps):
            if labels:
                rnn_df.append(data.iloc[i + self.num_steps].as_matrix())
            else:
                data_ = data.iloc[i: i + self.num_steps].as_matrix()
                rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

            return np.array(rnn_df, dtype=np.float32)

    def model_fn(self, features, targets, mode, params):

        def dnn_layers(input_layers, layers):
            if layers and isinstance(layers, dict):
                return tflayers.stack(input_layers, tflayers.fully_connected,
                                      layers['layers'],
                                      activation=layers.get('activation'),
                                      dropout=layers.get('dropout'))
            elif layers:
                return tflayers.stack(input_layers, tflayers.fully_connected, layers)
            else:
                return input_layers

        def lstm_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(params["num_steps"], forget_bias=0.0,
                                                                state_is_tuple=True),output_keep_prob=params['keep_prob'])
        def lstm_forward_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(params["num_steps"], forget_bias=0.2,
                                                                state_is_tuple=True), output_keep_prob=params['keep_prob'])
        def lstm_backword_cell():
            return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(params["num_steps"], forget_bias=0.2,
                                                                state_is_tuple=True), output_keep_prob=params['keep_prob'])


        def _lstm_model(features, targets):

            lstm_fw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_forward_cell() for _ in range(params['num_layers_rnn'])],
                                                             state_is_tuple=True)
            lstm_bw_multicell = tf.contrib.rnn.MultiRNNCell([lstm_backword_cell() for _ in range(params['num_layers_rnn'])],
                                                             state_is_tuple=True)

            # initial_state_fw = lstm_fw_multicell.zero_state(batch_size=self.batch_size, dtype='float32')
            # initial_state_bw = lstm_bw_multicell.zero_state(batch_size=self.batch_size, dtype='float32')
            # init_state = tf.zeros([params["num_steps"]*2])
            # initial_state = cell.zero_state(batch_size, tf.float32)

            features = tf.unstack(features, num=params["num_steps"], axis=1)

            with tf.variable_scope("RNN"):
                # output, state = tf.contrib.rnn.static_rnn(lstm_fw_multicell, features, dtype=tf.float32)
                output, state = tf.contrib.learn.models.bidirectional_rnn(lstm_fw_multicell, lstm_bw_multicell, features,
                                                                          # initial_state_fw=initial_state_fw,
                                                                          # initial_state_bw=initial_state_bw,
                                                                          # sequence_length=[5, 5],
                                                                          dtype='float32')
            # targets = tf.one_hot(targets, params['number_of_labels'], 1, 0)
            # targets = tf.Variable(tf.random_normal([1]))
            # logits = tf.contrib.layers.fully_connected(output[-1], 180)
            # loss = tf.contrib.losses.softmax_cross_entropy(logits, targets)

            output = dnn_layers(output[-1], [params['dnn_layer_size'], params['dnn_layer_size']])
            # # first = tf.contrib.layers.fully_connected(output[-1], num_outputs=300)
            # # first = tf.contrib.layers.fully_connected(first, num_outputs=200)
            # # first = tf.contrib.layers.fully_connected(first, num_outputs=100)
            # # first = tf.contrib.layers.fully_connected(first, num_outputs=50)
            # # first = tf.contrib.layers.fully_connected(first, num_outputs=25)
            # output = tf.contrib.layers.fully_connected(output, num_outputs=5)
            # output = tf.contrib.layers.linear(output, 1)
            # output = tf.reshape(output, [-1])
            output = self.extract(output, 'input')
            labels = self.extract(targets, 'labels')
            labels = tf.cast(labels, tf.float32)
            # prediction, loss = tf.contrib.learn.models.linear_regression(outputs, labels)

            # with tf.variable_scope('softmax'):
            #     W = tf.get_variable('W', [5, 180])
            #     b = tf.get_variable('b', [180], initializer=tf.constant_initializer(0.0))
            # labels = tf.reshape(labels, [-1])
            # logits = tf.matmul(outputs, W) + b
            # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            # prediction = tf.argmax(logits, 1)

            # prediction = tf.get_variable('b', [180], initializer=tf.constant_initializer(0.0))
            # labels= tf.cast(labels, tf.float32)
            # n_observations = 180
            # for pow_i in range(1, 5):
            #     # W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
            #     W = tf.get_variable('weight_%d' % pow_i, [5, 180])
            #     logits = tf.matmul(tf.pow(outputs, pow_i), W)+ prediction
            #
            # loss = tf.reduce_sum(tf.pow(prediction - labels, 2)) / (n_observations - 1)

            theta = tf.Variable(tf.random_normal([5, 1], stddev=0.01), name="Theta")
            lambda_val = tf.constant(0.1)
            y_predicted = tf.matmul(output, theta, name="y_predicted")

            for pow_i in range(1, 2):
                W = tf.Variable(tf.random_normal([5, 1]), name='weight_%d' % pow_i)
                y_predicted = tf.matmul(tf.pow(output, pow_i), W)+ y_predicted

            with tf.name_scope('cost') as scope:
                loss = (tf.nn.l2_loss(y_predicted - labels) + lambda_val * tf.nn.l2_loss(theta)) / float(self.batch_size)
                loss_summary = tf.summary.scalar('cost', loss)

            # with tf.name_scope("prediction") as scope:
                # labels = tf.cast(labels, tf.int32)
                # correct_prediction = tf.subtract(tf.cast(1, 'float'), tf.reduce_mean(tf.subtract(y_predicted, labels)), name="angles")
            #     correct_prediction = tf.reduce_mean(tf.subtract(y_predicted, labels), name="angles")

            # with tf.name_scope("accuracy") as scope:
            #     accuracy = tf.cast(correct_prediction, "float")

            # theta = tf.Variable(tf.random_normal([5, 1], stddev=0.01), name="Theta")
            # lambda_val = tf.constant(0.1)
            # prediction = tf.matmul(output, theta, name="y_predicted")
            # with tf.name_scope('cost') as scope:
            #     # cost_func = tf.multiply(tf.cast(1/(2*batch_size), 'float'), tf.cast(tf.add(polynomial_cost_part, regularization_cost_part), 'float'))
            #     # cost_func = (tf.nn.l2_loss(prediction - labels) + lambda_val * tf.nn.l2_loss(theta)) / float(self.batch_size)
            #     # DEPRECATED*** cost_summary = tf.scalar_summary("cost", cost_func)
            #     loss = (tf.nn.l2_loss(prediction - labels) + lambda_val * tf.nn.l2_loss(theta)) / float(self.batch_size)
            #     # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=labels))


            # # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            # %% And now we can look at the mean of our network's correct guesses

            # train_op = tf.train.AdamOptimizer(params["learning_rate"]).minimize(total_loss)
            # logits = tf.contrib.layers.fully_connected(output, 15, activation_fn=None)
            # labels = array_ops.reshape(
            #     constant_op.constant(
            #         targets, dtype=tf.int32), (-1, 1))
            # prediction, loss = tf.contrib.learn.models.logistic_regression_zero_init(output, labels)

            train_op = tf.contrib.layers.optimize_loss(
                loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
                learning_rate=params["learning_rate"])

            # correct_prediction = tf.equal(tf.argmax(train_prediction, 1), train_labels)

            predictions_dict = {"classes": tf.argmax(input=y_predicted, axis=1, name="angles")}

            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(tf.cast(y_predicted, tf.float32), tf.cast(labels, tf.float32))
            }

            return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions_dict, loss=loss, train_op=train_op
                                           , eval_metric_ops=eval_metric_ops)

        return _lstm_model(features, targets)

    def extract(self, data, key):
        if isinstance(data, dict):
            assert key in data
            return data[key]
        else:
            return data

    def execute(self):
        self.import_data(1)
        estimator = tf.contrib.learn.Estimator(model_fn=self.model_fn, params=self.model_params,
                                        model_dir="/home/runge/openbci/git/OpenBCI_Python/neuralnet/net/model3/model",
                                        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=20))
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            self.test_x,
            self.test_y,
            every_n_steps=50,
            metrics=self.validation_metrics,
            early_stopping_metric="loss",
            early_stopping_metric_minimize=True,
            early_stopping_rounds=200)
        tensors_to_log = {"classes": "angles"}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
        estimator.fit(x=self.train_x, y=self.train_y, steps=50, monitors=[validation_monitor, logging_hook], batch_size=self.batch_size)

        test_results = estimator.evaluate(x=self.test_x, y=self.test_y, steps=1)
        print("Loss: %s" % test_results["loss"])
        print("Root Mean Squared Error: %s" % test_results["rmse"])
        # self.validate_x = self.validate_x
        predictions = estimator.predict(x=self.validate_x, as_iterable=True)
        for i, p in enumerate(predictions):
            print("Prediction %s: %s" % (self.validate_y[i], p["classes"]))



project_loction = '/home/runge/openbci/git/OpenBCI_Python'
model = Deviatev1(project_loction)
model.execute()






































# a smarter learning rate for gradientOptimizer
# learningRate = tf.train.exponential_decay(learning_rate=0.0008,
#                                           global_step= 1,
#                                           decay_steps=train_x.shape[0],
#                                           decay_rate= 0.95,
#                                           staircase=True)
#
# X = tf.placeholder(tf.float32, [None, numFeatures])
# yGold = tf.placeholder(tf.float32, [None, numLabels])
# weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
#                                        mean=0,
#                                        stddev=(np.sqrt(6/numFeatures+
#                                                          numLabels+1)),
#                                        name="weights"))
# bias = tf.Variable(tf.random_normal([1,numLabels],
#                                     mean=0,
#                                     stddev=(np.sqrt(6/numFeatures+numLabels+1)),
#                                     name="bias"))
#
#
# init_OP = tf.global_variables_initializer()
# apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
# add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
# activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
# cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")
# training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)
#
# epoch_values=[]
# accuracy_values=[]
# cost_values=[]
#
# plt.ion()
# fig = plt.figure()
# ax1 = plt.subplot("211")
# ax1.set_title("TRAINING ACCURACY", fontsize=18)
# ax2 = plt.subplot("212")
# ax2.set_title("TRAINING COST", fontsize=18)
# plt.tight_layout()
#
#
# sess = tf.Session()
# sess.run(init_OP)
#
# correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
# accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
# activation_summary_OP = tf.summary.histogram("output", activation_OP)
# accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
# cost_summary_OP = tf.summary.scalar("cost", cost_OP)
# weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
# biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))
#
# all_summary_OPS = tf.summary.merge_all()
# writer = tf.summary.FileWriter("summary_logs", sess.graph)
#
# # Initialize reporting variables
# cost = 0
# diff = 1
# # Declare batch size
# batch_size = 25
# # Training epochs
# for i in range(numEpochs):
#     if i > 1 and diff < .0001:
#         print("change in cost %g; convergence."%diff)
#         break
#     else:
#         rand_index = np.random.choice(len(train_x), size=batch_size)
#         rand_x = train_x[rand_index]
#         rand_y = np.transpose([train_y[rand_index]])
#         # Run training step
#         step = sess.run(training_OP, feed_dict={X: rand_x, yGold: rand_y})
#         # Report occasional stats
#         if i % 10 == 0:
#             # Add epoch to epoch_values
#             epoch_values.append(i)
#             # Generate accuracy stats on test data
#             summary_results, train_accuracy, newCost = sess.run(
#                 [all_summary_OPS, accuracy_OP, cost_OP],
#                 feed_dict={X: rand_x, yGold: rand_y}
#             )
#             # Add accuracy to live graphing variable
#             accuracy_values.append(train_accuracy)
#             # Add cost to live graphing variable
#             cost_values.append(newCost)
#             # Write summary stats to writer
#             writer.add_summary(summary_results, i)
#             # Re-assign values for variables
#             diff = abs(newCost - cost)
#             cost = newCost
#
#             #generate print statements
#             print("step %d, training accuracy %g"%(i, train_accuracy))
#             print("step %d, cost %g"%(i, newCost))
#             print("step %d, change in cost %g"%(i, diff))
#
#             # Plot progress to our two subplots
#             accuracyLine, = ax1.plot(epoch_values, accuracy_values)
#             costLine, = ax2.plot(epoch_values, cost_values)
#             fig.canvas.draw()
#             time.sleep(1)
#
# rand_index = np.random.choice(len(test_x), size=len(test_x))
# rand_x = test_x[rand_index]
# rand_y = np.transpose([test_y[rand_index]])
# # How well do we perform on held-out test data?
# print("final accuracy on test set: %s" %str(sess.run(accuracy_OP,
#                                                      feed_dict={X: rand_x,
#                                                                 yGold: rand_y})))
# # Create Saver
# saver = tf.train.Saver()
# # Save variables to .ckpt file
# saver.save(sess, "trained_variables.ckpt")
# sess.close()
#
# # To view tensorboard:
#     #1. run: tensorboard --logdir=/path/to/log-directory
#     #2. open your browser to http://localhost:6006/
#
#
#
