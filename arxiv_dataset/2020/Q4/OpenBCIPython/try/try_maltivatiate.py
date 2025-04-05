import tensorflow as tf
import numpy as np


# generate some dataset
DIMENSIONS = 5
DS_SIZE = 5000
TRAIN_RATIO = 0.5 # 60% of the dataset isused for training
_train_size = int(DS_SIZE*TRAIN_RATIO)
_test_size = DS_SIZE - _train_size
f = lambda(x): sum(x) # the "true" function: f = 0 + 1*x1 + 1*x2 + 1*x3 ...
noise = lambda: np.random.normal(0,10) # some noise
# training globals
LAMBDA = 1e6 # L2 regularization factor
# generate the dataset, the labels and split into train/test
ds = [[np.random.rand()*1000 for d in range(DIMENSIONS)] for _ in range(DS_SIZE)]
ds = [([1]+x, [f(x)+noise()]) for x in ds] # add x[0]=1 dimension and labels
np.random.shuffle(ds)
train_data, train_labels = zip(*ds[0:_train_size])
test_data, test_labels = zip(*ds[_train_size:])



def normalize_data(matrix):
    averages = np.average(matrix,0)
    mins = np.min(matrix,0)
    maxes = np.max(matrix,0)
    ranges = maxes - mins
    return ((matrix - averages)/ranges)

def run_regression(X, Y, X_test, Y_test, lambda_value = 0.1, normalize=False, batch_size=10, alpha=1e-8):
    x_train = normalize_data(X) if normalize else X
    y_train = Y
    x_test = X_test
    y_test = Y_test
    session = tf.Session()
    # Calculate number of features for X and Y
    x_features_length = len(X[0])
    y_features_length = len(Y[0])
    # Build Tensorflow graph parts
    x = tf.placeholder('float', [None, x_features_length], name="X")
    y = tf.placeholder('float', [None, y_features_length], name="Y")
    theta = tf.Variable(tf.random_normal([x_features_length, y_features_length], stddev=0.01), name="Theta")
    lambda_val = tf.constant(lambda_value)
    # Trying to implement this way http://openclassroom.stanford.edu/MainFolder/DocumentPage.php?course=MachineLearning&doc=exercises/ex5/ex5.html
    y_predicted = tf.matmul(x, theta, name="y_predicted")
    #regularization_cost_part = tf.cast(tf.multiply(lambda_val,tf.reduce_sum(tf.pow(theta,2)), name="regularization_param"), 'float')
    #polynomial_cost_part = tf.reduce_sum(tf.pow(tf.subtract(y_predicted, y), 2), name="polynomial_sum")
    # Set up some summary info to debug
    with tf.name_scope('cost') as scope:
        #cost_func = tf.multiply(tf.cast(1/(2*batch_size), 'float'), tf.cast(tf.add(polynomial_cost_part, regularization_cost_part), 'float'))
        cost_func = (tf.nn.l2_loss(y_predicted - y)+lambda_val*tf.nn.l2_loss(theta))/float(batch_size)
        #DEPRECATED*** cost_summary = tf.scalar_summary("cost", cost_func)
        cost_summary = tf.summary.scalar('cost', cost_func)# Add a scalar summary for the snapshot loss.
    training_func = tf.train.GradientDescentOptimizer(alpha).minimize(cost_func)
    with tf.name_scope("test") as scope:
        correct_prediction = tf.subtract(tf.cast(1, 'float'), tf.reduce_mean(tf.subtract(y_predicted, y)))
        accuracy = tf.cast(correct_prediction, "float")
        #DEPRECATED*** accuracy_summary = tf.scalar_summary("accuracy", accuracy)
        #accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    saver = tf.train.Saver()
    #DEPRECATED*** merged = tf.merge_all_summaries()
    merged = tf.summary.merge_all()
    #DEPRECATED*** writer = tf.train.SummaryWriter("/tmp/football_logs", session.graph_def)
    writer = tf.summary.FileWriter("/tmp/football_logs", session.graph)
    #DEPRECATED*** init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    session.run(init)
    for i in range(1, (len(x_train)/batch_size)):
        session.run(training_func, feed_dict={x: x_train[i*batch_size:i*batch_size+batch_size], y: y_train[i*batch_size:i*batch_size+batch_size]})
        if i % batch_size == 0:
            print "test accuracy %g"%session.run(accuracy, feed_dict={x: x_test, y: y_test})
            #result = session.run([merged, accuracy], feed_dict={x: x_test, y: y_test})
            # writer.add_summary(result[0], i)
            # print "step %d, training accuracy %g"%(i, result[1])
            #writer.flush()
    print "final test accuracy %g"%session.run(accuracy, feed_dict={x: x_test, y: y_test})
    # save_path = saver.save(session, "/tmp/football.ckpt")
    # print "Model saved in file: ", save_path
    session.close()

run_regression(train_data, train_labels, test_data, test_labels, normalize=False, alpha=1e-8)