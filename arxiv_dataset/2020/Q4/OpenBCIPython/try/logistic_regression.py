from sklearn import datasets
from sklearn import metrics
import tensorflow as tf


iris = datasets.load_iris()

def my_model(features, labels):
  """DNN with three hidden layers."""
  # Convert the labels to a one-hot tensor of shape (length of features, 3) and
  # with a on-value of 1 for each one-hot vector of length 3.
  labels = tf.one_hot(labels, 3, 1, 0)

  # Create three fully connected layers respectively of size 10, 20, and 10.
  features = tf.contrib.layers.stack(features, tf.contrib.layers.fully_connected, [10, 20, 10])

  # Create two tensors respectively for prediction and loss.
  prediction, loss = (
      tf.contrib.learn.models.logistic_regression(features, labels)
  )

  # Create a tensor for training op.
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)

  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

classifier =  tf.contrib.learn.Estimator(model_fn=my_model)
classifier.fit(iris.data, iris.target, steps=1000)

y_predicted = [
  p['class'] for p in classifier.predict(iris.data, as_iterable=True)]
score = metrics.accuracy_score(iris.target, y_predicted)
print('Accuracy: {0:f}'.format(score))