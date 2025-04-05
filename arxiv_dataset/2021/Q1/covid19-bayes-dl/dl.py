## Part of this code comes from:
## https://www.tensorflow.org/tutorials/structured_data/time_series


import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import datetime
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

MAX_EPOCHS = 50
INPUT_WIDTH = 7*4 # Days of input in each batch
OUT_STEPS = LABEL_WIDTH = 60 # Days of predictions

# Read data
df = pd.read_csv("dlio/civinput.csv")
dftest = pd.read_csv("dlio/civinput-test.csv")
column_indices = {name: i for i, name in enumerate(df.columns)}
train_df = df
val_df = df
test_df = dftest
num_features = df.shape[1]
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')



# Window Generator for everything
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}
    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift
    self.total_window_size = input_width + shift
    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)
  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])
  return inputs, labels

WindowGenerator.split_window = split_window


def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)
  ds = ds.map(self.split_window)
  return ds



WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)
  
@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.test))
  # And cache it for next time
  self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
  model.compile(loss=tf.losses.MeanAbsoluteError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])
  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping],verbose=0)
  return history
  



multi_window = WindowGenerator(input_width=INPUT_WIDTH,
                               label_width=LABEL_WIDTH,
                               shift=OUT_STEPS)

multi_window

mymodel = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
#    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False,recurrent_dropout = 0.2,dropout = 0.2)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
     # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)
    
feedback_model = FeedBack(units=64, out_steps=OUT_STEPS)

def warmup(self, inputs):
  # inputs.shape => (batch, time, features)
  # x.shape => (batch, lstm_units)
  x, *state = self.lstm_rnn(inputs)
  # predictions.shape => (batch, features)
  prediction = self.dense(x)
  return prediction, state

FeedBack.warmup = warmup

def call(self, inputs, training=None):
  # Use a TensorArray to capture dynamically unrolled outputs.
  predictions = []
  # Initialize the lstm state
  prediction, state = self.warmup(inputs)
  # Insert the first prediction
  predictions.append(prediction)
  # Run the rest of the prediction steps
  for n in range(1, self.out_steps):
    # Use the last prediction as input.
    x = prediction
    # Execute one lstm step.
    x, state = self.lstm_cell(x, states=state,
                              training=training)
    # Convert the lstm output to a prediction.
    prediction = self.dense(x)
    # Add the prediction to the output
    predictions.append(prediction)
  # predictions.shape => (time, batch, features)
  predictions = tf.stack(predictions)
  # predictions.shape => (batch, time, features)
  predictions = tf.transpose(predictions, [1, 0, 2])
  return predictions

FeedBack.call = call

pastdaystrain = tf.keras.preprocessing.timeseries_dataset_from_array(multi_window.train_df,targets=None,sequence_length=multi_window.total_window_size)
pastdaystest = tf.keras.preprocessing.timeseries_dataset_from_array(multi_window.test_df,targets=None,sequence_length=multi_window.total_window_size)

## AUTOREGRESSIVE PREDICTIONS THIS IS USED IN THE PAPER
history = compile_and_fit(feedback_model, multi_window)
predaystrain = feedback_model.predict(pastdaystrain)
predaystest = feedback_model.predict(pastdaystest)

## ONE STEP PREDICTIONS THIS IS NOT USED IN THE PAPER BUT IT MAY BE USEFUL
#history = compile_and_fit(mymodel, multi_window)
#predays = mymodel.predict(pastdays)

pd.DataFrame(history.history).to_csv('dlio/history.csv')


for region in range(predaystest.shape[2]):
  predaystrain[:, :, region] = train_mean[region] + predaystrain[:, :, region]*train_std[region]
  predaystest[:, :, region] = train_mean[region] + predaystest[:, :, region]*train_std[region]


for i in range(predaystrain.shape[0]):
    for j in range(predaystrain.shape[1]):
        for k in range(predaystrain.shape[2]):
            predaystrain[i, j, k] = round(predaystrain[i, j, k])

for i in range(predaystest.shape[0]):
    for j in range(predaystest.shape[1]):
        for k in range(predaystest.shape[2]):
            predaystest[i, j, k] = round(predaystest[i, j, k])

# pred=predays[:,6,13]
# obs=df["MD"][df.shape[0]-pred.shape[0]:df.shape[0]]
# mdf={"pred":pred,"obs":obs}
# mdf = pd.DataFrame(data=mdf)
# mdf.corr()
# 
# plt.close()
# plt.show()
# fig = plt.figure(figsize=(14,8))
# plt.plot(obs, pred, 'o', color='black');
# plt.show()

for i in range(predaystrain.shape[1]):
  pd.DataFrame(predaystrain[:,i,:],columns=df.columns).to_csv('dlio/civout'+str(i+1)+'.csv')

for i in range(predaystest.shape[1]):
  pd.DataFrame(predaystest[:,i,:],columns=df.columns).to_csv('dlio/civout-test'+str(i+1)+'.csv')
