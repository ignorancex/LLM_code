import tensorflow as tf
import pdb

class DCCM():
	def __init__(self,layer_name, filter_size,num_hidden,is_training,keep_prob):
		self.layer_name=layer_name
		self.filter_size=filter_size
		self.num_hidden=num_hidden
		self.is_training = is_training
		self.keep_prob = keep_prob
		
	def __call__(self, h, reuse=False):
		with tf.variable_scope(self.layer_name, reuse=False):
			num_hidden = self.num_hidden
			filter_size = self.filter_size
			is_training = self.is_training
			keep_prob = self.keep_prob
			h0_0 = tf.layers.conv2d(h,num_hidden,1, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c11')
			h0 = tf.layers.conv2d(h,num_hidden,filter_size, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c12')
			h0 = tf.nn.dropout(h0, keep_prob)

			h1 = tf.concat([h0_0,h0],axis=-1)
			h1 = tf.layers.conv2d(h1,num_hidden,1, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='f1')
			h1_0 = tf.layers.conv2d(h0,num_hidden,1, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c21')
			h1 = tf.layers.conv2d(h1,num_hidden,filter_size, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c22')
			h1 = tf.nn.dropout(h1, keep_prob)	

			h2 = tf.concat([h0_0,h1_0,h1],axis=-1)
			h2 = tf.layers.conv2d(h2,num_hidden,1, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='f2')
			h2_0 = tf.layers.conv2d(h1,num_hidden,1, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c31')
			h2 = tf.layers.conv2d(h2,num_hidden,filter_size, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c32')
			h2 = tf.nn.dropout(h2, keep_prob)

			h3 = tf.concat([h0_0,h1_0,h2_0,h2],axis=-1)
			h3 = tf.layers.conv2d(h3,num_hidden,1, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='f3')
			h3_0 = tf.layers.conv2d(h2,num_hidden,1, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c41')
			h3 = tf.layers.conv2d(h3,num_hidden,filter_size, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c42')
			h3 = tf.nn.dropout(h3, keep_prob)

			h4 = tf.concat([h0_0,h1_0,h2_0,h3_0,h3],axis=-1)
			h4 = tf.layers.conv2d(h4,num_hidden,1, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='f4')
			h4 = tf.layers.conv2d(h4,num_hidden,filter_size, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='c52')
			h4 = tf.nn.leaky_relu(h4)
			h4 = tf.nn.dropout(h4, keep_prob)
			out = h4
			return out

