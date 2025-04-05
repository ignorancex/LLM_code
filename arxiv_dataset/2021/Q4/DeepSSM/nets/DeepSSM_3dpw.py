import tensorflow as tf
from layers.DCCM import DCCM  as DB
import pdb
import numpy as np

def DeepSSM(images,keep_prob, seq_length, input_length, stacklength1, stacklength2, num_hidden,filter_size):
	with tf.variable_scope('DeepSSM', reuse=False):
		print 'DeepSSM'
		#print 'is_training', is_training
		tem=tf.concat([tf.expand_dims(images[:,0],1), images[:,0:seq_length-1,:,:]], 1)
		images_diff=tf.subtract(images,tem)
		
		h = images[:,0:input_length,:,:]
		gt_images=images[:,input_length:]
		input_diff=images_diff[:,0:input_length,:,:]
		gt_diff=images_diff[:,input_length:]

		last_diff = input_diff[:,-1]	
		last_frame = h[:,-1]
		dims=gt_images.shape[2]*gt_images.shape[3]

		inputsall = []
		inputsall.append(h)
		inputsall.append(input_diff)
		out=[]
		loss = 0
		pred_length=seq_length- input_length
		seqw=np.linspace(1,2*pred_length,pred_length,endpoint=True)
		seqw=seqw/np.sum(seqw)
		seqw=seqw[::-1]
		#seqw=np.ones((pred_length,1)) # train later prediction
		w =  tf.constant([3.0,1.0])
		if keep_prob is not 1:
			training=True
		else:
			training=False	
		general_diff1 = []
		for s in range(2):
			inputs = inputsall[s]
			inputs_db = []
			with tf.variable_scope('Encoder', reuse = bool(general_diff1)):
				for c in range(3):
					inputs_db1 = inputs[:,:,:,c]
					inputs_db1 = tf.expand_dims(inputs_db1,axis=3)
					if c is 0:
						reuse = False
					else:
						reuse = True
					with tf.variable_scope('Encoder_xyz', reuse = reuse):
						for i in range(stacklength1):
							inputs_db1 = DB('DCCM_xyz'+str(i),filter_size,num_hidden,training,keep_prob)(inputs_db1,reuse)
						#inputs_db1 = inputs_db1 + inputs_db2
					if c is 0:	
						inputs_db = inputs_db1
					else:
						inputs_db = tf.concat([inputs_db,inputs_db1],axis=-1)		
				# predict future speed
				for i in range(stacklength2):
					inputs_db=DB('Encoder_DCCM_joint'+str(i),filter_size,num_hidden,training,keep_prob)(inputs_db)
					#num_hidden = num_hidden*2	

				general_diff1.append(inputs_db)
		general_diff = tf.concat([general_diff1[0],general_diff1[1]],axis=-1)
		general_diff = tf.layers.conv2d(general_diff, num_hidden, filter_size, padding='same',activation=tf.nn.leaky_relu,
				kernel_initializer=tf.contrib.layers.xavier_initializer(),
				name='Fusion_layer')

		gen_diff=[]
		gen_images = []
		history_info=[]
		history_info.append(general_diff)
		for i in range(seq_length-input_length):
			if i%2 == 0:
				idx=np.linspace(0,i,i//2,endpoint=False)
				general_diff = general_diff + history_info[0]
				for k in range(len(idx)):
					general_diff = tf.concat([general_diff,history_info[int(idx[k]+1)]],axis=-1)
				#general_diff = general_diff + history_info[int(idx[k]+1)]
				# fusion feature
				#print('i,ii',i,i//2)
				with tf.variable_scope('hier_fusing_info'+str(i//2), reuse=False):
					general_diff = tf.layers.conv2d(general_diff, num_hidden, 1, padding='same',activation=tf.nn.leaky_relu,
							kernel_initializer=tf.contrib.layers.xavier_initializer(),
							name='hier_fusing_1')
					general_diff = tf.layers.conv2d(general_diff, num_hidden, filter_size, padding='same',activation=tf.nn.leaky_relu,
							kernel_initializer=tf.contrib.layers.xavier_initializer(),
							name='hier_fusing_2')

			with tf.variable_scope('decoder_diff', reuse=bool(gen_diff)):
				# last speed pose
				last_diff = tf.expand_dims(last_diff,axis=1)
				last_diff = tf.layers.conv2d(last_diff, num_hidden, filter_size, padding='same',activation=tf.nn.leaky_relu,
						kernel_initializer=tf.contrib.layers.xavier_initializer(),
						name='decoder1')
				# history speed

				general_diff = tf.layers.conv2d(general_diff, num_hidden, filter_size, padding='same',activation=tf.nn.leaky_relu,
						kernel_initializer=tf.contrib.layers.xavier_initializer(),
						name='History_de1')
				general_diff = tf.layers.conv2d(general_diff, num_hidden, filter_size, padding='same',activation=tf.nn.leaky_relu,
						kernel_initializer=tf.contrib.layers.xavier_initializer(),
						name='History_de2')

				general_diff = tf.add(general_diff,last_diff)
				history_info.append(general_diff)
				gen_diff1 = tf.layers.conv2d(general_diff, num_hidden//2, filter_size, padding='same',activation=tf.nn.leaky_relu,
						kernel_initializer=tf.contrib.layers.xavier_initializer(),
						name='decoder2')
				gen_diff1=tf.layers.flatten(gen_diff1)
				gen_diff1 = tf.layers.dense(inputs=gen_diff1, units=dims, activation=None)
				gen_diff1 = tf.reshape(gen_diff1,gt_images[:,0].shape)
				gen_diff.append(gen_diff1)
				last_diff = gen_diff1

				last_frame = last_frame + gen_diff1
				gen_images.append(last_frame)
				loss += w[0] * seqw[i] * tf.reduce_mean(tf.norm(gen_diff1-gt_diff[:,i], axis=2, keep_dims=True, name='normal'))
				loss += w[1] * seqw[i] * tf.reduce_mean(tf.norm(last_frame-gt_images[:,i], axis=2, keep_dims=True, name='normal'))

		# loss
		gen_images = tf.stack(gen_images,axis=1)
		loss = 1000*loss
		return [gen_images, loss]

