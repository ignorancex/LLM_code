import tensorflow as tf
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


tf.app.flags.DEFINE_string('f', '', 'kernel')   # 添加的，不报错

flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 50, 'number of iterations.')

'''
infor: number of clusters 
'''
infor = {'wiki' : 17, 'cora' : 7, 'citeseer' : 6, 'celegans' : 12, 'email' : 12, 'texas' : 12, 'wisconsin' : 12, 'polbooks' : 12, 'karate' : 12}


'''
We did not set any seed when we conducted the experiments described in the paper;
We set a seed here to steadily reveal better performance of ARGA
'''
seed = 7
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

def get_settings(dataname, model, task):
    if model != 'arga_ae' and model != 'arga_vae':
        print('error: wrong model name')

    if task == 'link_prediction':
        #iterations = 4 * FLAGS.iterations
        iterations = 400
        re = {'data_name': dataname, 'iterations' : iterations,'model' : model}

    return re

