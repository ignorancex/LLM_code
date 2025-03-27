import tensorflow as tf
import matplotlib.pyplot as plt
# import skimage.transform
# import numpy as np
import time
import os
# import cPickle as pickle
# from scipy import ndimage
from utils import *
from evaluate import *
from visual import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
GPU_MEMORY = 1


class AttCaptionsSolver(object):
    def __init__(self, model, data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
                - word_to_idx: Mapping dictionary from word to index
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path
            - model_path: String; model path for saving
            - test_model: String; model path for test
        """

        self.model = model
        self.data = data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_every = kwargs.pop('print_every', 100)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/kenn/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.bool_save_model = kwargs.pop('bool_save_model', False)
        self.generated_caption_len = kwargs.pop('generated_caption_len', 4)
        self.bool_val = kwargs.pop('bool_val', True)
        self.bool_selector = kwargs.pop('bool_selector', True)

        self.bool_save_this = False
        self.ini_acc = 0.50

    def train(self):
        n_examples = self.data['data'].shape[0]
        data = self.data['data']
        captions = self.data['captions']
        data_idxs = self.data['data_idxs']  # 多条caps对应单个数据时用来对应的。
        # val_data = self.val_data['data']
        # n_iters_val = int(np.ceil(float(val_data.shape[0]) / self.batch_size))
        captions_s, idxs_s = shuffle(captions, data_idxs)
        train_caps, train_idxs, val_caps, val_idxs = cross_val(captions_s, idxs_s, 1, cross=5)

        # build graphs for training model and sampling captions
        # This scope fixed things!!
        with tf.variable_scope(tf.get_variable_scope()):
            loss = self.model.build_model()
            tf.get_variable_scope().reuse_variables()
            _, _, generated_captions = self.model.build_sampler(max_len=self.generated_caption_len)

        # train op
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = tf.gradients(loss, tf.trainable_variables())
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        print("The number of epoch: %d" % self.n_epochs)
        print("Data size: %d" % n_examples)
        print("Batch size: %d" % self.batch_size)
        # print("Iterations per epoch: %d" % n_iters_per_epoch)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY
        with tf.Session(config=sess_config) as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print("Start training with pretrained Model..")
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            start_t = time.time()

            for e in range(self.n_epochs):
                train_caps, train_idxs = shuffle(train_caps, train_idxs)
                n_iters_per_epoch = train_caps.shape[0] // self.batch_size
                for i in range(n_iters_per_epoch):
                    captions_batch = train_caps[i * self.batch_size:(i + 1) * self.batch_size]
                    idxs_batch = train_idxs[i * self.batch_size:(i + 1) * self.batch_size]
                    data_batch = data[idxs_batch]
                    feed_dict = {self.model.data: data_batch, self.model.captions: captions_batch}
                    _, loss_ = sess.run([train_op, loss], feed_dict)
                    curr_loss += loss_

                if (e + 1) % self.print_every == 0:
                    print("################ Epoch %d ################" % (e + 1))
                    print("Previous epoch loss: ", prev_loss)
                    print("Current epoch loss: ", curr_loss)
                    print("Elapsed time: ", time.time() - start_t)
                    if self.bool_val:
                        val_data = data[val_idxs]
                        val_pred = sess.run(generated_captions,
                                            feed_dict={self.model.data: val_data, self.model.captions: val_caps})
                        val_acc = abs_evaluate(val_caps, val_pred)
                        val_acc_ = acc_evaluate(val_caps, val_pred)
                        print("Epoch {} validated Acc:{} \n ".format(e + 1, val_acc_))

                        for _ in range(10):
                            n = np.random.randint(val_data.shape[0])
                            val_true_decode = decode_captions(val_caps[n], self.model.idx_to_word)
                            val_pred_decode = decode_captions(val_pred[n], self.model.idx_to_word)
                            print("Val data no.{} Ground-True:{} Pred:{}".format(n, val_true_decode, val_pred_decode))

                    prev_loss = curr_loss
                    curr_loss = 0

                    # if val_acc > self.ini_acc:
                    #     self.ini_acc = val_acc
                    #     self.bool_save_this = True
                    # save model's parameters
                    self.bool_save_this = True
                    if self.bool_save_this and self.bool_save_model:
                        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e + 1)
                        print("model-%s saved." % (e + 1))
                        self.bool_save_this = False

    def test(self, save_visual_para=True):
        """
        Args: - data: dictionary with the following keys: - features: Feature vectors of shape (5000, 196,
        512) - file_names: Image file names of shape (5000, ) - captions: Captions of shape (24210, 17) - image_idxs:
        Indices for mapping caption to image of shape (24210, ) - features_to_captions: Mapping feature to captions (
        5000, 4~5) - split: 'train', 'val' or 'test' - attention_visualization: If True, visualize attention weights
        with images for each sampled word. (ipthon notebook) - save_sampled_captions: If True, save sampled captions
        to pkl file for computing BLEU scores.
        """

        # features = data['features']
        data = self.data['data']
        print("data", data.shape)
        captions = self.data['captions']
        print(captions.shape)
        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=self.generated_caption_len)
        # (N, max_len, L), (N, max_len)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY
        with tf.Session(config=sess_config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            # features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = {self.model.data: data}
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_captions(sam_cap, self.model.idx_to_word)
            de_true = decode_captions(captions, self.model.idx_to_word)
            print("Acc:{}".
                  format(acc_evaluate(captions, sam_cap)))
            if save_visual_para:
                np.save(os.path.join(self.log_path, 'pred_cap_lb.npy'), sam_cap)
                np.save(os.path.join(self.log_path, 'pred_cap.npy'), decoded)
                np.save(os.path.join(self.log_path, 'true_cap.npy'), de_true)
                np.save(os.path.join(self.log_path, 'att_map.npy'), alps)
