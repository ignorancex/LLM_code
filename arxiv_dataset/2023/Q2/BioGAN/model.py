from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import vgg, pdb, time
import functools

from module import *
from utils import *

class cyclegan(object):
    def __init__(self, sess, args , features , CONTENT_WEIGHT , STYLE_WEIGHT , TV_WEIGHT):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.Blender_discriminator = Blender_discriminator
        self.discriminator = discriminator
        self.vgg_path = args.VGG_dir
        self.STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

        self.CONTENT_LAYER = 'relu4_2'
        self.Style_features = features
        self.CONTENT_WEIGHT = CONTENT_WEIGHT
        self.STYLE_WEIGHT = STYLE_WEIGHT
        self.TV_WEIGHT= TV_WEIGHT
        self.batch_shape = (args.batch_size,1024,768,3)

        if args.use_resnet:
            self.generator = generator_unet
            # generator_resnet
            self.Blender_generator = Blender_generator_resnet
            self.blender = blender
            self.Object_blender = Object_blender
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterions


        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        print("Building the mdoe")
        self._build_model()

        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)

    def _tensor_size(self , tensor):
        from operator import mul
        return functools.reduce(mul, (d for d in tensor.get_shape()[1:]), 1)


    def _build_model(self):
        
        self.real_data = tf.placeholder(tf.float32,
                                        [None, 1024, 768,
                                         self.input_c_dim + self.output_c_dim ],
                                        name='real_A_and_B_images')

        self.image = self.real_data[:, :, :, :self.input_c_dim]
        self.style = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

# //////////////////////  Style features ////////////////////////////////////
        # print("222222222222222222222222")

        # self.style_image_pre = vgg.preprocess(self.style)
        # print("11111111111111111111111111111" , self.style_image_pre.shape)
        # self.net = vgg.net(self.vgg_path, self.style_image_pre)
        # print("333333333333333333333333" )

        # self.Style_features[self.CONTENT_LAYER] = self.net[self.CONTENT_LAYER]

        # self.style_pre = np.array([self.style])
        # for layer in self.STYLE_LAYERS:
        #     self.features = self.net[layer]
        #     print("FEATURE SHAOE ISS S S S S" , self.features)
        #     self.features = np.reshape(self.features, (-1, self.features.shape[3]))
            

        #     self.gram = np.matmul(self.features.T, self.features) / self.features.size
        #     self.style_features[layer] = self.gram

# //////////////////////  Contetnt features ////////////////////////////////////   

        temp_image = reverse_image(self.image)
        self.X_pre = vgg.preprocess(temp_image)

        # precompute content features
        self.content_features = {}
        self.content_net = vgg.net(self.vgg_path, self.X_pre)

        self.content_features[self.CONTENT_LAYER] = self.content_net[self.CONTENT_LAYER]
# /////////////////////////////////////////////////////////////////////////////////////////////////
        temp_image = reverse_image(self.style)
        self.X_pre = vgg.preprocess(temp_image)

        # precompute content features
        self.content_features_ = {}
        self.content_net_ = vgg.net(self.vgg_path, self.X_pre)

        self.content_features_[self.CONTENT_LAYER] = self.content_net_[self.CONTENT_LAYER]
# /////////////////////////////////////////////////////////////////////////////////////////////////


        self.fake = self.generator(self.image ,  self.options, False, name="generator")
        self.D_fake = self.discriminator(self.fake, self.options, reuse=False, name="discriminator")


        self.preds_pre = vgg.preprocess(reverse_image(self.fake))

        self.net = vgg.net(self.vgg_path, self.preds_pre)

        self.content_size = self._tensor_size(self.content_features[self.CONTENT_LAYER])*self.batch_size
        assert self._tensor_size(self.content_features[self.CONTENT_LAYER]) == self._tensor_size(self.net[self.CONTENT_LAYER])

        self.content_loss = self.CONTENT_WEIGHT * (2 * tf.nn.l2_loss(
            self.net[self.CONTENT_LAYER] - self.content_features[self.CONTENT_LAYER]) / tf.cast(self.content_size , dtype=float)
        )
# ///////////////////////////////////////////////////////////////////
        self.content_size_ = self._tensor_size(self.content_features_[self.CONTENT_LAYER])*self.batch_size
        assert self._tensor_size(self.content_features_[self.CONTENT_LAYER]) == self._tensor_size(self.net[self.CONTENT_LAYER])

        self.content_loss_ = self.CONTENT_WEIGHT * (2 * tf.nn.l2_loss(
            self.net[self.CONTENT_LAYER] - self.content_features_[self.CONTENT_LAYER]) / tf.cast(self.content_size_ , dtype=float)
        )


        self.style_losses = []
        for style_layer in self.STYLE_LAYERS:
            layer = self.net[style_layer]
            # bs, height, width, filters = map(lambda i:i,layer.get_shape())
            bs, height, width, filters = tf.shape(layer)[0] , tf.shape(layer)[1], tf.shape(layer)[2] , tf.shape(layer)[3]
            size = height * width * filters
            feats = tf.reshape(layer, (bs, height * width, filters))
            feats_T = tf.transpose(a=feats, perm=[0,2,1])
            grams = tf.matmul(feats_T, feats) / tf.cast(size , dtype=float)
            style_gram = self.Style_features[style_layer]
            self.style_losses.append(2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size)
            # self.style_losses = tf.math.add(self.style_losses , (2 * tf.nn.l2_loss(grams - style_gram)/style_gram.size))

        self.style_loss = self.STYLE_WEIGHT * functools.reduce(tf.add , self.style_losses) / self.batch_size

        # total variation denoising
        tv_y_size = tf.cast(self._tensor_size(self.fake[:,1:,:,:]) , dtype=float)
        tv_x_size = tf.cast(self._tensor_size(self.fake[:,:,1:,:]) , dtype=float)
        y_tv = tf.cast(tf.nn.l2_loss(self.fake[:,1:,:,:] - self.fake[:,:self.batch_shape[1]-1,:,:]) , dtype=float)
        x_tv = tf.cast(tf.nn.l2_loss(self.fake[:,:,1:,:] - self.fake[:,:,:self.batch_shape[2]-1,:]) , dtype=float)
        self.tv_loss = tf.cast( self.TV_WEIGHT  , dtype=float) *2* (x_tv/tv_x_size + y_tv/tv_y_size)/ tf.cast(self.batch_size , dtype=float)
        # self.tv_loss = tf.cast( self.TV_WEIGHT  , dtype=float) *2* (x_tv/tv_x_size + y_tv/tv_y_size)/ tf.cast(self.batch_size , dtype=float)
        # loss = content_loss + style_loss + tv_loss

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////
        # /////////  The perceptual lost from the VGG16 should be calculated and added to the generator lost  //////////
        # //////////////////////////////////////////////////////////////////////////////////////////////////////////////

        self.AdLoss = (10000 * self.criterionGAN(self.D_fake, tf.ones_like(self.D_fake)))
        self.g_loss =  self.AdLoss + self.style_loss + self.content_loss 
        # ////////////////////////////////////////////////////////////////////////////////////////////////////

        self.fake_sample = tf.placeholder(tf.float32,
                                            [None, 1024, 768,
                                             self.output_c_dim], name='fake_sample')

        # ///////////////////////////////////////////////////////////////////////////////////////////////////
        self.D_fake_sample = self.discriminator(self.fake_sample, self.options, reuse=True, name="discriminator")
        self.D_real = self.discriminator(self.style, self.options, reuse=True, name="discriminator")

        self.d_loss_real = self.criterionGAN(self.D_real, tf.ones_like(self.D_real))
        self.d_loss_fake = self.criterionGAN(self.D_fake_sample, tf.zeros_like(self.D_fake_sample))
        self.d_loss = 10000 * (self.d_loss_real + self.d_loss_fake) / 2

        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)

        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_sum = tf.summary.merge(
            [self.d_loss_sum, self.d_loss_real_sum, self.d_loss_fake_sum]
        )

# ///////////////////////////////////////////////////////
        self.test_A = tf.placeholder(tf.float32,
                                     [None, 1024, 768,
                                      self.input_c_dim], name='test')
# ///////////////////////////////////////////////////////

        self.testB = self.generator(self.test_A ,  self.options, True, name="generator")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        for var in t_vars: print(var.name)


    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()
        
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            # //////// Data A contains the Object and Data B containg the target image ////////////////////
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            # dataC = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainC'))


            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            # np.random.shuffle(dataC)

            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            print("EOCOHH" , batch_idxs)
            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))

                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # with tf.Graph().as_default(), tf.device('/cpu:0'), tf.compat.v1.Session() as sess:
                #     self.style_image = tf.compat.v1.placeholder(tf.float32, shape=style_shape, name='style_image')
                #     self.style_image_pre = vgg.preprocess(self.style_image)
                #     self.net = vgg.net(self.vgg_path, self.style_image_pre)
                #     self.style_pre = np.array([self.style])
                #     for layer in self.STYLE_LAYERS:
                #         self.features = self.net[layer].eval(feed_dict={self.style_image:self.style_pre})
                #         self.features = np.reshape(self.features, (-1, self.features.shape[3]))
                #         self.gram = np.matmul(self.features.T, self.features) / self.features.size
                #         self.style_features[layer] = self.gram

                # Update G network and record fake outputs
                fake_A,  _, summary_str , adloss , Styleloss , Conloss , GLOSS = self.sess.run(
                    [self.fake, self.g_optim, self.g_sum , self.AdLoss , self.style_loss , self.content_loss , self.g_loss],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                # [fake_A] = self.pool([fake_A])

                # Update D network
                _, summary_str , Dloss = self.sess.run(
                    [self.d_optim, self.d_sum , self.d_loss ],
                    feed_dict={self.real_data: batch_images,
                               self.fake_sample: fake_A,
                            #    self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                counter += 1

                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f Ad loss: %4.4f Style loss: %4.4f Contentloss: %4.4f Dloss: %4.4f G loss %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time, adloss , Styleloss , Conloss , Dloss , GLOSS )))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 1:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        # dataC = dataB

        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        # np.random.shuffle(dataC)

        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A = self.sess.run(
            [self.fake],
            feed_dict={self.real_data: sample_images}
        )
        fake_A  = np.squeeze(np.asarray(fake_A), 0)


        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        # save_images(fake_B, [self.batch_size, 1],
        #             './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else None

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir, os.path.basename(sample_file))
            # ind = image_path.find('.')
            # Sample = image_path[0:ind]
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_test_images(fake_img, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
