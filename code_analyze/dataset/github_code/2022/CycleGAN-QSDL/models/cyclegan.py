import numpy as np
import os
import random
import matplotlib.pyplot as plt
import scipy.io as io
from sklearn.utils import resample
from datetime import datetime
from scipy.io import savemat

import scipy.misc
import scipy.io

from models.models import *
from util.loss import *

class CycleGAN:
    def __init__(self, config,is_train, restore_ckpt, model_dir='./Network/cycleGAN/',summary_dir='./log/'):
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        
        
        #Training parameters
        self.patch_size = 128
        # self.whole_size = 192
        self.restore_ckpt = restore_ckpt     #
        self.lambda_a = config.lambda_a     #   1
        self.lambda_b = config.lambda_b     #   1
        self.pool_size = config.pool_size   #  50
        self.base_lr_g = config.base_lr_g
        self.base_lr_d = config.base_lr_d
        self.max_step = config.epochs
        self.batch_size = config.batch
        self.alpha = tf.random_uniform(
            shape=[self.batch_size,self.patch_size,self.patch_size, 1],
            minval=0.,
            maxval=1
        )

        self.model_dir = model_dir
        self.summary_dir = summary_dir

        #Initalize fake images
        self.fake_a = []
        self.fake_b = []

        self.train_y_path = './DATA/train_label/'
        self.train_x_path = './DATA/train_bp/'
        self.val_y_path = './DATA/val_label/'
        self.val_x_path = './DATA/val_bp/'

    def get_batch_data(self,data,idx):

        sub_data = data[idx:idx+self.batch_size]

        return sub_data
    def get_randam_patches(self, LDCT_slice, NDCT_slice):

        h = w = self.patch_size

        # patch image center(coordinate on whole image)

        h_pc, w_pc = np.random.choice(range(0, 64)), np.random.choice(range(0, 64))

        LDCT_patch = LDCT_slice[:,h_pc:(h_pc+h),w_pc:(w_pc+w),:]

        NDCT_patch = NDCT_slice[:,h_pc:(h_pc+h),w_pc:(w_pc+w),:]


        return LDCT_patch, NDCT_patch

    def setup(self):
        self.real_a = tf.placeholder(tf.float32, 
                                     [None, self.patch_size, self.patch_size, 1],
                                     name='input_a')
        self.real_b = tf.placeholder(tf.float32, 
                                     [None, self.patch_size, self.patch_size, 1],
                                     name='input_b')
        self.fake_pool_a = tf.placeholder(tf.float32, 
                                          [None, self.patch_size, self.patch_size, 1],
                                          name='fake_pool_a')
        self.fake_pool_b = tf.placeholder(tf.float32, 
                                          [None, self.patch_size, self.patch_size, 1],
                                          name='fake_pool_b')
        self.NMI = tf.placeholder(tf.float32, 
                                  
                                  name='NMI')  
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        self.num_fake_inputs = 0
        self.n_fake_a = 0
        self.n_fake_b = 0
        self.lr_g = tf.placeholder(tf.float32, shape=[], name='lr_g')
        self.lr_d = tf.placeholder(tf.float32, shape=[], name='lr_d')

        self.train_y_dir = os.listdir(self.train_y_path)
        self.train_x_dir = os.listdir(self.train_x_path)
        self.val_y_dir = os.listdir(self.val_y_path)
        self.val_x_dir = os.listdir(self.val_x_path)
        self.train_gen = self.batch_gen(self.train_y_path, self.train_x_path, self.train_y_dir, self.train_x_dir, self.batch_size)
        self.val_gen = self.batch_gen(self.val_y_path, self.val_x_path, self.val_y_dir, self.val_x_dir,self.batch_size)
        self.forward()
    
    def forward(self):
        with tf.variable_scope('CycleGAN') as scope:
            #D(A), D(B)
            self.p_real_a = discriminator(self.real_a, scope='d_a')
            self.p_real_b = discriminator(self.real_b, scope='d_b')
            
            #G(A), G(B)
            self.fake_img_b = generator(self.real_a, scope='g_a')
            self.fake_img_a = generator(self.real_b, scope='g_b')
            
            scope.reuse_variables()
            
            #D(G(B)), D(G(A))
            self.p_fake_a = discriminator(self.fake_img_a, scope='d_a')
            self.p_fake_b = discriminator(self.fake_img_b, scope='d_b')

            #G(G(A)), G(G(B))
            self.cycle_a = generator(self.fake_img_b, scope='g_b')
            self.cycle_b = generator(self.fake_img_a, scope='g_a')

            scope.reuse_variables()
            
            self.p_fake_aa = generator(self.real_a, scope='g_b')
            self.p_fake_bb = generator(self.real_b, scope='g_a')
            
            scope.reuse_variables()

            #Fake pool for discriminator loss
            self.p_fake_pool_a = discriminator(self.fake_pool_a, scope='d_a')
            self.p_fake_pool_b = discriminator(self.fake_pool_b, scope='d_b')



    #Calculate NMI function
    def nmi(self,x,y):
        size = x.shape[-1]
        x = np.array(x)
        y = np.array(y)
        px = np.histogram(x, 256, (0, 255))[0] / size
        py = np.histogram(y, 256, (0, 255))[0] / size
        hx = - np.sum(px * np.log(px + 1e-8))
        hy = - np.sum(py * np.log(py + 1e-8))
 
        hxy = np.histogram2d(x, y, 256, [[0, 255], [0, 255]])[0]
        hxy /= (1.0 * size)
        hxy = - np.sum(hxy * np.log(hxy + 1e-8))
 
        r = hx + hy - hxy
        nmi = 2*r/(hx+hy)
        return nmi




    def batch_gen(self,jpg_y_path, jpg_x_path, y_path, x_path, batch_size):
        while 1:
            jpg_x_name = resample(x_path, n_samples=batch_size)
            jpg_test_y = []
            jpg_test_x = []
            nmi_array = []
            for i in range(batch_size):
                jpg_y = io.loadmat(jpg_y_path + jpg_x_name[i])['img']
                jpg_x = io.loadmat(jpg_x_path + jpg_x_name[i])['fbp']
                
                img_x = jpg_x
                img_y = jpg_y
                y = np.reshape(img_y,-1)
                x = np.reshape(img_x,-1)
                mi = self.nmi(x, y)
                nmi_array.append(mi)
                
                jpg_test_y.append(jpg_y)
                jpg_test_x.append(jpg_x)
            jpg_test_y = np.array(jpg_test_y, dtype=np.float32)
            jpg_test_x = np.array(jpg_test_x, dtype=np.float32)
            nmi_array = np.array(nmi_array, dtype=np.float32)
            jpg_test_y = jpg_test_y.reshape((-1, 128, 128, 1))
            jpg_test_x = jpg_test_x.reshape((-1, 128, 128, 1))
            nmi_array = nmi_array.reshape((-1, 1, 1, 1))

            yield jpg_test_y, jpg_test_x, nmi_array

    def loss(self):
        #Cycle consistency loss
        self.cyclic_loss_a = cyclic_loss(self.real_a, self.cycle_a)
        self.cyclic_loss_b = cyclic_loss(self.real_b, self.cycle_b)


        #LSGAN loss
        self.lsgan_loss_a = generator_loss(type = "lsgan", fake = self.p_fake_a)
        self.lsgan_loss_b = generator_loss(type = "lsgan", fake = self.p_fake_b)


        #QSDL loss
        self.qsdl_loss_a = QSDL_loss(self.real_a, self.fake_img_a, self.NMI)
        self.qsdl_loss_b = QSDL_loss(self.real_b, self.fake_img_b, self.NMI)
        

        #Identity loss
        self.identity_loss_a = cyclic_loss(self.real_a, self.p_fake_aa)/2
        self.identity_loss_b = cyclic_loss(self.real_b, self.p_fake_bb)/2


        #Generator loss
        self.g_a_loss = self.cyclic_loss_a + self.cyclic_loss_b + self.lsgan_loss_b + self.qsdl_loss_a + self.identity_loss_a
        self.g_b_loss = self.cyclic_loss_b + self.cyclic_loss_a + self.lsgan_loss_a + self.qsdl_loss_b + self.identity_loss_b
        self.g_loss = self.g_a_loss + self.g_b_loss


        #Discriminator loss
        self.d_a_loss = discriminator_loss(type = "lsgan", real = self.p_real_a, fake = self.p_fake_pool_a)
        self.d_b_loss = discriminator_loss(type = "lsgan", real = self.p_real_b, fake = self.p_fake_pool_b)
        self.d_loss = self.d_a_loss + self.d_b_loss


        #Isolate variables
        self.vars = tf.trainable_variables()
        d_a_vars = [v for v in self.vars if 'd_a' in v.name]
        d_b_vars = [v for v in self.vars if 'd_b' in v.name]
        g_a_vars = [v for v in self.vars if 'g_a' in v.name]
        g_b_vars = [v for v in self.vars if 'g_b' in v.name]

        #Train while freezing other variables
        optimizer = tf.train.AdamOptimizer(self.lr_d, beta1=0.5)
        self.d_train = optimizer.minimize(self.d_loss, var_list=d_b_vars+d_a_vars)

        optimizer = tf.train.AdamOptimizer(self.lr_g, beta1=0.5)
        self.g_train = optimizer.minimize(self.g_loss, var_list=g_a_vars+g_b_vars)


        #Summary
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.cyclic_a_summary = tf.summary.scalar('cyclic_a_summary', self.cyclic_loss_a)
        self.cyclic_b_summary = tf.summary.scalar('cyclic_b_summary', self.cyclic_loss_b)
        self.lsgan_a_summary = tf.summary.scalar('lsgan_a_summary', self.lsgan_loss_a)
        self.lsgan_b_summary = tf.summary.scalar('lsgan_b_summary', self.lsgan_loss_b)
        self.g_qsdl_a_summary = tf.summary.scalar('g_qsdl_a_summary', self.qsdl_loss_a)
        self.g_qsdl_b_summary = tf.summary.scalar('g_qsdl_b_summary', self.qsdl_loss_b)
        self.identity_a_summary = tf.summary.scalar('identity_a_summary', self.identity_loss_a)
        self.identity_b_summary = tf.summary.scalar('identity_b_summary', self.identity_loss_b)
        self.input_ii = tf.summary.image('input',self.real_a)
        self.cycle_ii = tf.summary.image('cycle',self.fake_img_b)
        self.real_ii = tf.summary.image('real', self.real_b)

    def fake_pool(self, fake, pool, n_fake):
        assert len(pool) <= self.pool_size
        results = []
        for i in range(self.batch_size):
            if n_fake < self.pool_size:
                # pool[n_fake] = fake[i]
                pool.append(fake[i])
                results.append(fake[i])
            else:
                p = random.random()
                if p < 0.5:
                    index = random.randint(0, self.pool_size - 1)
                    temp = pool[index]
                    pool[index] = fake[i]
                    results.append(temp)
                else:
                    results.append(fake[i])
            n_fake += 1
        return np.array(results)


    def train(self):

        self.setup()
        self.loss()

        # total = len(data)
        total = 20000

        init = (tf.global_variables_initializer(), 
                tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep = 200)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            sess.run(init)
            
            if self.restore_ckpt:
                ckpt_name = tf.train.latest_checkpoint(self.model_dir)
                saver.restore(sess, ckpt_name)
            if not os.path.exists(self.summary_dir):
                os.makedirs(self.summary_dir)
            writer = tf.summary.FileWriter(self.summary_dir + '/train')
            writer2 = tf.summary.FileWriter(self.summary_dir +'/test')

            print("Start training ... ")
            for epoch in range(sess.run(self.global_step), self.max_step):
                saver.save(sess, self.model_dir, global_step=epoch)
                
                if epoch < 100:
                    current_lr_g = self.base_lr_g
                    current_lr_d = self.base_lr_d
                else:
                    current_lr_g = self.base_lr_g - self.base_lr_g * (epoch - 100)/100
                    current_lr_d = self.base_lr_d - self.base_lr_d * (epoch - 100)/100

                for i in range(0,total,self.batch_size):
                    batch_label, batch_data, nmi_array = next(self.train_gen)

                    ###############################Optimize G##########################################################################
                    run_list = [self.g_train, self.fake_img_b, self.fake_img_a,
                                self.g_loss_summary,
                                self.cyclic_a_summary, self.cyclic_b_summary,
                                self.lsgan_a_summary, self.lsgan_b_summary, 
                                self.g_qsdl_a_summary, self.g_qsdl_b_summary,
                                self.identity_a_summary, self.identity_b_summary,
                                self.g_loss, self.g_b_loss, self.g_a_loss]
                    feed_dict = {self.real_a: batch_data, self.real_b: batch_label, self.NMI: nmi_array,
                                 self.lr_g: current_lr_g}
                    _, fake_b_temp, fake_a_temp, summary, summary_cyc_a, summary_cyc_b, summary_ls_a, summary_ls_b, summary_qsdl_a, summary_qsdl_b, summary_idt_a, summary_idt_b, g_loss, g_b_loss, g_a_loss = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    writer.add_summary(summary_cyc_a, epoch * total + i)
                    writer.add_summary(summary_cyc_b, epoch * total + i)
                    writer.add_summary(summary_ls_a, epoch * total + i)
                    writer.add_summary(summary_ls_b, epoch * total + i)
                    writer.add_summary(summary_qsdl_a, epoch * total + i)
                    writer.add_summary(summary_qsdl_b, epoch * total + i)
                    writer.add_summary(summary_idt_a, epoch * total + i)
                    writer.add_summary(summary_idt_b, epoch * total + i)

                    #Sample from fake  pool
                    fake_b_sample = self.fake_pool(fake_b_temp, self.fake_b, self.n_fake_b)
                    self.n_fake_b += self.batch_size
                    fake_a_sample = self.fake_pool(fake_a_temp, self.fake_a, self.n_fake_a)
                    self.n_fake_a += self.batch_size

                    #################################Optimize D########################################################################
                    run_list = [self.d_train, self.d_loss_summary, self.d_loss, self.d_a_loss, self.d_b_loss]
                    feed_dict = {
                             self.real_a: batch_data, self.real_b: batch_label,
                             self.lr_d: current_lr_d,
                             self.fake_pool_b: fake_b_sample,
                             self.fake_pool_a: fake_a_sample
                            }
                    _, summary, d_loss, d_a_loss, d_b_loss = sess.run(run_list, feed_dict)
                    writer.add_summary(summary, epoch * total + i)
                    if epoch % 1 == 0 and i % 671 == 0:
                        batch_label_test, batch_data_test, nmi_array_test = next(self.val_gen)
                        run_list = [self.g_loss_summary,
                                    self.cyclic_a_summary, self.cyclic_b_summary,
                                    self.lsgan_a_summary, self.lsgan_b_summary,
                                    self.g_qsdl_a_summary, self.g_qsdl_b_summary,
                                    self.identity_a_summary, self.identity_b_summary,
                                    self.input_ii, self.cycle_ii, self.real_ii]
                        feed_dict = {self.real_a: batch_data_test, self.real_b: batch_label_test, self.NMI: nmi_array_test,
                                     self.lr_g: current_lr_g}
                        summary, summary_cyc_a, summary_cyc_b, summary_ls_a, summary_ls_b, summary_qsdl_a, summary_qsdl_b, summary_idt_a, summary_idt_b, input_summary, cycle_summary, label_summary = sess.run(run_list, feed_dict)
                        writer2.add_summary(summary, epoch * total + i)
                        writer2.add_summary(summary_cyc_a, epoch * total + i)
                        writer2.add_summary(summary_cyc_b, epoch * total + i)
                        writer2.add_summary(summary_ls_a, epoch * total + i)
                        writer2.add_summary(summary_ls_b, epoch * total + i)
                        writer2.add_summary(summary_qsdl_a, epoch * total + i)
                        writer2.add_summary(summary_qsdl_b, epoch * total + i)
                        writer2.add_summary(summary_idt_a, epoch * total + i)
                        writer2.add_summary(summary_idt_b, epoch * total + i)
                        writer2.add_summary(input_summary, epoch * total + i)
                        writer2.add_summary(cycle_summary, epoch * total + i)
                        writer2.add_summary(label_summary, epoch * total + i)

                    writer.flush()

                    if total-i <= self.batch_size:
                        print('Epoch: %d - gen_loss: %.6f - g_a_loss: %.6f - g_b_loss: %.6f - d_loss: %.6f - d_a_loss: %.6f - d_b_loss: %.6f ' % (epoch, g_loss, g_a_loss, g_b_loss, d_loss, d_a_loss, d_b_loss))
                        # print('Epoch: %d - gen_loss: %.6f - g_a_loss: %.6f - g_b_loss: %.6f - d_loss: %.6f - d_a_loss: %.6f - d_b_loss: %.6f' % (epoch, g_loss, g_a_loss, g_b_loss, d_loss, d_a_loss, d_b_loss))

                    
                sess.run(tf.assign(self.global_step, epoch + 1))
                # saver.save(sess, self.model_dir+'cycleWGAN_'+repr(epoch+1)+'_'+repr(i+1)+'.ckpt')
                if epoch % 10 == 0:
                    saver.save(sess, self.model_dir + 'cycleWGAN_' + repr(epoch + 1) + '.ckpt')
                # saver.save(sess, self.model_dir, global_step=epoch + 1)
            writer.add_graph(sess.graph)
    
    def test(self):
        self.setup()

        saver = tf.train.Saver()
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, './Network/cycleGAN/cycleWGAN_151.ckpt')
            for i in range(2500):
                test_bp = io.loadmat('./patient_one/5e4/bp_norm/{0:04d}.mat'.format(i + 1))['fbp']
                test_bp = np.reshape(test_bp, (1, 128, 128, 1))
                test_img = io.loadmat('./patient_one/5e4/img_norm/{0:04d}.mat'.format(i + 1))['img']
                test_img = np.reshape(test_img, (1, 128, 128, 1))

                imgs = [self.fake_img_a,self.fake_img_b,self.cycle_a,self.cycle_b]
                b2a,a2b,aba,bab = sess.run(imgs,feed_dict={self.real_a: test_bp, self.real_b: test_img})
                a2b = np.reshape(a2b, [128, 128])
                b2a = np.reshape(b2a, [128, 128])
                aba = np.reshape(aba, [128, 128])
                bab = np.reshape(bab, [128, 128])
                savemat('./patient_one/recon/fake_img_a/{0:04d}.mat'.format(i + 1), {'img': b2a})
                savemat('./patient_one/recon/fake_img_b/{0:04d}.mat'.format(i + 1), {'img': a2b})
                savemat('./patient_one/recon/cycle_a/{0:04d}.mat'.format(i + 1), {'img': aba})
                savemat('./patient_one/recon/cycle_b/{0:04d}.mat'.format(i + 1), {'img': bab})
        return 0