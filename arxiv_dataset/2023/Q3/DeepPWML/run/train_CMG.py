
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from models.modules import CLS_module, CMG_module
from util.data_prep import DataGen_CLS_CMG_P_SEG
from util.prep import maybe_mkdir_p
from util.misc import save_params, cosine_annealing
from util.loss import ACC, cycle_loss, L1_norm, L2_norm, CE_loss, Tv_loss

K = keras.backend

# ==============================================================
#         Set GPU Environment And Initialize Networks         #
# ==============================================================

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
clear_session()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True


class Train_CMG:
    def __init__(self):
        self.save_path = "Network_weight_save_path"
        self.CLS_trained_weight_path = "CLS_trained_weight_path"
        maybe_mkdir_p(self.save_path)
        self.train_dir = 'T1_data_path'
        self.train_gt_dir = 'label_path'
        self.val_dir = 'T1_data_path'
        self.val_gt_dir = 'label_path'
        self.train_datalist = "train_data_id_list"
        self.val_datalist = "val_data_id_list"
        self.lr = 1e-3
        self.epochs = 150
        self.train_nums_per_epoch = 300

        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def build_model(self):
        # ==================Initialize Networks==========================
        self.CMG = CMG_module().build_model()
        print(self.CMG.name)
        self.train_vars = []
        self.train_vars += self.CMG.trainable_variables
        #  load trained CLS to train CMG
        self.trained_CLS = CLS_module().build_model()
        print(f"trained_CLS loaded from {self.CLS_trained_weight_path}")
        self.trained_CLS.load_weights(self.CLS_trained_weight_path)
        for layer in self.trained_CLS.layers:
            layer.trainable = False

    def codemap(self, condition):
        size1, size2, size3 = 32, 16, 8
        c1 = np.zeros((len(condition), size1, size1, size1, 2))
        c2 = np.zeros((len(condition), size2, size2, size2, 2))
        c3 = np.zeros((len(condition), size3, size3, size3, 2))
        for batch in range(len(condition)):
            for classes in range(condition.shape[-1]):
                c1[batch, ..., classes], c2[batch, ..., classes] = condition[batch, classes], condition[batch, classes]
                c3[batch, ..., classes] = condition[batch, classes]
        return c1, c2, c3

    def CMG_train_one_batch(self, dat_all, dat_mask, gen_optim, train_vars):
        label_softmax = self.trained_CLS(dat_all)
        label = np.argmax(label_softmax, axis=1)

        switch_stat = np.ones_like(label) - label
        label_one_hot = to_categorical(label)
        switch_stat_one_hot = to_categorical(switch_stat)

        t1, t2, t3 = self.codemap(condition=switch_stat_one_hot)
        with tf.GradientTape() as tape:
            cfmap = \
            self.CMG({"input": dat_all, "mask": dat_mask, "c1": t1, "c2": t2, "c3": t3}, training=True)[
                "CF_output"]

            CF_map_add = tf.where(switch_stat[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], cfmap, 0)
            CF_map_reduce = tf.where(label[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], cfmap, 0)

            pseudo_image = dat_all + CF_map_add - CF_map_reduce
            pred = self.trained_CLS(pseudo_image)
            l1_loss, l2_loss = L1_norm(effect_map=cfmap), L2_norm(effect_map=cfmap)
            cls_loss = self.cls_loss(switch_stat, pred)

            l1 = 10.0 * l1_loss
            l2 = 10.0 * l2_loss
            cls = 1.0 * cls_loss
            train_CMG_loss = l1 + l2 + cls

        grads = tape.gradient(train_CMG_loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))
        return train_CMG_loss, cls, l1, l2

    def CMG_val_block(self, CMG, CLS):
        print("CMG_val")

    def main(self):
        self.build_model()
        optim = Adam(lr=self.lr)

        CMG_loss_epochs,cls_loss_epochs,l1_epochs,l2_epochs = [], [], [], []
        train_gen = DataGen_CLS_CMG_P_SEG(self.train_dir, self.train_gt_dir, self.train_datalist).make_gen_CMG()

        val_cls_epochs = []
        # val_seg_epochs = []

        for cur_epoch in tqdm.trange(self.epochs, desc="%s" % f"{self.epochs}"):

                CMG_loss_batch, cls_loss_batch, l1_batch, l2_batch = [], [], [], []
                for cur_step in tqdm.trange(self.train_nums_per_epoch,
                                            desc="%s" % (f"{cur_epoch}/{self.epochs}")):
                    data, class_label, mask = next(train_gen)
                    data = data['input']
                    mask = mask['mask'].astype(float)
                    class_label = class_label['class_label']
                    CMG_loss, cls_loss, l1, l2 = self.CMG_train_one_batch( data, mask, optim, self.train_vars)

                    CMG_loss_batch.append(CMG_loss)
                    cls_loss_batch.append(cls_loss)
                    l1_batch.append(l1)
                    l2_batch.append(l2)
                    print(f"CMG_loss：{np.mean(CMG_loss_batch)}")

                print(f"{cur_epoch}epochs下的平均cls_loss：{np.mean(cls_loss_batch)} \n")
                print(f"{cur_epoch}epochs下的平均CMG_loss：{np.mean(CMG_loss_batch)} \n")
                #validation
                # val_CMG_loss, val_cls_loss, val_l1, val_l2 = self.CMG_val_block(self.CMG, self.trained_CLS)
                # print(f"{cur_epoch}epochs下的平均val_cls_loss：{val_cls_loss} \n")
                # print(f"{cur_epoch}epochs下的平均val_CMG_loss：{val_CMG_loss} \n")
                # print(f"{cur_epoch}epochs下的平均val_l1_loss：{val_l1} \n")
                # print(f"{cur_epoch}epochs下的平均val_l2_loss：{val_l2} \n")

                self.CMG.save_weights(
                    os.path.join(self.save_path + '/%d_CMG.h5' % (cur_epoch + 1)))
                CMG_loss_epochs.append(np.mean(CMG_loss_batch))
                cls_loss_epochs.append(np.mean(cls_loss_batch))
                l1_epochs.append(np.mean(l1_batch))
                l2_epochs.append(np.mean(l2_batch))

        print("````````````````train````````````````")
        CMG_loss_epochs = np.array(CMG_loss_epochs)
        cls_loss_epochs = np.array(cls_loss_epochs)

        print(f"min train_loss:{min(CMG_loss_epochs)}")
        print(f"which min loss epoch:{np.argmin(CMG_loss_epochs) + 1}")
        print(f"min cls_loss:{min(cls_loss_epochs)}")
        print(f"which min cls_loss epoch:{np.argmin(cls_loss_epochs) + 1}")

        plt.figure()
        plt.plot(range(self.epochs), CMG_loss_epochs, label='CMG_loss_epochs')
        plt.plot(range(self.epochs), cls_loss_epochs, label='cls_loss_epochs')
        plt.plot(range(self.epochs), l1_epochs, label='l1_epochs')
        plt.plot(range(self.epochs), l2_epochs, label='l2_epochs')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('CMG_train')
        plt.show()

Train_CMG = Train_CMG()
Train_CMG.main()




