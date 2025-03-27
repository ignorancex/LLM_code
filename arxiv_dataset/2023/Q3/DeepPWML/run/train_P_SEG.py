
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from models.modules import T_SEG_module,CLS_module, CMG_module, P_SEG_module
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


class Train_P_seg:
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
        self.lr = 1e-4
        self.epochs = 150
        self.train_nums_per_epoch = 300
        self.val_nums_per_epoch = 50
        self.model_select = False

        self.valid_loss, self.compare_loss = 0, 100
        tf.keras.backend.set_image_data_format("channels_last")

        self.cls_loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def build_model(self):
        # ==================Initialize Networks==========================
        self.P_SEG = P_SEG_module().build_model()
        print(self.P_SEG.name)
        self.train_vars = []
        self.train_vars += self.P_SEG.trainable_variables
        # ============load trained T_SEG & CMG to train P_SEG============
        self.trained_T_SEG = T_SEG_module().build_model()
        self.trained_CMG = CMG_module().build_model()

        print(f"trained_T_SEG loaded from {self.T_SEG_trained_weight_path}")
        self.trained_T_SEG.load_weights(self.T_SEG_trained_weight_path)

        print(f"trained_CMG loaded from {self.CMG_trained_weight_path}")
        self.trained_CMG.load_weights(self.CMG_trained_weight_path)

        for layer in self.trained_T_SEG.layers:
            layer.trainable = False
        for layer in self.trained_CMG.layers:
            layer.trainable = False
        print("end build_model")

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

    def dice_coefficient(self, y_true, y_pred, smooth=0.00001):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def P_seg_train_one_batch(self, dat_all, label_cubes, dat_mask, gen_optim, train_vars):
        switch_stat = np.zeros((len(dat_all), 1))
        switch_stat_one_hot = to_categorical(switch_stat)

        t1, t2, t3 = self.codemap(condition=switch_stat_one_hot)
        cfmap = \
            self.trained_CMG({"input": dat_all, "mask": dat_mask, "c1": t1, "c2": t2, "c3": t3})[
                "CF_output"]
        SPmap = self.trained_T_SEG(dat_all)

        pwml_input = np.concatenate((dat_all, cfmap, SPmap), axis=-1)

        with tf.GradientTape() as tape:
            pwml_mask = self.P_SEG(pwml_input)
            label_cubes = tf.cast(label_cubes, dtype=tf.float32)
            loss = 1 - self.dice_coefficient(label_cubes, pwml_mask)
        grads = tape.gradient(loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))
        return loss

    def P_seg_val_block(self, val_dat_all, val_dat_label, val_mask):
        switch_stat = np.zeros((len(val_dat_all), 1))
        switch_stat_one_hot = to_categorical(switch_stat)

        t1, t2, t3 = self.codemap(condition=switch_stat_one_hot)
        cfmap = \
            self.trained_CMG({"input": val_dat_all, "mask": val_mask, "c1": t1, "c2": t2, "c3": t3})[
                "CF_output"]
        SPmap = self.trained_T_SEG(val_dat_all)
        pwml_input = np.concatenate((val_dat_all, cfmap, SPmap), axis=-1)
        val_dat_label = tf.cast(val_dat_label, dtype=tf.float32)
        pwml_mask = self.P_SEG(pwml_input)
        val_loss = 1 - self.dice_coefficient(val_dat_label, pwml_mask)
        return val_loss

    def main(self):
        self.build_model()
        optim = Adam(lr=self.lr)
        P_SEG_loss_epochs = []
        val_P_SEG_loss_epochs = []
        train_gen = DataGen_CLS_CMG_P_SEG(self.train_dir, self.train_gt_dir, self.train_datalist).make_gen_P_SEG()
        val_gen = DataGen_CLS_CMG_P_SEG(self.val_dir, self.val_gt_dir, self.val_datalist).make_gen_P_SEG()

        for cur_epoch in tqdm.trange(self.epochs, desc="%s" % f"{self.epochs}"):

                P_loss_batch = []
                #training
                for cur_step in tqdm.trange(self.train_nums_per_epoch,
                                            desc="%s" % (f"{cur_epoch}/{self.epochs}")):
                    data, label_cubes, _, mask = next(train_gen)
                    data = data['input']
                    label_cubes = label_cubes['P_label']
                    mask = mask['mask'].astype(float)
                    P_loss = self.P_seg_train_one_batch(data, label_cubes, mask, optim, self.train_vars)

                    P_loss_batch.append(P_loss)
                    print(f"P_loss：{np.mean(P_loss_batch)}")

                print(f"{cur_epoch}epochs P_loss mean：{np.mean(P_loss_batch)} \n")
                P_SEG_loss_epochs.append(np.mean(P_loss_batch))

                val_batch = []
                # validation
                for val_step in tqdm.trange(self.val_nums_per_epoch,
                                            desc="validation: %s" % (f"{cur_epoch}/{self.epochs}")):

                    val_data, val_patch_label,_, val_mask = next(val_gen)
                    val_data = val_data['input']
                    val_patch_label = val_patch_label['P_label']
                    val_mask = val_mask['mask'].astype(float)
                    val_loss = self.P_seg_val_block(val_dat_all=val_data, val_dat_label=val_patch_label,
                                              val_mask=val_mask)
                    val_batch.append(val_loss)

                self.valid_loss = np.mean(val_batch)
                val_batch.append(self.valid_loss)
                if self.valid_loss <= self.compare_loss:
                    self.model_select = True
                    self.compare_loss = self.valid_loss

                if self.model_select == True:
                    self.P_SEG.save_weights(
                        os.path.join(self.save_path + '/best_P_seg.h5'))
                    self.model_select = False
                val_P_SEG_loss_epochs.append(np.mean(val_batch))

        print("````````````````train````````````````")
        P_SEG_loss_epochs = np.array(P_SEG_loss_epochs)
        print(f"min train_loss:{min(P_SEG_loss_epochs)}")
        print(f"which min loss epoch:{np.argmin(P_SEG_loss_epochs) + 1}")
        print("````````````````val````````````````")
        val_P_SEG_loss_epochs = np.array(val_P_SEG_loss_epochs)
        print(f"min val_loss:{min(val_P_SEG_loss_epochs)}")
        print(f"which min loss epoch:{np.argmin(val_P_SEG_loss_epochs) + 1}")

        plt.figure()
        plt.plot(range(self.epochs), P_SEG_loss_epochs, label='train_loss')

        plt.plot(range(self.epochs), val_P_SEG_loss_epochs, label='val_loss')
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('P_seg')
        plt.show()


Train_P_seg = Train_P_seg()
Train_P_seg.main()




