
import os
from functools import partial
import datetime
import numpy as np
import tensorflow.python.keras
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import clear_session, set_session
from tensorflow.python.keras.optimizers import adam_v2
import tensorflow.python.keras as keras
from models.modules import T_SEG_module, cross_entropy

from util.data_prep import DataGen_T_SEG
from util.prep import maybe_mkdir_p
from util.misc import save_params, cosine_annealing

def main():

    # ======================================
    #       Set Environment Variable      #
    # ======================================
    save_path = "Network_weight_save_path"
    maybe_mkdir_p(save_path)
    save_file_name = os.path.join(
        save_path, '{epoch:02d}.h5')
    train_dir = 'T1_data_path'
    train_gt_dir = 'label_path'
    val_dir = 'T1_data_path'
    val_gt_dir = 'label_path'
    train_datalist = "train_data_id_list"
    val_datalist = "val_data_id_list"
    train_datalist = range(24, 31)
    val_datalist = range(31, 32)

    lr_init = 1e-3#5e-6
    lr_min = 1e-5#1e-9
    epochs = 150
    train_nums_per_epoch = 300
    val_nums_per_epoch = 50
    # ======================================
    #       Close Useless Information     #
    # ======================================

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

    model = T_SEG_module().build_model()

    save_schedule = keras.callbacks.ModelCheckpoint(filepath=save_file_name, period=1)

    # ===================================================================
    #        Set Training Callbacks and Initialize Data Generator      #
    # ===================================================================
    train_gen = DataGen_T_SEG(train_dir, train_gt_dir, train_datalist).make_gen()
    val_gen = DataGen_T_SEG(val_dir, val_gt_dir, val_datalist).make_gen()

    lr_schedule_fn = partial(cosine_annealing, lr_init=lr_init, lr_min=lr_min)

    lr_schedule = keras.callbacks.LearningRateScheduler(lr_schedule_fn)

    call_backs = [lr_schedule, save_schedule]
    model.compile(optimizer=adam_v2.Adam(lr=lr_init), loss=cross_entropy,metrics=["accuracy"])

    history = model.fit(x=train_gen,
                        steps_per_epoch=train_nums_per_epoch,
                        epochs=epochs,
                        validation_data=val_gen,
                        validation_steps=val_nums_per_epoch,
                        callbacks=call_backs)

    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    print("````````````````train````````````````")
    train_loss = np.array(train_loss)
    print(f"min train_loss:{min(train_loss)}")
    print(f"which min loss epoch:{np.argmin(train_loss) + 1}")

    print(f"max val_accuracy:{max(train_accuracy)}")
    train_accuracy = np.array(train_accuracy)
    print(f"which max accuracy epoch:{np.argmax(train_accuracy) + 1}")
    print("\t\n")

    print("````````````````val````````````````")
    val_loss = np.array(val_loss)
    print(f"min val_loss:{min(val_loss)}")
    print(f"which min loss epoch:{np.argmin(val_loss) + 1}")

    print(f"max val_accuracy:{max(val_accuracy)}")
    val_accuracy = np.array(val_accuracy)
    print(f"which max accuracy epoch:{np.argmax(val_accuracy) + 1}")

    # figure 1
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('CLS_train')

    # figure 2
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('CLS_train')
    plt.show()

    print(f"'train_accuracy'max{max(train_accuracy)}")

    print(f"'val_accuracy'max{max(val_accuracy)}")
    print(f"'val_accuracy'mean{sum(val_accuracy[-20:-1])/len(val_accuracy[-20:-1])}")

if __name__ == "__main__":
    main()