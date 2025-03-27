import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#not show info from tensorflow
import cv2
import numpy as np
from model import *
from data_stream import TFDataGenerator
from data import *
import time
from tf_metric import *
from test_each_class import each_class_statistics

class ResultCompute(ModelComputeTemplate):
    def result(self, data_list, label_list):
        pred_list = self.model(data_list, training=False)
        result_dict = {}

        # get raw output
        raw_output = pred_list[2].numpy()
        result_dict["raw_output"] = raw_output

        #get max index
        max_index = pred_list[1].numpy()
        result_dict["max_index"] = max_index

        #get softmax output
        softmax_output = pred_list[0].numpy()
        result_dict["softmax_output"] = softmax_output

        #get max raw output
        label = label_list.numpy()
        result_dict["label"] = label

        #get whether answer is correct
        index_output = pred_list[1].numpy()
        correct_list = [1 if index_output[label_index]==label else 0 for label_index, label in enumerate(label_list)]
        result_dict["correct"] = correct_list

        #get raw true output
        raw_true_list = tf.gather(pred_list[2], label_list, batch_dims=1).numpy()
        result_dict["raw_true"] = raw_true_list

        #get raw false output
        batch_size, class_count = pred_list[2].shape[:2]
        index_array = np.array(range(class_count))
        index_map = np.broadcast_to([index_array], (batch_size, class_count))
        index_map = np.array([np.delete(temp_index_array, label[map_index]) for map_index, temp_index_array in enumerate(index_map)])
        raw_false_list = tf.gather(pred_list[2], index_map, batch_dims=1).numpy()
        result_dict["raw_false"] = raw_false_list

        #get raw max output
        raw_max_list = tf.gather(pred_list[2], max_index, batch_dims=1).numpy()
        result_dict["raw_max"] = raw_max_list

        #get raw false output
        batch_size, class_count = pred_list[2].shape[:2]
        index_array = np.array(range(class_count))
        index_map = np.broadcast_to([index_array], (batch_size, class_count))
        index_map = np.array([np.delete(temp_index_array, max_index[map_index]) for map_index, temp_index_array in enumerate(index_map)])
        raw_non_max_list = tf.gather(pred_list[2], index_map, batch_dims=1).numpy()
        result_dict["raw_non_max"] = raw_non_max_list
        return result_dict

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, steps_per_epoch, end_epoch):
        self.initial_learning_rate = initial_learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.end_epoch = end_epoch

    def __call__(self, step):
        epoch = step//self.steps_per_epoch
        lr = self.initial_learning_rate
        lr = tf.where(self.end_epoch-epoch<=10, lr*0.1, lr)

        return lr
class TrainHandle():
    def __init__(self, class_count, input_shape=(32,32,3), model=None,
        adjust_brightness=False, adjust_contrast=False, add_noise=False, 
        adjust_size=True, random_flip=True, **model_info_dict):
        self.class_count = class_count
        self.input_shape = input_shape
        # model
        print("set up model...")
        if isinstance(model, type(None)):
            self.model = MyResnet(class_count, **model_info_dict)
        else:
            self.model = model(class_count, **model_info_dict)
        self.model_name = self.model.__class__.__name__
        print("Model name is", self.model_name)

        init_model_path = "./init_model/"+self.model_name+"_"+str(class_count)+"/init"
        force_out = self.model(np.zeros((1,)+input_shape))
        checkpoint = tf.train.Checkpoint(model=self.model)
        if not os.path.isdir(init_model_path):
            os.makedirs(init_model_path)
        checkpoint.write(init_model_path)
        self.init_model_path = init_model_path

        # data augmentation
        self.data_aug_model = DataAugmentGenerator(\
            adjust_brightness=adjust_brightness, adjust_contrast=adjust_contrast, add_noise=add_noise, \
                adjust_size=adjust_size, random_flip=random_flip)

        print("Setting completed.")
    
    def set_loss_and_optimizer(self, is_train, **setting_dict):#overwrite this to get different training
        # loss and metric
        train_dict = self._get_compute_dict()
        train_dict["loss"]["loss"] = tf.keras.losses.SparseCategoricalCrossentropy()
        train_dict["loss reg"]["loss"] = tf.keras.metrics.Mean(name='train_loss')
        train_dict["acc"]["acc"] = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        val_dict = dict(train_dict)
        
        if is_train:
            lr = setting_dict["lr"]
            train_steps = setting_dict["train_steps"]
            end_epoch = setting_dict["end_epoch"]

            #set optimizer
            lr_schedule = MyLRSchedule(lr, steps_per_epoch=train_steps, end_epoch=end_epoch)
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)#, nesterov=True)

            train_dict["optimizer"] = optimizer

            return train_dict, val_dict
        else:
            return val_dict

    def warm_up_adjust(self, epoch, start_epoch, train_dict, **setting_dict):#overwrite this to get different training
        # print("loss sum weight:", train_dict["loss sum weight"])
        pass

    def train_one_step(self, train_labels, train_imgs, train_dict, **setting_dict):#overwrite this to get different training
        grad_test_list, loss_dict = train_compute(train_labels, train_imgs, self.model, train_dict, training=True)
        for grad_test in grad_test_list:#debug
            if np.any(grad_test):
                print()
                print(loss_dict)
                break

    def test_one_step(self, val_labels, val_imgs, val_dict, **setting_dict):#overwrite this to test differently
        test_compute(val_labels, val_imgs, self.model, val_dict)

    def train(self, epochs, train_data_handle, test_data_handle,
        lr=0.1, batch_size=1024, reinitial=False,
        FUKL=False, orthogonol_regu=False, data_aug=True, start_epoch=0, save_dir='./train/new/',
        **extra_setting_dict):
        #set data
        print("set data...")
        train_iter = train_data_handle.iterator(batch_size)
        train_steps = train_data_handle.get_data_len()//batch_size
        val_iter = test_data_handle.iterator(batch_size)
        val_steps = test_data_handle.get_data_len()//batch_size
        print("train_steps:", train_steps)

        # set loss and optimizer
        train_dict, val_dict = self.set_loss_and_optimizer(is_train=True, lr=lr, batch_size=batch_size,
            train_steps=train_steps, end_epoch=epochs,
            FUKL=FUKL, orthogonol_regu=orthogonol_regu, **extra_setting_dict)

        # load weight if not from scratch
        checkpoint = tf.train.Checkpoint(optimizer=train_dict["optimizer"], model=self.model)
        ckpt_dir = save_dir+'ckpt/'
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
        if start_epoch>0:
            print("load weight...")
            checkpoint.read(ckpt_dir+str(start_epoch-1))
        else:# restore to train many times.
            print("restore initialized model.")
            init_model_ckpt = tf.train.Checkpoint(self.model)
            init_model_ckpt.read(self.init_model_path).expect_partial()

        # train
        print("start train")
        print(train_dict["loss sum weight"])
        message_handle = MessageHandle(save_dir+"train.txt")
        for epoch in range(start_epoch, epochs):
            # Reset the metrics at the start of the next epoch
            for loss_key, train_loss in train_dict["loss reg"].items():
                train_loss.reset_states()
            for acc_key, train_acc in train_dict["acc"].items():
                train_acc.reset_states()

            start_time = time.time()
            #warm up adjust
            self.warm_up_adjust(epoch, start_epoch, train_dict, 
                FUKL=FUKL, orthogonol_regu=orthogonol_regu,
                **extra_setting_dict)
            
            # train
            for train_step, (train_imgs, train_labels) in enumerate(train_iter):
                if data_aug:
                    train_imgs = self.data_aug_model(train_imgs)
                self.train_one_step(train_labels, train_imgs, train_dict, **extra_setting_dict)
                
                message_handle.refresh()
                message_handle.add(
                    f'Epoch:{epoch} '
                    f'{(train_step+1)/train_steps:.1%} '
                    f'{(int)(time.time()-start_time)}s'
                    )
                for loss_key, train_loss in train_dict["loss reg"].items():
                    message_handle.add(f'{loss_key}: {train_loss.result():.4f}')
                for acc_key, train_acc in train_dict["acc"].items():
                    message_handle.add(f'{acc_key}: {train_acc.result():.4f}')
            message_handle.save()

            # Reset the metrics at the start of the test
            for loss_key, test_loss in val_dict["loss reg"].items():
                test_loss.reset_states()
            for acc_key, test_acc in val_dict["acc"].items():
                test_acc.reset_states()
            # validate
            for val_step, (val_imgs, val_labels) in enumerate(val_iter):
                self.test_one_step(val_labels, val_imgs, val_dict, **extra_setting_dict)
                message_handle.add(f'{(val_step+1)/val_steps:.1%}')
                message_handle.refresh()
            for loss_key, test_loss in val_dict["loss reg"].items():
                message_handle.add(f'{loss_key}: {test_loss.result():.4f}')
            for acc_key, test_acc in val_dict["acc"].items():
                message_handle.add(f'{acc_key}: {test_acc.result():.4f}')
            message_handle.save()
            
            # save weight and test
            if epoch%10==9:
                # check point
                checkpoint.write(ckpt_dir+str(epoch))
    def load_weight(self, weight_dir, weight_epoch):
        print("load weight from", weight_dir, weight_epoch)
        #load weight
        checkpoint = tf.train.Checkpoint(model=self.model)
        checkpoint.read(weight_dir+"ckpt/"+str(weight_epoch)).expect_partial()

    def dataset_result_customized(self, batch_size, test_data_handle, ComputeHandle, **extra_setting_dict):
        # use ComputeHandle to decide what result to compute
        # ComputeHandle should inherit from ModelComputeTemplate
        # return a dict. in the dict, a list fo each key

        # do not load weight here
        #set data
        test_iter = test_data_handle.iterator(batch_size)
        data_len = test_data_handle.get_data_len()
        test_steps = data_len//batch_size

        #start computing
        compute_handle = ComputeHandle(self.model, self._get_compute_dict(), **extra_setting_dict)
        result_dict = None
        for test_step, (test_imgs, test_labels) in enumerate(test_iter):
        # for test_step in range(test_steps):
        #     test_imgs, test_labels = next(test_gen)
            print(f'{(test_step+1)/test_steps:.1%}    ', end="\r")
            batch_result_dict = compute_handle.result(test_imgs, test_labels)
            if isinstance(result_dict, type(None)):
                result_dict = {key: [] for key in batch_result_dict.keys()}
            for result_key, results in batch_result_dict.items():
                result_dict[result_key].extend(results)
        return result_dict
    
    def statistics(self, test_data_handle, save_dir):
        #get results for statistics of this model
        result_dict = self.dataset_result_customized(1000, test_data_handle, ResultCompute)

        correct_index_list = [index for index, correct in enumerate(result_dict["correct"]) if correct==1]
        correct_result_dict = {data_key:[data_list[index] for index in correct_index_list] \
            for data_key, data_list in result_dict.items()}

        incorrect_index_list = [index for index, correct in enumerate(result_dict["correct"]) if correct==0]
        incorrect_result_dict = {data_key:[data_list[index] for index in incorrect_index_list] \
            for data_key, data_list in result_dict.items()}

        #get statistics
        data_result_dict = {
            "true_inc":incorrect_result_dict["raw_true"],
            "false_inc":np.reshape(incorrect_result_dict["raw_false"], (-1,)),
            "true_cor":correct_result_dict["raw_true"],
            "false_cor":np.reshape(correct_result_dict["raw_false"], (-1,)),
        }
        color_dict = {
            "true_inc":"red",
            "false_inc":"black",
            "true_cor":"green",
            "false_cor":"blue",
        }
        plot_info = {
            "bin_width":0.1, 
            # x_range:(-10, 10),
            "x_label":"raw output",
            "y_label":"numbers in ratio",
        }
        distribution_draw(PltQuantityPlot, data_result_dict, color_dict, save_dir+"raw distrbution.png", **plot_info)

    def test_data(self, batch_size, test_data_handle, **extra_setting_dict):
        val_iter = test_data_handle.iterator(batch_size)
        val_steps = test_data_handle.get_data_len()//batch_size
        val_dict = self.set_loss_and_optimizer(is_train=False, batch_size=batch_size, **extra_setting_dict)
        for val_step, (val_imgs, val_labels) in enumerate(val_iter):
            self.test_one_step(val_labels, val_imgs, val_dict, **extra_setting_dict)
            print(
                f'{(val_step+1)/val_steps:.1%}', end="\r"
            )
        for loss_key, test_loss in val_dict["loss reg"].items():
            print(f'{loss_key}: {test_loss.result():.4f},', end=" ")
        for acc_key, test_acc in val_dict["acc"].items():
            print(f'{acc_key}: {test_acc.result():.4f},', end=" ")
        print()
    
    def _get_compute_dict(self):
        #structure of compute_dict
        #dict in dict for "loss", "loss data weight", "loss reg", "acc", "acc data weight"
        #one callback in dict for "optimizer"
        #key = ["loss", "loss data weight", "optimizer", "loss reg", "acc", "acc data weight"]
        compute_dict = {
            "loss":{},
            "loss reg":{},
            "loss data weight":{},#sample weight
            "loss sum weight":{},#default is 1
            "loss output index":{},#default is 0
            "acc":{},
            "acc data weight":{},
            "acc output index":{},#default is 0
        }
        return compute_dict
@tf.function
def input_gradient(labels, inputs, model, compute_dict, training=False):
    inputs = tf.cast(inputs, tf.float32)
    # training=False to avoid changing batch normalization
    with tf.GradientTape() as tape:
        tape.watch([inputs])
        pred_list = model(inputs, training=training)
        loss_dict = compute_loss(labels, pred_list, compute_dict)
        total_loss = loss_sum(loss_dict, compute_dict)
    #get new gradient
    gradients = tape.gradient(total_loss, inputs, unconnected_gradients='zero')
    # apply metric
    compute_loss_reg(loss_dict, compute_dict)
    compute_acc(labels, pred_list, compute_dict)
    return gradients, pred_list

@tf.function
def train_compute(labels, inputs, model, compute_dict,
    training=True):
    with tf.GradientTape() as tape:
        pred_list = model(inputs, training=training)
        loss_dict = compute_loss(labels, pred_list, compute_dict)
        total_loss = loss_sum(loss_dict, compute_dict)
    if "optimizer" in compute_dict:
        gradients = tape.gradient(total_loss, model.trainable_variables, unconnected_gradients='zero')
        safe_grad = []
        debug_test = []
        for grad in gradients:
            debug_test.append(tf.math.is_nan(grad))
            temp_grad = tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)#ignore nan gradient
            temp_grad = tf.where(tf.math.is_inf(temp_grad), tf.zeros_like(temp_grad), temp_grad)#ignore inf gradient, the reason of NAN
            safe_grad.append(temp_grad)
        compute_dict["optimizer"].apply_gradients(zip(safe_grad, model.trainable_variables))
    else:
        print("no optimizer in train compute.")
        sys.exit()
    # apply metric
    compute_loss_reg(loss_dict, compute_dict)
    compute_acc(labels, pred_list, compute_dict)
    return debug_test, loss_dict

@tf.function
def test_compute(labels, inputs, model, compute_dict, training=False):
    pred_list = model(inputs, training=training)
    loss_dict = compute_loss(labels, pred_list, compute_dict)
    # apply metric
    compute_loss_reg(loss_dict, compute_dict)
    compute_acc(labels, pred_list, compute_dict)
    return pred_list
@tf.function 
def tf_reg_compute(value_dict, reg_dict):
    for value_key, value in value_dict.items():
        reg_dict[value_key](value)

def compute_loss(labels, pred_list, compute_dict, sample_weight=None):
    loss_dict = {}
    for loss_key, loss_object in compute_dict["loss"].items():
        loss_data_weight = sample_weight
        if "loss data weight" in compute_dict:
            if loss_key in compute_dict["loss data weight"]:#overwrite "loss data weight"
                loss_data_weight = compute_dict["loss data weight"][loss_key]
        if loss_key in compute_dict["loss output index"]:#choose output index, default 0
            loss_output_index = compute_dict["loss output index"][loss_key]
        else:
            loss_output_index = 0
        loss = loss_object(labels, pred_list[loss_output_index], sample_weight=loss_data_weight)
        loss_dict[loss_key] = loss
    return loss_dict
def loss_sum(loss_dict, compute_dict):
    #sum by "loss sum weight"
    out_loss = 0
    for loss_key, loss in loss_dict.items():
        if loss_key in compute_dict["loss sum weight"]:
            sum_weight = compute_dict["loss sum weight"][loss_key]
        else:
            sum_weight = 1
        out_loss += sum_weight*loss
    return out_loss
def compute_loss_reg(loss_dict, compute_dict):
    for loss_key, loss in loss_dict.items():
        if loss_key not in compute_dict["loss reg"]:
            continue
        loss_reg = compute_dict["loss reg"][loss_key]
        loss_reg(loss)#loss is scalar, no sample weight

def compute_acc(labels, pred_list, compute_dict):
    for acc_key, acc_metric in compute_dict["acc"].items():
        if acc_key in compute_dict["acc data weight"]:
            acc_data_weight = compute_dict["acc data weight"][acc_key]
        else:
            acc_data_weight = None
        if acc_key in compute_dict["acc output index"]:#choose output
            acc_output_index = compute_dict["acc output index"][acc_key]
        else:
            acc_output_index = 0
        acc_metric(labels, pred_list[acc_output_index], sample_weight=acc_data_weight)

def distribution_draw(draw_func, data_result_dict, color_dict, save_path, normalized=True, cumsum=False,
    **plot_info):
    if not os.path.isdir(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    #do not load weight here
    #set default parameter
    plot = draw_func(**plot_info)
    for data_key, data_result in data_result_dict.items():
        color = color_dict[data_key]
        plot.add_data(data_result, color=color, normalized=normalized, cumsum=cumsum, label=data_key)
    plot.save(save_path, add_legend=True)
    plt.close('all')

def classifier_similarity(classifier_w):
    #classifier_w shape=(feature_len, class_len)
    #norm
    weight = tf_normalized_weight(classifier_w)
    inner_product = tf.matmul(tf.transpose(weight, perm=[1, 0]), weight)#shape=(10,10)
    inner_product = np.around(inner_product,3)
    return inner_product

if __name__ == "__main__":
    #set up data
    dataset_name = "cifar10"
    train_data_handle = TFDataGenerator(dataset_name, "train", shuffle=True)
    test_data_handle = TFDataGenerator(dataset_name, "test")

    train_info = train_data_handle.info
    input_shape = train_info.features["image"].shape
    num_classes = train_info.features["label"].num_classes

    #set up model
    train_handle = TrainHandle(class_count=num_classes, input_shape=input_shape)
    info = {
        "save_dir":"./train/"+train_handle.model_name+" "+dataset_name+"/"
    }

    #train and statistics
    train_handle.train(epochs=60, lr=1e-1, batch_size=1024, data_aug=True,
        train_data_handle=train_data_handle, test_data_handle=test_data_handle, start_epoch=0,
            **info)

    # train_handle.load_weight(info["save_dir"], 59)
    # train_handle.test_data( 1000, test_data_handle)

    train_handle.statistics(test_data_handle, info["save_dir"]+"statistics/")
    each_class_statistics(train_handle, test_data_handle, info["save_dir"]+"statistics/")

    # print(classifier_similarity(train_handle.model.out_dense.weights[0]))
    # print(tf.norm(train_handle.model.out_dense.weights[0], axis=0))





