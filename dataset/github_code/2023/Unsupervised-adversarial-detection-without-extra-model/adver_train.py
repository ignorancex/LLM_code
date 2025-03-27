from proto_train import *

class AdversarialTrainHandle(TrainHandle):
    def set_loss_and_optimizer(self, is_train, **setting_dict):#overwrite this to get different training
        batch_size = setting_dict["batch_size"]
        origin_data_weight = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])
        atk_data_weight = np.concatenate([np.zeros(batch_size), np.ones(batch_size)])
        atk_dict = self._get_compute_dict()
        atk_dict["loss"] = {"grad":setting_dict["grad_func"]}
        atk_dict["loss output index"] = {"grad":setting_dict["grad_func_output"]}
        atk_dict["update_rate"] = setting_dict["update_rate"] #how far a step can go
        atk_dict["update_steps"] = setting_dict["update_steps"] #how many steps i will walk
        atk_dict["grad_adjust_func"] = setting_dict["grad_adjust_func"] #how to walk
        atk_dict["epsilon"] = setting_dict["epsilon"] #constrain the difference

        # loss and metric
        train_dict = self._get_compute_dict()
        train_dict["loss"]["loss"] = tf.keras.losses.SparseCategoricalCrossentropy()
        train_dict["loss reg"]["loss"] = tf.keras.metrics.Mean(name='train_loss')

        train_dict["acc"]["atk acc"] = tf.keras.metrics.SparseCategoricalAccuracy()

        train_dict["acc"]["atk target"] = MeanWrapper(ShowTarget())#show origin target
        train_dict["acc output index"]["atk target"] = 2

        train_dict["atk"] = atk_dict

        val_dict = self._get_compute_dict()
        val_dict["loss"]["loss"] = tf.keras.losses.SparseCategoricalCrossentropy()
        val_dict["loss data weight"]["loss"] = atk_data_weight#just for atk data
        val_dict["loss reg"]["loss"] = tf.keras.metrics.Mean(name='train_loss')

        val_dict["acc"]["origin acc"] = tf.keras.metrics.SparseCategoricalAccuracy()
        val_dict["acc data weight"]["origin acc"] = origin_data_weight

        val_dict["acc"]["atk acc"] = tf.keras.metrics.SparseCategoricalAccuracy()
        val_dict["acc data weight"]["atk acc"] = atk_data_weight

        val_dict["acc"]["origin target"] = MeanWrapper(ShowTarget())#show origin target
        val_dict["acc data weight"]["origin target"] = origin_data_weight
        val_dict["acc output index"]["origin target"] = 2

        val_dict["acc"]["atk target"] = MeanWrapper(ShowTarget())#show origin target
        val_dict["acc data weight"]["atk target"] = atk_data_weight
        val_dict["acc output index"]["atk target"] = 2

        val_dict["atk"] = atk_dict

        if is_train:
            lr = setting_dict["lr"]
            train_steps = setting_dict["train_steps"]
            end_epoch = setting_dict["end_epoch"]

            #set optimizer
            # lr_schedule = MyLRSchedule(lr, steps_per_epoch=train_steps, end_epoch=end_epoch)
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(lr, end_epoch*train_steps)
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)#, nesterov=True)

            train_dict["optimizer"] = optimizer
            return train_dict, val_dict
        else:
            return val_dict

    def train_one_step(self, train_labels, train_imgs, train_dict, **setting_dict):#overwrite this to get different training
        atk_imgs = generate_atk_data(train_labels, train_imgs, self.model, train_dict["atk"], training=False)
        train_compute(train_labels, atk_imgs, self.model, train_dict, training=True)

    def test_one_step(self, val_labels, val_imgs, val_dict, **setting_dict):#overwrite this to test differently
        atk_imgs = generate_atk_data(val_labels, val_imgs, self.model, val_dict["atk"], training=False)
        atk_imgs = atk_imgs.numpy()#atk_imgs is tensorflow tensor
        val_imgs = val_imgs.numpy()
        val_imgs.astype(atk_imgs.dtype)
        concat_val_imgs = np.concatenate([val_imgs, atk_imgs], axis=0)
        concat_val_labels = np.concatenate([val_labels,val_labels], axis=0)
        test_compute(concat_val_labels, concat_val_imgs, self.model, val_dict)

def max_norm_grad(grad, grad_sum, update_rate):#grad_adjust_func
    max_grad = tf.reduce_max(tf.abs(grad), axis=(1,2,3), keepdims=True)#get the max value in abs
    max_grad = tf.where(max_grad<=0, tf.ones_like(max_grad), max_grad)#almost impossible
    update_step = grad/max_grad*update_rate
    grad_sum += update_step
    return grad_sum
def FGSM(grad, grad_sum, update_rate):#grad_adjust_func
    update_step = tf.math.sign(grad)*update_rate
    grad_sum += update_step
    return grad_sum

@tf.function
def generate_atk_data(labels, imgs, model, atk_dict,
    training=False, include_process=False):#not loading weight
    #get parameter from atk_dict
    update_steps = atk_dict["update_steps"]
    update_rate = atk_dict["update_rate"]
    grad_adjust_func = atk_dict["grad_adjust_func"]
    epsilon = atk_dict["epsilon"]

    #start
    origin_imgs = tf.cast(imgs, tf.float32)
    grad_sum = tf.zeros_like(origin_imgs)
    test_imgs = origin_imgs

    if include_process:
        test_imgs_list = []
    for i in range(update_steps):
        grad_tensor, last_pred_list = input_gradient(labels, test_imgs, model, atk_dict, training=training)
        
        # get updated grad_sum
        grad_sum = grad_adjust_func(grad_tensor, grad_sum, update_rate)

        grad_sum = tf.clip_by_value(grad_sum, -epsilon, epsilon)#set constraint
        test_imgs = tf.clip_by_value(origin_imgs-grad_sum, 0, 255)
        if include_process:
            test_imgs_list.append(test_imgs)
    if include_process:
        return test_imgs_list
    return test_imgs


if __name__ == "__main__":
    id_num = 10
    train_data_handle = TFDataGenerator("cifar10", "train", shuffle=True)
    test_data_handle = TFDataGenerator("cifar10", "test")

    train_handle = AdversarialTrainHandle(class_count=id_num)
    info = {
        "save_dir":"./train/"+train_handle.model_name+" cifar"+str(id_num)+" PGD batch 128/",
        "grad_func":ReverseLoss(tf.keras.losses.SparseCategoricalCrossentropy(tf.keras.losses.Reduction.SUM)),
        "grad_func_output":0,#softmax output
        "grad_adjust_func":FGSM,
        "update_rate":2,
        "update_steps":5,
        "epsilon":8,
    }

    train_handle.train(epochs=110, lr=1e-1, batch_size=128, data_aug=True,
        train_data_handle=train_data_handle, test_data_handle=test_data_handle, start_epoch=0,
            **info)

    # train_handle.load_weight(info["save_dir"], 109)
    # train_handle.test_data(1000, test_data_handle, **info)

    train_handle.statistics(test_data_handle, info["save_dir"]+"statistics/")
    each_class_statistics(train_handle, test_data_handle, save_dir+"statistics/")