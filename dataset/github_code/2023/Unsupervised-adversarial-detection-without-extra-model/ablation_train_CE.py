from adver_train import *

class AblationTrainHandle(TrainHandle):
    def set_loss_and_optimizer(self, is_train, **setting_dict):#overwrite this to get different training
        batch_size = setting_dict["batch_size"]
        atk_size = (int)(batch_size*setting_dict["atk_ratio"])
        raw_size = batch_size-atk_size
        origin_data_weight = np.concatenate([np.ones(raw_size), np.zeros(atk_size)])
        atk_data_weight = np.concatenate([np.zeros(raw_size), np.ones(atk_size)])

        # loss and metric
        train_dict = self._get_compute_dict()
        train_dict["loss"]["CE"] = tf.keras.losses.SparseCategoricalCrossentropy()
        train_dict["loss data weight"]["CE"] = origin_data_weight#not pull atk data
        train_dict["loss reg"]["CE"] = tf.keras.metrics.Mean()

        train_dict["loss"]["FUKL"] = FalseUniformKL(axis=[0,1])#false outputs to uniform dsitribution, through batch
        train_dict["loss output index"]["FUKL"] = 2
        train_dict["loss sum weight"]["FUKL"] = 0.2
        train_dict["loss reg"]["FUKL"] = tf.keras.metrics.Mean()

        train_dict["acc"]["acc"] = tf.keras.metrics.SparseCategoricalAccuracy()
        train_dict["acc data weight"]["acc"] = origin_data_weight

        train_dict["acc"]["origin target"] = MeanWrapper(ShowTarget())#show origin target
        train_dict["acc data weight"]["origin target"] = origin_data_weight
        train_dict["acc output index"]["origin target"] = 2

        train_dict["acc"]["origin max false"] = MeanWrapper(ShowMaxWrong())#show origin target
        train_dict["acc data weight"]["origin max false"] = origin_data_weight
        train_dict["acc output index"]["origin max false"] = 2

        train_dict["acc"]["atk target"] = MeanWrapper(ShowTarget())#show origin target
        train_dict["acc data weight"]["atk target"] = atk_data_weight
        train_dict["acc output index"]["atk target"] = 2

        train_dict["acc"]["atk max false"] = MeanWrapper(ShowMaxWrong())#show origin target
        train_dict["acc data weight"]["atk max false"] = atk_data_weight
        train_dict["acc output index"]["atk max false"] = 2

        atk_dict = self._get_compute_dict()
        atk_dict["loss"]["grad"] = setting_dict["grad_func"]
        atk_dict["loss output index"]["grad"] = setting_dict["grad_func_output"]
        atk_dict["update_rate"] = setting_dict["update_rate"] #how far a step can go
        atk_dict["update_steps"] = setting_dict["update_steps"] #how many steps i will walk
        atk_dict["grad_adjust_func"] = setting_dict["grad_adjust_func"] #how to walk
        atk_dict["epsilon"] = setting_dict["epsilon"] #constrain the difference

        train_dict["atk"] = atk_dict

        val_dict = dict(train_dict)#copy and share function
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
        if epoch==5 or (start_epoch==epoch and start_epoch>5):
            train_dict["loss sum weight"]["FUKL"] = 1
            print("loss sum weight:", train_dict["loss sum weight"])

    def train_one_step(self, train_labels, train_imgs, train_dict, **setting_dict):#overwrite this to get different training
        #check which samples should be attacked
        batch_size = train_labels.shape[0]
        atk_size = (int)(batch_size*setting_dict["atk_ratio"])
        raw_size = batch_size-atk_size

        #training=False, otherwise change batch_norm too much
        if atk_size>0:
            atk_imgs = generate_atk_data(train_labels[raw_size:], train_imgs[raw_size:], self.model, train_dict["atk"], training=False)
            atk_imgs = atk_imgs.numpy()#atk_imgs is tensorflow tensor
            train_imgs = train_imgs.numpy()
            train_imgs.astype(atk_imgs.dtype)
            train_imgs[raw_size:] = atk_imgs

        grad_test_list, loss_dict = train_compute(train_labels, train_imgs, self.model, train_dict, training=True)
        for grad_test in grad_test_list:#debug
            if np.any(grad_test):
                print()
                print(loss_dict)
                break

    def test_one_step(self, val_labels, val_imgs, val_dict, **setting_dict):#overwrite this to test differently
        #check which samples should be attacked
        batch_size = val_labels.shape[0]
        atk_size = (int)(batch_size*setting_dict["atk_ratio"])
        raw_size = batch_size-atk_size

        if atk_size>0:
            atk_imgs = generate_atk_data(val_labels[raw_size:], val_imgs[raw_size:], self.model, val_dict["atk"], training=False)
            atk_imgs = atk_imgs.numpy()#atk_imgs is tensorflow tensor
            val_imgs = val_imgs.numpy()
            val_imgs.astype(atk_imgs.dtype)
            val_imgs[raw_size:] = atk_imgs
        test_compute(val_labels, val_imgs, self.model, val_dict)


if __name__ == "__main__":
    #set up data
    dataset_name = "cifar10"
    train_data_handle = TFDataGenerator(dataset_name, "train", shuffle=True)
    test_data_handle = TFDataGenerator(dataset_name, "test")

    train_info = train_data_handle.info
    input_shape = train_info.features["image"].shape
    num_classes = train_info.features["label"].num_classes

    #set up model
    train_handle = AblationTrainHandle(class_count=num_classes, input_shape=input_shape)
    info = {
        "save_dir":"./train/"+train_handle.model_name+" "+dataset_name+" ablation with CE/",
        "atk_ratio":0.1,
        "grad_func":ReverseLoss(TargetDirLoss()),
        "grad_func_output":2,#raw output
        "grad_adjust_func":FGSM,
        "update_rate":2,
        "update_steps":5,
        "epsilon":8,
    }

    #set up data
    train_data_handle = TFDataGenerator("cifar10", "train", shuffle=True)
    test_data_handle = TFDataGenerator("cifar10", "test")
    save_dir = info["save_dir"]
    
    train_handle.train(epochs=60, lr=1e-1, batch_size=1024, data_aug=True,
        train_data_handle=train_data_handle, test_data_handle=test_data_handle, start_epoch=0,
            **info)

    # train_handle.load_weight(info["save_dir"], 59)
    # train_handle.test_data( 1000, test_data_handle)

    # print("classifier_similarity:\n", classifier_similarity(train_handle.model.out_dense.weights[0]))
    train_handle.statistics(test_data_handle, info["save_dir"]+"statistics/")
    each_class_statistics(train_handle, test_data_handle, info["save_dir"]+"statistics/")

