from proto_train import *

class AblationTrainHandle(TrainHandle):
    def set_loss_and_optimizer(self, is_train, **setting_dict):#overwrite this to get different training
        batch_size = setting_dict["batch_size"]

        # loss and metric
        train_dict = self._get_compute_dict()
        train_dict["loss"]["MHKL"] = MultiHotKL(axis=0)
        train_dict["loss output index"]["MHKL"] = 2
        train_dict["loss sum weight"]["MHKL"] = 1
        train_dict["loss reg"]["MHKL"] = tf.keras.metrics.Mean()

        train_dict["loss"]["FUKL"] = FalseUniformKL(axis=[0,1])#false outputs to uniform dsitribution, through batch
        train_dict["loss output index"]["FUKL"] = 2
        train_dict["loss sum weight"]["FUKL"] = 0.2
        train_dict["loss reg"]["FUKL"] = tf.keras.metrics.Mean()

        train_dict["acc"]["acc"] = tf.keras.metrics.SparseCategoricalAccuracy()

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
            train_dict["loss sum weight"]["MHKL"] = 1
            train_dict["loss sum weight"]["FUKL"] = 1
            print("loss sum weight:", train_dict["loss sum weight"])

    def train_one_step(self, train_labels, train_imgs, train_dict, **setting_dict):
        grad_test_list, loss_dict = train_compute(train_labels, train_imgs, self.model, train_dict, training=True)
        for grad_test in grad_test_list:#debug
            if np.any(grad_test):
                print()
                print(loss_dict)
                break

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
        "save_dir":"./train/"+train_handle.model_name+" "+dataset_name+" ablation no atk/"
    }

    #set up data
    train_data_handle = TFDataGenerator("cifar10", "train", shuffle=True)
    test_data_handle = TFDataGenerator("cifar10", "test")
    
    train_handle.train(epochs=60, lr=1e-1, batch_size=1024, data_aug=True,
        train_data_handle=train_data_handle, test_data_handle=test_data_handle, start_epoch=0,
            **info)

    # train_handle.load_weight(info["save_dir"], 59)
    # train_handle.test_data( 1000, test_data_handle)

    # print("classifier_similarity:\n", classifier_similarity(train_handle.model.out_dense.weights[0]))
    train_handle.statistics(test_data_handle, info["save_dir"]+"statistics/")
    each_class_statistics(train_handle, test_data_handle, info["save_dir"]+"statistics/")

