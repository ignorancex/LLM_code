from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, AutoAttack, ProjectedGradientDescent, DeepFool
from art.estimators.classification import TensorFlowV2Classifier
from model import *
from data import *
import numpy as np
import os
import tensorflow_datasets as tfds
import sys
def np_save(data_path, save_data):
    data_dir = os.path.dirname(data_path)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    np.save(data_path, save_data)
    print("data saved", data_path)

def TF_dataset_reader(dataset_name, data_name):
    dataset, info = tfds.load(dataset_name, split=data_name, as_supervised=True, batch_size=-1, with_info=True)
    input_shape = info.features["image"].shape
    num_classes = info.features["label"].num_classes
    x_data, y_data = tfds.as_numpy(dataset)
    return x_data, y_data, input_shape, num_classes

if __name__ == "__main__":
    # global info
    dataset_name = "cifar10"
    weight_name = "MyResnet cifar10 proposed"
    config_dir = "./art_data_config/" 
    config_name = "main" # the config of main and ablation are different.
    weight_dir = "./train/"+weight_name+"/"
    weight_epoch = 59
    weight_path = weight_dir+"ckpt/"+str(weight_epoch)
    
    #prepare data
    x_test, y_test, input_shape, num_classes = TF_dataset_reader(dataset_name, "test")
    raw_save_dir = "./test/"+dataset_name+"/raw data/"
    np_save(raw_save_dir+"raw_x.npy", x_test)
    np_save(raw_save_dir+"labels.npy", y_test)
    x_test = x_test.astype(np.float32)

    #prepare model
    raw_model = MyResnet(num_classes)
    checkpoint = tf.train.Checkpoint(model=raw_model)
    checkpoint.read(weight_path)
    temp_input = Input(input_shape)
    tf_model_raw = Model(inputs=temp_input, outputs=raw_model(temp_input)[2])#choose raw output

    classifier = TensorFlowV2Classifier(
        model=tf_model_raw,
        loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        nb_classes=num_classes,
        input_shape=input_shape,
        clip_values=(0, 255),
    )
    # load config
    config = load_json(config_dir+config_name+".json")
    atk_str_list = config["attack_strength"]
    atk_type_dict = config["attack_type"]

    #start computing
    for atk_str in atk_str_list:
        #the strength of attack
        epsilon = atk_str["epsilon"]
        update_rate = atk_str["update_rate"]
        update_steps = atk_str["update_steps"]

        #info
        save_dir = "./test/"+dataset_name+"/"+weight_name+"/"+config_name+"/"
        strength_label = "_"+str(epsilon)+"_"+str(update_rate)+"_"+str(update_steps)
        
        #get data for AutoAttack inf
        if atk_type_dict["AUTO"]:
            attack = AutoAttack(estimator=classifier, norm=np.inf, eps=epsilon, eps_step=update_rate, batch_size=1000)
            adv_data = attack.generate(x=x_test)
            np_save(save_dir+"auto_inf"+strength_label+".npy", adv_data)

        #FGSM
        if atk_type_dict["FGSM"]:
            attack = FastGradientMethod(estimator=classifier, eps=epsilon, eps_step=update_rate, batch_size=1000)
            adv_data = attack.generate(x=x_test)
            print(np.max(np.absolute(x_test-adv_data)))
            np_save(save_dir+"fgsm"+strength_label+".npy", adv_data)
        
        #BIM
        if atk_type_dict["BIM"]:
            attack = BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=update_rate, batch_size=1000, max_iter=int(epsilon*1.25))
            adv_data = attack.generate(x=x_test)
            np_save(save_dir+"bim"+strength_label+".npy", adv_data)

        #get data for PGD inf
        if atk_type_dict["PGD"]:
            attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=epsilon, eps_step=update_rate, 
                batch_size=1000, max_iter=update_steps)
            adv_data = attack.generate(x=x_test)
            np_save(save_dir+"pgd_inf"+strength_label+".npy", adv_data)

        #deep fool
        if atk_type_dict["DEEPFOOL_inf"]:
            attack = DeepFool(classifier=classifier, batch_size=1000, max_iter=update_steps)
            adv_data = attack.generate(x=x_test)
            diff = adv_data-x_test
            diff = np.clip(diff, -epsilon, epsilon)
            adv_data = x_test+diff
            np_save(save_dir+"deepfool_inf"+strength_label+".npy", adv_data)

        if atk_type_dict["DEEPFOOL_unlimited"]:
            attack = DeepFool(classifier=classifier, batch_size=1000, max_iter=update_steps)
            adv_data = attack.generate(x=x_test)
            np_save(save_dir+"deepfool_unlimited"+strength_label+".npy", adv_data)



