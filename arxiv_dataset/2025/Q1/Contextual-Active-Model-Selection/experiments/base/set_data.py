"""Preprocess the model predictions"""
from src.evaluation.aux.compute_precision_measures import *
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import config as config

#acknowledgement: this file is partially referenced from our baseline model picker. We extended it to be suitable for online contextual data streaming settings.

class MiniSet(Dataset):
  def __init__(self,fileroots,labels,predictions,transform):
    self.fileroots=fileroots
    self.labels=labels
    self.predictions = predictions
    self.transform=transform

  def __len__(self):
    return len(self.fileroots)

  def __getitem__(self,idx):
    img=Image.open(self.fileroots[idx])
    img=self.transform(img)
    return img,self.labels[idx],self.fileroots[idx],self.predictions[idx]


class SetData():
    def __init__(self, data_set_name, pool_size, pool_setting, budgets, num_reals, resultsdir,
                  which_methods, policy_index, random_policy):
        """
        Base class to set data for the experiments.

        Parameters:
        :param dataset: Options include {'CIFAR10', 'DRIFT', 'VERTEBRAL', 'HIV'}
        :param pool_size: Size of streaming instances in a realization
        :param pool_setting: It must be 'floating' by default. Options include {'floating', 'fixed'}
        :param budgets: A list of budgets/query cost which the model selection methods will be evaluated
        :param num_reals: Number of realizations over which the evaluations are averaged. Set it to a few thousands at least
        :param resultsdir: the directory to save the results.
        :param which_methods: Methods under testing
        :param policy_index: index of the policy running
        :param random_policy: if we using only random policies. It must be set to "false" by default,
        """

        if data_set_name == "HIV_contextual":
            # setting for HIV experiment
            # Data path
            cwd = os.getcwd()

            print("current working directory:",cwd)
            path_HIV=cwd+"/resources/contextual_data/HIV/"

            # Preprocess
            oracle = np.loadtxt(str(path_HIV) + "/oracle.out")
            predictions = np.loadtxt(str(path_HIV) + "/predictions.out")

            self._predictions = predictions
            self._oracle = oracle
            self._data_set_name = data_set_name

            print("num classes:",len(np.unique(self._oracle)))
            self._num_classes = len(np.unique(self._oracle))
            self._num_models = np.size(self._predictions, 1)
            print("num classifiers: ",self._num_models)

            identity_m=[]
            non_identity=[]

            if not random_policy:

                # CAMS
                #load advice matrix
                random_matrix=np.load(str(path_HIV)+"/advice_matrix_random_policy.npz")["advice_matrix"]
                self._advice_matrix_identity=np.load(str(path_HIV)+"/advice_matrix_identity.npz")["advice_matrix"]

                # creating the policies and classifer advice matrix
                for idx in range(len(self._advice_matrix_identity)):
                    
                    temp=[]
                    temp_identity=[]

            
                    for item in self._advice_matrix_identity[idx][[0,1,2,3,4,5,6,2,7,3,7],:]:
                        temp.append(item)

                    for item in self._advice_matrix_identity[idx][[0,1,2,3,4,5,6,2,7,3,7,7,8,9,10],:]:
                        temp_identity.append(item)
                    
                    if config.task =="task5":     
                        for item in random_matrix[idx][[0,0,0,1,1,2,2,3,3],:]:
                            temp_identity.append(item)
                            temp.append(item)

                    elif config.task == "task7":

                        temp_identity=[]
                        for item in self._advice_matrix_identity[idx][7:]:
                            temp_identity.append(item)
                    else:

                        for item in random_matrix[idx][[0,0,0,1,1,2,2,3,3],:]:
                            temp_identity.append(item)
                            temp.append(item)

                    non_identity.append(temp)
                    identity_m.append(temp_identity)

            else:
                print("use random policy matrix only")
                self._advice_matrix_identity=np.load(str(path_HIV)+"/advice_matrix_random_policy.npz")["advice_matrix"]

                for item in self._advice_matrix_identity:
                    identity_m.append(item)  

            self._advice_matrix=np.asarray(non_identity)
            self._num_policies=np.size(self._advice_matrix, 1)
            print("advice matrix(policies) size:",np.size(self._advice_matrix, 1))
            self._advice_matrix_identity=np.asarray(identity_m)
            self._num_policies_identity = np.size(self._advice_matrix_identity, 1)
            print("advice matrix(policies+classifiers) size:",np.size(self._advice_matrix_identity, 1))

            best_policy=[]
            for item in self._advice_matrix_identity:
                best_policy.append(item[policy_index,:])

            self._best_policy=np.asarray(best_policy)
            self._num_best_policy= np.size(self._best_policy, 1)
            self._size_entire_pool = np.size(self._predictions, 0)


        elif data_set_name == 'VERTEBRAL_contextual':
            # setting for VERTEBRAL experiment

            # Data path
            cwd = os.getcwd()

            print("current working directory:",cwd)
            path_VERTEBRAL=cwd+"/resources/contextual_data/VERTEBRAL/"

            # Preprocess
            oracle = np.loadtxt(str(path_VERTEBRAL) + "/oracle.out")
            predictions = np.loadtxt(str(path_VERTEBRAL) + "/predictions.out")
 
            self._predictions = predictions
            self._oracle = oracle
            self._data_set_name = data_set_name

            print("number of classes:",len(np.unique(self._oracle)))
            self._num_classes = len(np.unique(self._oracle))
            self._num_models = np.size(self._predictions, 1)
            print("number of classifiers: ",self._num_models)

            identity_m=[]
            non_identity=[]


            if not random_policy:

                random_matrix=np.load(str(path_VERTEBRAL)+"/advice_matrix_random_policy.npz")["advice_matrix"]
                self._advice_matrix_identity=np.load(str(path_VERTEBRAL)+"/advice_matrix_identity.npz")["advice_matrix"]

                # print("random matrix shape:",np.shape(random_matrix))
                # print("advice matrix identity(policies and classifiers) shape:",np.shape(self._advice_matrix_identity))

                # creating the policies and classifer advice matrix
                for idx in range(len(self._advice_matrix_identity)):
                    
                    temp=[]
                    temp_identity=[]

                    for item in self._advice_matrix_identity[idx][[0,1,2,3,4,5],:]:
                        temp.append(item)

                    for item in self._advice_matrix_identity[idx][[0,1,2,3,4,5,6,7,8,9,10,11],:]:
                        temp_identity.append(item)

                    if config.task =="task2":
                        for item in random_matrix[idx][[0,1,2,3,4],:]:
                            temp_identity.append(item)
                            temp.append(item)                        
                    
                    elif config.task == "task7":
                        temp_identity=[]
                        for item in self._advice_matrix_identity[idx][6:]:
                            temp_identity.append(item)

                    else:
                        for item in random_matrix[idx][[0,1,2,3,4,0,1,2,3,4,3],:]:
                            temp_identity.append(item)
                            temp.append(item)

                    non_identity.append(temp)
                    identity_m.append(temp_identity)

 
            else:
                print("using random policies")
                self._advice_matrix_identity=np.load(str(path_VERTEBRAL)+"/advice_matrix_random_policy.npz")["advice_matrix"]

                for item in self._advice_matrix_identity:
                    identity_m.append(item)  


            self._advice_matrix=np.asarray(non_identity)
            self._num_policies=np.size(self._advice_matrix, 1)
            print("advice matrix(policies):",np.size(self._advice_matrix, 1))
            self._advice_matrix_identity=np.asarray(identity_m)
            self._num_policies_identity = np.size(self._advice_matrix_identity, 1)
            print("advice matrix(policies+classifiers):",np.size(self._advice_matrix_identity, 1))


            best_policy=[]
            for item in self._advice_matrix_identity:
                best_policy.append(item[policy_index,:])

            print("policy index:",policy_index)
            self._best_policy=np.asarray(best_policy)
            self._num_best_policy= np.size(self._best_policy, 1)
            self._size_entire_pool = np.size(self._predictions, 0)


        elif data_set_name == 'drift_contextual':
            # setting for DRIFT experiment

            # Data path
            cwd = os.getcwd()

            print("current working directory:",cwd)
            path_drift=cwd+"/resources/contextual_data/DRIFT/"

            # Preprocess
            oracle = np.loadtxt(str(path_drift) + "/oracle.out")
            predictions = np.loadtxt(str(path_drift) + "/predictions.out")

            self._predictions = predictions
            self._oracle = oracle
            self._data_set_name = data_set_name

            print("num classes:",len(np.unique(self._oracle)))
            self._num_classes = len(np.unique(self._oracle))
            self._num_models = np.size(self._predictions, 1)
            print("num classifiers: ",self._num_models)

            identity_m=[]
            non_identity=[]

            if not random_policy:

                random_matrix=np.load(str(path_drift)+"/advice_matrix_random_policy.npz")["advice_matrix"]
                self._advice_matrix_identity=np.load(str(path_drift)+"/advice_matrix_identity.npz")["advice_matrix"]

                # creating the policies and classifer advice matrix
                for idx in range(len(self._advice_matrix_identity)):
                    
                    temp=[]
                    temp_identity=[]

                    for item in self._advice_matrix_identity[idx][[0,1,2,3,4,5,6,7],:]:
                        temp.append(item)

                    for item in self._advice_matrix_identity[idx][[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],:]:
                        temp_identity.append(item)

                    if config.task =="task2":
                        for item in random_matrix[idx][[0,1,2,3,4],:]:
                            temp_identity.append(item)
                            temp.append(item)                        

                    elif config.task =="task1" or config.task =="task5" or config.task == "task6" or config.task == "task8":

                        for item in random_matrix[idx][[1,2,3],:]:
                            temp_identity.append(item)
                            temp.append(item)
                    
                    elif config.task == "task7":
                        temp_identity=[]
                        for item in self._advice_matrix_identity[idx][8:]:
                            temp_identity.append(item)

                    else:
                        print("wrong!!")
                        exit()

                    non_identity.append(temp)
                    identity_m.append(temp_identity)

 
            else:
                print("using random polices")

                self._advice_matrix_identity=np.load(str(path_drift)+"/advice_matrix_identity.npz")["advice_matrix"]
                self._advice_matrix_identity=np.load(str(path_drift)+"/advice_matrix_random_policy.npz")["advice_matrix"]

                for item in self._advice_matrix_identity:
                    identity_m.append(item)  


            self._advice_matrix=np.asarray(non_identity)
            self._num_policies=np.size(self._advice_matrix, 1)
            print("advice matrix(policies) size:",np.size(self._advice_matrix, 1))
            self._advice_matrix_identity=np.asarray(identity_m)
            self._num_policies_identity = np.size(self._advice_matrix_identity, 1)
            print("advice matrix(policies+classifers) size:",np.size(self._advice_matrix_identity, 1))


            best_policy=[]
            for item in self._advice_matrix_identity:
                best_policy.append(item[policy_index,:])

            print("policy index:",policy_index)

            self._best_policy=np.asarray(best_policy)
            self._num_best_policy= np.size(self._best_policy, 1)

            print("best policy:",np.size(self._best_policy, 1))
            self._size_entire_pool = np.size(self._predictions, 0)

        elif data_set_name == 'cifar_contextual':
            # Data path
            cwd = os.getcwd()

            print("current working directory:",cwd)

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            path_cifar10=cwd+"/resources/contextual_data/CIFAR10/"

            # Preprocess
            oracle = np.loadtxt(str(path_cifar10) + "/oracle.out")
            predictions = np.loadtxt(str(path_cifar10) + "/predictions.out")

            self._predictions = predictions
            self._oracle = oracle

            path = np.genfromtxt(str(path_cifar10) + "/path.out", dtype=str)

            path_arr=[]
            for item in path:
                tmp=item.replace("./",cwd+"/resources/contextual_data/CIFAR10/")
                path_arr.append(tmp)
            # Dataset specific attributes
            path_arr=np.asarray(path_arr)
            self._paths = np.asarray(path_arr)

            testloader = DataLoader(MiniSet(path_arr, oracle, predictions, transform_test), shuffle=False,
                                    batch_size=1)

            inputs_arr=[]
            for batch_idx, (inputs_, targets_, paths_, predictions_) in enumerate(testloader):
                inputs_arr.append(inputs_.numpy())

            self._inputs = np.asarray(inputs_arr)

            print("num classes:",len(np.unique(oracle)))
            self._num_classes = len(np.unique(oracle))
            self._num_models = np.size(predictions, 1)
            print("num classifiers: ",self._num_models)

            self._size_entire_pool = np.size(predictions, 0)
            self._path_cifar=path_cifar10

            non_identity=[]
            identity_m =[]

            if not random_policy:

                random_matrix=np.load(str(path_cifar10)+"/advice_matrix_random_policy.npz")["advice_matrix"]
                self._advice_matrix_identity=np.load(str(path_cifar10)+"/advice_matrix_identity.npz")["advice_matrix"]

                # creating the policies and classifer advice matrix
                for idx in range(len(self._advice_matrix_identity)):
                    temp_identity=[]
                    temp=[]

                    for item in self._advice_matrix_identity[idx][0:85]:
                        temp.append(item)

                    for item in self._advice_matrix_identity[idx]:
                        temp_identity.append(item)
                    
                    if config.task =="task2":
                        for item in random_matrix[idx][[0,1,2,3,4],:]:
                            temp_identity.append(item)
                            temp.append(item)                        
                    
                    elif config.task == "task1" or config.task == "task6" or config.task == "task8":
                        for item in random_matrix[idx][[0,1,1,2,3,4,1,2],:]:
                            temp_identity.append(item)
                            temp.append(item)
                    
                    elif config.task == "task7":
                        temp_identity=[]
                        for item in self._advice_matrix_identity[idx][85:]:
                            temp_identity.append(item)
                    
                    elif config.task =="task5":
                        temp_identity=[]
                        for item in self._advice_matrix_identity[idx][85:]:
                            temp_identity.append(item)
                        for item in random_matrix[idx][[0,0,0,1,1, 2,2,3,3,4, 4],:]:
                            temp_identity.append(item)

                    else:
                        print("error")
                        exit()

                    non_identity.append(temp)
                    identity_m.append(temp_identity)                

            else:
                print("using random policies")
                #random policy matrix
                self._advice_matrix_identity=np.load(str(path_cifar10)+"/advice_matrix_random_policy.npz")["advice_matrix"]

                for item in self._advice_matrix_identity:
                    identity_m.append(item)  

            self._advice_matrix=np.asarray(non_identity)
            self._num_policies=np.size(self._advice_matrix, 1)
            print("advice matrix (policies) size:",np.size(self._advice_matrix, 1))

            self._advice_matrix_identity=np.asarray(identity_m)
            self._num_policies_identity = np.size(self._advice_matrix_identity, 1)
            print("advice matrix (policies+classifiers) size:",np.size(self._advice_matrix_identity, 1))

            best_policy=[]
            for item in self._advice_matrix_identity:
                best_policy.append(item[policy_index,:])

            print("policy index:",policy_index)
            self._best_policy=np.asarray(best_policy)
            self._num_best_policy= np.size(self._best_policy, 1)
            self._size_entire_pool = np.size(self._predictions, 0)

        else:
            assert 'Dataset name has not been specified!'

        # Attribute other values to self
        self._parameter_testing_bound=[]
        self._budgets = budgets
        self._num_reals = num_reals
        self._num_instances = pool_size  # This parameter is more experiment dependent, is different (smaller) than the entire pool size
        self._resultsdir = resultsdir
        self._pool_setting = pool_setting
        self._data_set_name = data_set_name
        self._which_methods = which_methods

        # Attribute the set of methods and their full names
        all_methods = list(['mp', 'qbc', 'sqbc', 'rs', 'iwal', 'efal',
                            "CAMS_best_policy","CAMS_identity","CAMS_test","contextual_qbc","contextual_iwal"])

        all_methods_fullname = list(
            ['Model Picker', 'Query by Committee', 'Structural Query by Committee', 'Random Sampling',
             'Importance Weighted Active Learning', 'Efficient Active Learning',
             "CAMS single policy","CAMS identity","CAMS test","Contextual Query by Committee","Contextual Importance Weighted Active Learning"])

        methods = []
        methods_fullname = []
        for i in range(len(which_methods)):
            if which_methods[i] == 1:
                methods.append(all_methods[i])
                methods_fullname.append(all_methods_fullname[i])

        self._methods = methods
        self._methods_fullname = methods_fullname

    def append_testing_bound(self,data_):
        self._parameter_testing_bound.append(data_)

    def save_data(self):
        """
        This function saves the setting details.
        """
        # Extract variables
        num_instances = self._num_instances
        pool_setting = self._pool_setting
        num_classes = self._num_classes
        methods_fullname = self._methods_fullname
        methods = self._methods
        num_models = self._num_models
        num_reals = self._num_reals
        budgets = self._budgets
        resultsdir = self._resultsdir
        data_set_name = self._data_set_name
        size_entire_pool = self._size_entire_pool
        which_methods = self._which_methods

        # Save data
        np.savez(str(resultsdir) + "/data.npz", num_instances=num_instances, pool_setting=pool_setting,
                 num_classes=num_classes, num_models=num_models, num_reals=num_reals,
                 budgets=budgets, methods=methods, methods_fullname=methods_fullname, data_set_name=data_set_name,
                 size_entire_pool=size_entire_pool,
                 which_methods=which_methods)


