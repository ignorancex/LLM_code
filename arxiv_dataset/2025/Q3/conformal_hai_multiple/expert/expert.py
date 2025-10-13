import numpy as np
from config import conf
import torch
import torch.nn.functional as F

from   sklearn.metrics import confusion_matrix
import pandas as pd

class Expert:
    "Expert base class"
    rng = None
    def __init__(self, conf) -> None:
        "Initialize expert accuracy and setup configuration"
        self.accuracy = conf.accuracy if conf.accuracy else None
        self.n_labels = conf.n_labels
        self.conf = conf
        Expert.rng = conf.rng

class ExpertReal(Expert):
    """Expert for real data experiments"""
    def __init__(self, X_test, model_name=None, y_humans_cnt=10, lapl_s=False, lapl_p=False) -> None: # conf, 
        super().__init__(conf)
        # Directory of expert predictions
        self.root_dir = conf.ROOT_DIR
        self.model_name = model_name
        self.X_test = X_test
        self.lapl_s = lapl_s
        self.lapl_p = lapl_p

        self.confusion_matrix = self.create_confusion_matrix()                      # 10x10
        if self.lapl_s is not False:
            self.confusion_matrix += self.lapl_p
            self.confusion_matrix /= self.confusion_matrix.sum(1).reshape(-1,1)
        self.mult_w_matrix = self.generate_multiple_humans_cm(y_humans_cnt)         # Generate varying CM's;
        self.w_matrix = self.get_w_from_confusion_matrix()                          # 10x10         # taking log not common

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)
    
    def generate_multiple_humans_cm(self, num_humans):
        humans_w_matrix = []
        cm_factor = [0.06 * val for val in range(1, num_humans+1)]
        
        # for i in range(num_humans):
        #     samples = np.random.normal(self.confusion_matrix, 0.05*self.cm_std, size=(self.n_labels, self.n_labels))   
        #     # samples[samples<0] = 1e-8                           # No negative values
        #     samples = samples + np.abs(np.repeat(samples.min(1),conf.n_labels).reshape(conf.n_labels,conf.n_labels)) + 1e-8
        #     samples = samples / samples.sum(1).reshape(-1,1)
        #     humans_w_matrix.append(np.log(samples))
        
        for _ in range(num_humans):
            human_i_cm = self.create_confusion_matrix(lim=None)
            if self.lapl_s is not None:
                human_i_cm += self.lapl_p
                human_i_cm /= human_i_cm.sum(1).reshape(-1,1)
            humans_w_matrix.append(np.log(human_i_cm))
        
        return humans_w_matrix
    
    def simulate_humans(self, X_test, lapl_smoothing=False, lapl_param=0.10):
        cm_for_test = self.cm_per_sample[X_test]                                 # 8000 x 10
        
        # Ignore for now
        # if lapl_smoothing:
        #     cm_for_test += lapl_param
        #     cm_for_test /= cm_for_test.sum(1).reshape(-1,1)
        
        cm_for_test = torch.from_numpy(cm_for_test).to(conf.device)
        y_humans = cm_for_test.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).cpu().numpy().squeeze()

        return y_humans

    def create_confusion_matrix(self, lim=None):

        if self.model_name == 'cnn_data':   # NOTE For CNN data only
            with open(f"{self.root_dir}/data/cnn_data.csv", "r") as f:
                data = np.loadtxt(f, delimiter=',')         # 10000 x 10     
                cm_per_sample = data[:, 1:11]
                cm_per_sample = cm_per_sample/cm_per_sample.sum(1).reshape(-1,1)
                self.cm_per_sample = cm_per_sample
                y = data[:, 0].astype(int)                  # 10000 # targets
        else:
            # For other data
            with open(f"{self.root_dir}/expert/cifar10h-probs.npy", "rb") as f:
                cm_per_sample = np.load(f)                          # 10000 x 10      
                self.cm_per_sample = cm_per_sample

            with open(f"{self.root_dir}/data/human_model_truth_cifar10h.csv", "r") as f:
                csv = np.loadtxt(f, delimiter=',')                  # 10000 x 21
                y = csv[:,-1].astype(int) - 1                       # 10000 # targets

        if conf.ignore_test: # changed    
            nontest = [val for val in range(cm_per_sample.shape[0]) if val not in self.X_test]
            cm_per_sample = cm_per_sample[nontest]
            y = y[nontest]          

        # Apply limit
        lim_num = int(lim*len(y)) if lim is not None else None
        y = y[:lim_num]
        cm_per_sample = cm_per_sample[:lim_num]
        
        cm = np.zeros(shape=(self.n_labels,self.n_labels))      # 10x10
        cm_std = np.zeros(shape=(self.n_labels,self.n_labels))  # 10x10 # changed    
        assert len(conf.expert_type) == 1
        for i in range(self.n_labels):        
            idx = np.argwhere(y == i).flatten()                 # 1000
            cm_std[i] = cm_per_sample[idx].std(axis=0)      # TODO experiment on varying the number of humans and varying the multiplier of the std
            if 'orig' in conf.expert_type:
                cm[i] = cm_per_sample[idx].mean(axis=0)     # Just getting the mean  # 1x10
            elif 'median' in conf.expert_type:
                cm[i] = np.median(cm_per_sample[idx], axis=0)

        self.cm_std = cm_std
        return cm + 1e-8                                    # 10x10      # Add 1e-8 for errors related to zero values

    

class ExpertRealMoreExpressive(Expert):
    """Expert for real data experiments using a more expressive context including the difficulty of samples"""
    def __init__(self, conf, y_groups) -> None:
        super().__init__(conf)
        self.root_dir = conf.ROOT_DIR
        self.y_groups = y_groups
        # Array of conditional confusion matrices, one per difficulty group 
        self.confusion_matrix = self.create_confusion_matrix()
        # Array of w_matrices, one for each conditional confusion matrix
        self.w_matrix = self.get_w_from_confusion_matrix()

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)

    def create_confusion_matrix(self):
        with open(f"{self.root_dir}/expert/cifar10h-probs.npy", "rb") as f:
            cm_per_sample = np.load(f)

        with open(f"{self.root_dir}/data/human_model_truth_cifar10h.csv", "r") as f:
            csv = np.loadtxt(f, delimiter=',')
            y = csv[:,-1].astype(int) - 1
        
        groups = set(self.y_groups)
        cm = np.zeros(shape=(len(groups), self.n_labels,self.n_labels))
        
        for g in groups:
            for i in range(self.n_labels):
                idx = np.argwhere((y == i) & (self.y_groups == g)).flatten()
                cm[g,i] = cm_per_sample[idx].mean(axis=0)
        return cm

class ExpertSynthetic(Expert):
    """Expert for synthetic data experiments"""
    def __init__(self, conf) -> None:
        super().__init__(conf)
        self.confusion_matrix = self.create_confusion_matrix(conf.class_probabilities, conf.is_oblivious)

        cm = np.zeros(shape=(self.n_labels,self.n_labels))          # 100x100
        minor_val = 0.20/(self.n_labels-1)
        for i in range(self.n_labels):
            init =  [minor_val] * i
            init += [0.80]
            init += [minor_val] * (self.n_labels-i-1)
            cm[i] = np.array(init, dtype=np.float32)
        self.confusion_matrix = cm

        self.w_matrix = self.get_w_from_confusion_matrix()
   
    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)

    def create_confusion_matrix(self, class_probs, is_oblivious):
        if is_oblivious:
            return np.ones(shape=(self.n_labels, self.n_labels))*(1/self.n_labels)

        a = class_probs                     # 1x100
        ind = list(range(self.n_labels))    # 100
        uniform_sol = self.accuracy         # 0.30
        # Assign first the uniform solution for each element of the diagonal of the confusion matrix (CM)
        x = np.ones(self.n_labels)*uniform_sol  # 100
        # Reassign random mass
        while  len(ind) >= 2:
            # Pick random pairs of the diagonal
            idx1  = Expert.rng.choice(ind)
            ind.remove(idx1)
            idx2  = Expert.rng.choice(ind)
            ind.remove(idx2)
            
            # Set normalization term
            tmp = idx1
            idx1 = idx1 if a[0][idx1] > a[0][idx2] else idx2
            idx2 = idx2 if tmp==idx1 else tmp
            ratio = a[0][idx2]/a[0][idx1]

            # Move random mass from one element to another, while keeping the CM valid
            epsilon = Expert.rng.uniform(0,np.minimum((1 - uniform_sol)*ratio, uniform_sol*ratio ))
            x[idx2] = uniform_sol - epsilon
            x[idx1] = uniform_sol + epsilon*ratio

        if len(ind):
            x[ind[0]] = uniform_sol
        
        assert (x < 1).all() and (x > 0).all() 
        
        self.better_than_random = True if (x >= 1/self.n_labels).any() else False
        
        cm = np.zeros(shape=(self.n_labels,self.n_labels))          # 100x100
        for i,ac in enumerate(x):
            # Compute the uniform solution to the off diagonal elements
            uniform_sol = (1 - ac)/(self.n_labels - 1) 
            indices = list(range(i))+list(range(i+1,self.n_labels))
            cm[i,i] = ac
            # Assign random mass to the off diagonal elements using 
            # random perturbations of the uniform solution
            while  len(indices) >= 2:
                # Pick random pairs of the off diagonal elements
                idx1 = Expert.rng.choice(indices)
                indices.remove(idx1)
                idx2 = Expert.rng.choice(indices)
                indices.remove(idx2)

                # Move random mass from the one element to the other
                epsilon = Expert.rng.normal(0, uniform_sol/6)
                cm[i,idx1] = uniform_sol - epsilon
                cm[i,idx2] = uniform_sol + epsilon

            if len(indices):
                cm[i,indices[0]] = uniform_sol
        
        # Diagnostics to confirm that the CM is valid  
        prob_faults = 0
        exp_faults = 0
        for j in range(self.n_labels):
            prob_faults += np.abs(sum(cm[j,:]) - 1) >= .001 # NOTE what is the intuition?
            exp_faults += sum(cm[:,j] * a[0]) >= 1
        acc_errors = 0
        s = 0
        for j in range(self.n_labels):
            s += cm[j, j] * a[0][j]     # scaling the diagonal
        acc_errors += np.abs(s - self.accuracy) >= .01      # maintaining this accuracy in creating synthetic dataset
        assert prob_faults==0 and exp_faults==0 and acc_errors==0

        return cm   # 100x100 # sum per row is 1 # sum per column mean is 1, std is ~0.02

class ExpertHateSpeech():
    """Expert for HateSpeech data experiments"""
    def __init__(self, dataset) -> None:
        super().__init__()
        # Directory of expert predictions
        self.hp = np.array(dataset.label_dist, dtype=np.float32)            # Nx3
        self.cm_per_sample = self.hp
        self.y = dataset.train_y
        self.n_labels = conf.n_labels
        self.confusion_matrix = self.create_confusion_matrix()              # 3x3    
        self.w_matrix = self.get_w_from_confusion_matrix()                  # 3x3     # taking log not common

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)

    def simulate_humans(self, X_test, lapl_smoothing=False, lapl_param=0.10):
        cm_for_test = self.cm_per_sample[X_test]     # 8000 x 10
        if lapl_smoothing:
            cm_for_test += lapl_param
            cm_for_test /= cm_for_test.sum(1).reshape(-1,1)
        cm_for_test = torch.from_numpy(cm_for_test).to(conf.device)
        y_humans = cm_for_test.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).cpu().numpy().squeeze()

        return y_humans

    def create_confusion_matrix(self):
        cm_per_sample = self.hp     # N x 3    # TODO Get the label dist

        # changed    
        use_real = False
        if use_real:
            nontest = [val for val in range(cm_per_sample.shape[0]) if val not in self.X_test]
            cm_per_sample = cm_per_sample[nontest]

        # changed    
        if use_real:
            y = y[nontest]          # TODO WHAT IF WE USE CAL ONLY

        cm = np.zeros(shape=(self.n_labels, self.n_labels))      # 10x10
        for i in range(self.n_labels):          # TODO Why use everything? even the test set values?
            idx = np.argwhere(self.y == i).flatten()                 # 1000
            cm[i] = cm_per_sample[idx].mean(axis=0)             # just getting the mean  # 1x10  # TODO Can you use the standard dev for reweighting? Say for those minority classes, apply uncertainty
            # Fixed expert v1            # changes    
            # init =  [0.09] * i
            # init += [0.19]
            # init += [0.09] * (9-i)
            # cm[i] = np.array(init, dtype=np.float32)
            # Fixed expert v2              # changes    
            # init =  [0.00222222222] * i
            # init += [0.98]
            # init += [0.00222222222] * (9-i)
            # cm[i] = np.array(init, dtype=np.float32)
            # Weird expert
            # init = [0.10555555] * i
            # init += [0.05]
            # init += [0.10555555] * (9-i)
            # cm[i] = np.array(init, dtype=np.float32)
            # All-equal expert
            # init =  [0.10] * i
            # init += [0.10]
            # init += [0.10] * (9-i)
            # cm[i] = np.array(init, dtype=np.float32)
        return cm                       # 3x3

class ExpertImageNet16H():
    """Expert for ImageNet16H data experiments"""
    def __init__(self, X_test, noise_version, y_humans_cnt, n_labels=None) -> None:
        super().__init__()
        # Directory of expert predictions
        # self.hp = np.array(dataset.label_dist, dtype=np.float32)            # 1200x16
        # self.cm_per_sample = self.hp                    
        # self.X_test = dataset.X_test_indices        
        # self.X_cal = dataset.X_cal_indices          
        # self.X_est = dataset.X_est_indices          
        # self.y = dataset.train_y      
        self.root_dir = conf.ROOT_DIR
        self.noise_version = noise_version
        self.X_test = X_test              
        if n_labels == None:
            self.n_labels = conf.n_labels
        else:
            self.n_labels = n_labels
            
        self.cm_per_sample, self.y = self.get_expert_cm()
        # Dirichlet
        self.diag_acc=0.75
        self.strength=1

        self.confusion_matrix = self.create_confusion_matrix()              # 16x16    
        self.mult_w_matrix = self.generate_multiple_humans_cm(y_humans_cnt)
        self.w_matrix = self.get_w_from_confusion_matrix()                  # 16x16     # taking log not common

    def get_expert_cm(self):
        # load the csv file
        data_behavioral = pd.read_csv(
            self.root_dir
            + "/data/human_only_classification_6per_img_export.csv"
        )           # 28997x32

        data_behavioral = data_behavioral[
            data_behavioral["noise_level"] == int(self.noise_version)
        ]           # 7247x32
        data_behavioral = data_behavioral[
            [
                "participant_id",
                "image_id",
                "image_name",
                "image_category",
                "participant_classification",
                "confidence",
            ]
        ]

        # get mapping from category to index # aligned with how the dataset is made
        self.category_to_idx = {"knife":0,
                                "keyboard":1,
                                "elephant":2,
                                "bicycle":3,	
                                "airplane":4,
                                "clock":5,
                                "oven":6,
                                "chair":7,
                                "bear":8,
                                "boat":9,
                                "cat":10,
                                "bottle":11,
                                "truck":12,
                                "car":13,
                                "bird":14,
                                "dog":15
                                }

        image_id_categories = dict(
            zip(data_behavioral["image_id"], data_behavioral["image_category"])
        )
        # for each image name, get all the participant classifications
        image_id_to_participant_classifications = {}
        for image_id in data_behavioral["image_id"].unique():
            image_id_to_participant_classifications[image_id] = data_behavioral[
                data_behavioral["image_id"] == image_id
            ]["participant_classification"].values      # there are 6 participants?, per image, there are 6 ?

        # sample a single classification from the participant classifications
        image_id_to_single_participant_classification = {}
        for image_id in image_id_to_participant_classifications:
            image_id_to_single_participant_classification[
                image_id
            ] = np.random.choice(image_id_to_participant_classifications[image_id])

        image_id = [x for x in range(1,1201)]
        
        # image_names = os.listdir(
        #     self.data_dir + "/Noisy Images/phase_noise_" + self.noise_version
        # )
        # image_names = [x for x in image_names if x.endswith(".png")]
        # # remove png extension
        # image_names = [x[:-4] for x in image_names]
        # image_paths = np.array(
        #     [
        #         self.data_path
        #         + x
        #         + ".png"
        #         for x in image_names
        #     ]
        # )
        # get label for image ids
        image_id_labels = np.array(
            [self.category_to_idx[image_id_categories[x]] for x in image_id]
        )
        # get prediction for image names
        image_id_human_predictions = np.array(
            [
                self.category_to_idx[image_id_to_single_participant_classification[x]]
                for x in image_id
            ]
        )                   # only uses single participant
        
        label_dist = []
        for x in image_id:
            dist = [0] * 16
            for h_pred in image_id_to_participant_classifications[x]:
                cls = self.category_to_idx[h_pred]
                dist[cls] += 1
            dist = np.array(dist) / 6   # There are six human experts
            label_dist.append(dist)
        
        # self.label_dist = label_dist
        return np.array(label_dist), image_id_labels
    
    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix + 1e-8)

    def simulate_humans(self, X_test, lapl_smoothing=False, lapl_param=0.10):
        cm_for_test = self.cm_per_sample[X_test]                # 8000 x 10
        if lapl_smoothing:
            cm_for_test += lapl_param
            cm_for_test /= cm_for_test.sum(1).reshape(-1,1)
        cm_for_test = torch.from_numpy(cm_for_test).to(conf.device)
        y_humans = cm_for_test.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).cpu().numpy().squeeze()

        return y_humans

    def generate_multiple_humans_cm(self, num_humans):
        humans_w_matrix = []
        cm_factor = [0.06*val for val in range(1, num_humans+1)]
        # for i in range(num_humans):
        #     samples = np.random.normal(self.confusion_matrix, 0.05*self.cm_std, size=(self.n_labels, self.n_labels))   
        #     # samples[samples<0] = 1e-8              # No negative values
        #     samples = samples + np.abs(np.repeat(samples.min(1),conf.n_labels).reshape(conf.n_labels,conf.n_labels)) + 1e-8
        #     samples = samples / samples.sum(1).reshape(-1,1)
        #     humans_w_matrix.append(np.log(samples))
        for i in range(num_humans):
            humans_w_matrix.append(np.log(self.create_confusion_matrix(lim=None)))
        
        return humans_w_matrix
    
    def create_confusion_matrix(self, lim=None):
        # changed    
        if conf.ignore_test:
            nontest = [val for val in range(self.cm_per_sample.shape[0]) if val not in self.X_test]
            cm_per_sample = self.cm_per_sample[nontest]
            y = self.y[nontest]         

        # Apply limit
        lim_num = int(lim*len(y)) if lim is not None else None
        y = y[:lim_num]
        cm_per_sample = cm_per_sample[:lim_num]
        
        cm = np.zeros(shape=(self.n_labels, self.n_labels))             # 16x16
        cm_std = np.zeros(shape=(self.n_labels,self.n_labels))  # 10x10 # changed    
        assert len(conf.expert_type) == 1
        for i in range(self.n_labels):                                 
            idx = np.argwhere(y == i).flatten()                    # 1000
            cm_std[i] = cm_per_sample[idx].std(axis=0)   
            cm[i] = cm_per_sample[idx].mean(axis=0)                     # just getting the mean  # 1x10
        
        # # Get MAP estimate of confusion matrix
        # y_h = self.simulate_humans(np.concatenate((np.array(self.X_est), np.array(self.X_cal))))
        # # for human in range(num_humans): # cm is done for each human    
        # alpha, beta = self.get_dirichlet_params(self.diag_acc, self.strength, conf.n_labels)            # self.diag_acc 0.75, self.strength 1
        # prior_matr = np.eye(conf.n_labels) * alpha + (np.ones(conf.n_labels) - np.eye(conf.n_labels)) * beta
        # posterior_matr = 1. * confusion_matrix(self.y, y_h, labels=np.arange(conf.n_labels))
        # posterior_matr += prior_matr        # add the prior_matr
        # posterior_matr = posterior_matr.T
        # posterior_matr = (posterior_matr - np.ones(conf.n_labels)) / (np.sum(posterior_matr, axis=0, keepdims=True) - conf.n_labels)
        # cm = posterior_matr
        
        self.cm_std = cm_std
        return (cm + 1e-8)/ cm.sum(0)

    def get_dirichlet_params(self, acc, strength, n_cls):
        # acc: desired off-diagonal accuracy
        # strength: strength of prior

        # Returns alpha,beta where the prior is Dir((beta, beta, . . . , alpha, . . . beta))
        # where the alpha appears for the correct class

        '''
        i think alpha here corresponds to the gamma on page 5's piecewise function
        '''

        beta = 0.1
        alpha = beta * (n_cls - 1) * acc / (1. - acc)

        alpha *= strength
        beta *= strength

        alpha += 1
        beta += 1

        return alpha, beta

class ExpertChestXray():
    """Expert for ChestXray data experiments"""
    def __init__(self, dataset, n_labels=None) -> None:
        super().__init__()
        # Directory of expert predictions
        self.hp = np.array(dataset.label_dist, dtype=np.float32)            # Nx2
        self.cm_per_sample = self.hp
        self.X_test = dataset.X_test_indices
        self.y = dataset.test_y
        if n_labels == None:
            self.n_labels = conf.n_labels
        else:
            self.n_labels = n_labels
        self.confusion_matrix = self.create_confusion_matrix()              # 2x2   
        self.w_matrix = self.get_w_from_confusion_matrix()                  # 2x2  # taking log not common

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix + 1e-8)

    def simulate_humans(self, X_test, lapl_smoothing=False, lapl_param=0.10):
        cm_for_test = self.cm_per_sample[X_test]     # 1048 x 2
        if lapl_smoothing:
            cm_for_test += lapl_param
            cm_for_test /= cm_for_test.sum(1).reshape(-1,1)
        cm_for_test = torch.from_numpy(cm_for_test).to(conf.device)
        y_humans = cm_for_test.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).cpu().numpy().squeeze()

        return y_humans

    def create_confusion_matrix(self):
        cm_per_sample = self.hp         # N x 2    # Get the label dist

        # changed    
        if conf.ignore_test:
            nontest = [val for val in range(cm_per_sample.shape[0]) if val not in self.X_test]
            cm_per_sample = cm_per_sample[nontest]
            self.y = self.y[nontest]          

        cm = np.zeros(shape=(self.n_labels, self.n_labels))      # 2x2
        for i in range(self.n_labels):                               
            idx = np.argwhere(self.y == i).flatten()             # 1000
            cm[i] = cm_per_sample[idx].mean(axis=0)              # just getting the mean  # 1x2  # TODO Can you use the standard dev for reweighting? Say for those minority classes, apply uncertainty
        return cm                       # 2x2

class ExpertCompass():
    """Expert for COMPASS data experiments"""
    def __init__(self, dataset, X_test_indices=None) -> None:
        super().__init__()
        # Directory of expert predictions
        self.hp = np.array(dataset.label_dist, dtype=np.float32)        # Nx3
        self.cm_per_sample = self.hp
        self.X_test = dataset.X_test_indices
        self.y = dataset.train_y_labels
        self.n_labels = conf.n_labels
        self.confusion_matrix = self.create_confusion_matrix()              # 3x3    
        self.w_matrix = self.get_w_from_confusion_matrix()                  # 3x3  # taking log not common

    def get_w_from_confusion_matrix(self):
        return np.log(self.confusion_matrix)

    def simulate_humans(self, X_test, lapl_smoothing=False, lapl_param=0.10):
        cm_for_test = self.cm_per_sample[X_test]            # 8000 x 10
        if lapl_smoothing:
            cm_for_test += lapl_param
            cm_for_test /= cm_for_test.sum(1).reshape(-1,1)
        cm_for_test = torch.from_numpy(cm_for_test).to(conf.device)
        y_humans = cm_for_test.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).cpu().numpy().squeeze()

        return y_humans

    def create_confusion_matrix(self):
        cm_per_sample = self.hp     # N x 3    # TODO Get the label dist

        # changed    
        if conf.ignore_test:
            nontest = [val for val in range(cm_per_sample.shape[0]) if val not in self.X_test]
            cm_per_sample = cm_per_sample[nontest]
            self.y = self.y[nontest]          # TODO WHAT IF WE USE CAL ONLY

        cm = np.zeros(shape=(self.n_labels, self.n_labels))      # 10x10
        for i in range(self.n_labels):          # TODO Why use everything? even the test set values?
            idx = np.argwhere(self.y == i).flatten()                 # 1000
            cm[i] = cm_per_sample[idx].mean(axis=0)             # just getting the mean  # 1x10  # TODO Can you use the standard dev for reweighting? Say for those minority classes, apply uncertainty
        return cm                       # 3x3