from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from config import conf
import numpy as np

def make_dataset(run_no, machine_accuracy):
    """Synthetic dataset generation"""
    # Class separation parameter to control the difficulty of 
    # the dataset, and so the accuracy of the classifier
    clas_sep = conf.class_sep[conf.n_labels][machine_accuracy]      # 1.75 for difficulty

    # Dirichlet distribution for the label distribution. TODO What if balanced dist?
    balanced = False         # changed   
    if balanced:
        clas_sep = 2.18     # for 30% # hyperparameter changing # different for each confidence of human
        cls_prob = [1/conf.n_labels] * conf.n_labels
        class_prob = np.array(cls_prob, dtype=np.float32)  # NOTE not change global conf.class_probabilities[0]
        x, y = make_classification(n_samples=conf.data_size, n_features=20, n_classes=conf.n_labels, n_informative=15,\
            n_redundant=5, class_sep=clas_sep, flip_y=0, weights=class_prob, random_state=1) # 10000x20, 10000 # conf.class_probabilities[0] is 100
                    
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, stratify=y, test_size=conf.test_split, random_state=42+run_no)    # changed    for stratify    # 8000x20, 2000x20, 8000, 2000

        X_train, X_cal_est, y_train, y_cal_est = train_test_split(
            X_train, y_train, stratify=y_train, test_size=2*conf.cal_split, random_state=42+run_no)   # 6400x20, 1600x20, 6400, 1600
        
        # Estimation and calibration sets have the same size
        X_cal, X_est, y_cal, y_est = train_test_split(
            X_cal_est, y_cal_est, stratify=y_cal_est, test_size=0.5, random_state=42+run_no)            # 800x20, 800x20, 800, 800
    else:
        x, y = make_classification(n_samples=conf.data_size, n_features=20, n_classes=conf.n_labels, n_informative=15,\
        n_redundant=5, class_sep=clas_sep, flip_y=0, weights=conf.class_probabilities[0], random_state=1) # 10000x20, 10000 # conf.class_probabilities[0] is 100
        # NOTE dataset is kinda imbalanced :O
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=conf.test_split, random_state=42+run_no)    # changed    for stratify    # 8000x20, 2000x20, 8000, 2000

        X_train, X_cal_est, y_train, y_cal_est = train_test_split(
            X_train, y_train, test_size=2*conf.cal_split, random_state=42+run_no)   # 6400x20, 1600x20, 6400, 1600
        
        # Estimation and calibration sets have the same size
        X_cal, X_est, y_cal, y_est = train_test_split(
            X_cal_est, y_cal_est, test_size=0.5, random_state=42+run_no)            # 800x20, 800x20, 800, 800

    return X_train, X_test,X_cal,X_est, y_train, y_test,y_cal, y_est 

def make_dataset_real(run_no, file_ground_truth):
    """Real dataset"""
    if file_ground_truth == 'r_low_acc':
        file_ground_truth = 'densenet-bc-L190-k40'      # Why DenseNet #changed   
    with open(f"{conf.ROOT_DIR}/data/{file_ground_truth}.csv", "r") as f:
        csv = np.loadtxt(f, delimiter=',')              # 10000x21
        # Ground truth labels
        y = csv[:,0].astype(int)            # 10000
        # Models need only the index of the sample as input
        x = np.arange(y.shape[0])          # changed   

    changed_by = False
    if not changed_by:
        X_test, X_cal_est, y_test, y_cal_est = train_test_split(
            x, y, stratify=y, test_size=2*conf.cal_split, random_state=42+run_no)               # changed to 0.30   # changed    to include stratify 
        
        # Estimation and calibration sets have the same size
        X_cal, X_est, y_cal, y_est = train_test_split(
            X_cal_est, y_cal_est, stratify=y_cal_est, test_size=0.5, random_state=42+run_no)    # changed    to include stratify
    
    else:   # ignore
        X_test, X_cal_est, y_test, y_cal_est = train_test_split(
            x, y, stratify=y, test_size=1150/9150, random_state=42+run_no) # changed    to hardcode test size

        to_remove_or_add = []   
        test_cnt = y_test.shape[0]
        for i in range(10): 
            to_remove_or_add.append(test_cnt/10 - (y_test==i).sum())        # negative if need to remove from this class and positive if need to add
        
        X_test_copy = X_test.copy()
        y_test_copy = y_test.copy()
        for idx, lbl in enumerate(y_test):
            if to_remove_or_add[lbl] < 0:   # remove from y_test
                X_cal_est = np.concatenate((X_cal_est.reshape(-1), np.array(X_test[idx]).reshape(-1)))
                y_cal_est = np.concatenate((y_cal_est.reshape(-1), np.array(y_test[idx]).reshape(-1)))
                X_test_copy[idx] = -1
                y_test_copy[idx] = -1
                to_remove_or_add[lbl] += 1
            if to_remove_or_add[lbl] > 0:   # add to y_test
                idx_to_add = [index for index, comp in enumerate(y_cal_est == lbl) if comp==True][0]
                X_test_copy = np.concatenate((X_test_copy.reshape(-1), np.array(X_cal_est[idx_to_add]).reshape(-1)))
                y_test_copy = np.concatenate((y_test_copy.reshape(-1), np.array(y_cal_est[idx_to_add]).reshape(-1)))
                X_cal_est = np.delete(X_cal_est, idx_to_add)
                y_cal_est = np.delete(y_cal_est, idx_to_add)
                to_remove_or_add[lbl] -= 1
        mask_neg_one = np.isclose(y_test_copy, -1, equal_nan=True)  # True for -1
        y_test = np.delete(y_test_copy, np.where(mask_neg_one)[0])
        X_test = np.delete(X_test_copy, np.where(mask_neg_one)[0])

        # Estimation and calibration sets have the same size
        X_cal, X_est, y_cal, y_est = train_test_split(
            X_cal_est, y_cal_est, stratify=y_cal_est, test_size=0.5, random_state=42+run_no) # changed    to include stratify
    
    return X_test, X_cal, X_est, y_test, y_cal, y_est

def make_dataset_imagenet(run_no, file_ground_truth):
    """Real dataset"""
    with open(f"{conf.ROOT_DIR}/data/imagenet_080.csv", "r") as f:
        csv = np.genfromtxt(f, delimiter=',', dtype=str, filling_values='')  
        # Ground truth labels
        y = csv[:,3]           
        y = imagenet_word_to_int(y)
        # Models need only the index of the sample as input
        x = np.arange(y.shape[0])        # changed   

    changed_by = False
    if not changed_by:
        X_test, X_cal_est, y_test, y_cal_est = train_test_split(
            x, y, stratify=y, test_size=2*conf.cal_split, random_state=42+run_no)   # changed to 0.30 # changed    to include stratify 
        
        # Estimation and calibration sets have the same size
        X_cal, X_est, y_cal, y_est = train_test_split(
            X_cal_est, y_cal_est, stratify=y_cal_est, test_size=0.5, random_state=42+run_no) # changed    to include stratify
        
    return X_test, X_cal, X_est, y_test, y_cal, y_est

def imagenet_word_to_int(words):
    y_int = []
    dict_word_to_int = {"knife":0,
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
    for word in words:
        y_int.append(dict_word_to_int[word])
        
    return np.array(y_int, dtype=int)
        
    

def make_dataset_real_with_difficulties(run_no):
    """Real dataset assuming different levels of difficulty across samples"""
    file_ground_truth = 'densenet-bc-L190-k40'
    with open(f"{conf.ROOT_DIR}/data/{file_ground_truth}.csv", "r") as f:
        csv = np.loadtxt(f, delimiter=',')
        # Ground truth labels
        y = csv[:,0].astype(int)
        # Models need only the index of the sample as input
        x = np.arange(10000) 

    # Each sample is assinged a difficulty level
    y_groups = find_difficulties()
    # Y labels extended with the difficulty groups
    y_with_dif = np.array(list(zip(y, y_groups)))
    
    X_test, X_cal_est, y_test, y_cal_est = train_test_split(
        x, y_with_dif, test_size=2*conf.cal_split, random_state=42+run_no)
    
    # Estimation and calibration sets have the same size
    X_cal, X_est, y_cal, y_est = train_test_split(
         X_cal_est, y_cal_est, test_size=0.5, random_state=42+run_no)
    
    return X_test, X_cal, X_est, y_test, y_cal, y_est, y_groups

def find_difficulties():
    """Define difficulty levels of all samples"""
    with open(f"{conf.ROOT_DIR}/expert/cifar10h-probs.npy", "rb") as f:
        cm_per_sample = np.load(f)

    with open(f"{conf.ROOT_DIR}/data/human_model_truth_cifar10h.csv", "r") as f:
        csv = np.loadtxt(f, delimiter=',')
        y = csv[:,-1].astype(int) - 1
    
    acc_per_sample = np.zeros(len(cm_per_sample))
    y_groups = np.zeros(len(cm_per_sample), dtype='int')
    # Average accuracy of experts for each sample
    for i,l in enumerate(y):
        acc_per_sample[i] = cm_per_sample[i][l]
    # Thresholds defining the difficulty levels
    threshold_hard = np.quantile(acc_per_sample, 0.25)
    threshold_easy = np.quantile(acc_per_sample, 0.5)

    for i,acc in enumerate(acc_per_sample):
        if acc > threshold_easy:
            # Easy samples
            y_groups[i] = 0
        elif acc < threshold_hard:
            # Hard samples
            y_groups[i] = 2
        else:
            # Medium difficulty samples
            y_groups[i] = 1
    
    return y_groups