from xgbKantchelianAttack_RF import main
from train_rf_one import eval
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import load_svmlight_file
import pickle
import os
import numpy as np

datasets = ['breast_cancer', 'binary_mnist', 'cod-rna', 'ijcnn']
splitters = ['best', 'heuristic', 'robust']
test_path = {
    "breast_cancer": "data/breast_cancer_scale0.test",
    "binary_mnist": "data/binary_mnist0.t",
    "cod-rna": "data/cod-rna_s.t",
    "ijcnn": "data/ijcnn1s0.t"
}
zero_based = {"binary_mnist": True,
              "breast_cancer": False,
              "cod-rna": True,
              "ijcnn": False}
model_dir = 'models/sk-rf'
output_dir = 'result/sk-rf'
test_pickle_path = 'train_with_validation/data'
attack_num = {
    "breast_cancer": 100,
    "binary_mnist": 100,
    "cod-rna": 5000,
    "ijcnn": 100
}
eps_vals = {
    "breast_cancer": 0.3,
    "binary_mnist": 0.3,
    "cod-rna": 0.03,
    "ijcnn": 0.03
}
n_feat = {
    "binary_mnist": 784,
    "breast_cancer": 10,
    "cod-rna": 8,
    "ijcnn": 22
}
tree_size = {
    "breast_cancer": {
        "best": (20, 4),
        "heuristic": (20, 4),
        "robust": (80, 8)
    },
    "binary_mnist": {
        "best": (20, 14),
        "heuristic": (100, 12),
        "robust": (100, 14)
    },
    "cod-rna": {
        "best": (40, 14),
        "heuristic": (20, 14),
        "robust": (40, 14)
    },
    "ijcnn": {
        "best": (100, 14),
        "heuristic": (100, 12),
        "robust": (60, 8)
    }
}

for dataset in datasets:

    log_fname = 'results/sk-rf/sklearn_RF_attack_result_%s.csv' % dataset
    if os.path.isfile(log_fname):
        log_file = open(log_fname, "a+")
    else:
        log_file = open(log_fname, "w")
        # log_file.write("type,dataset,splitter,eps,nt,d,test acc,fpr,optimize for L0,optimize for L1,optimize for L2,optimize for Linf,time\n")
        log_file.write("type,dataset,splitter,eps,nt,d,test acc,fpr,optimize for L1,optimize for L2,optimize for Linf,time\n")

    # load test data
    with open(os.path.join(test_pickle_path, "{}_test.pickle".format(dataset)), "rb") as fin:
        x_test, y_test = pickle.load(fin)
    x_test = np.around(x_test, decimals=6)

    for splitter in splitters[0:1]:
        if splitter == 'best':
            eps = 0.0
        else:
            eps = eps_vals[dataset]
        nt, d = tree_size[dataset][splitter]
        model_path = os.path.join(model_dir, "sklearn_{}_{}_eps{}_nt{}_d{}.pickle".format(
            dataset, splitter, eps, nt, d
        ))
        with open(model_path, "rb") as fin:
            model = pickle.load(fin)
        y_hat = model.predict(x_test)
        test_acc, fpr = eval(y_test, y_hat)

        json_path = '%s.json' % model_path.split('.pickle')[0]
        cmd = 'python3 save_sklearn_rf_to_json.py \
                --model_path %s \
                --output %s' % (model_path, json_path)
        os.system(cmd)

        avg_norm = list()
        attack_time = list()
        # objectives = [0, 1, 2, -1]
        objectives = [1, 2, -1]
        for idx, order in enumerate(objectives):
            args = dict()
            args["data"] = test_path[dataset]
            args["model"] = model_path
            args["model_type"] = "sklearn"
            args["model_json"] = json_path
            args["num_classes"] = 2
            args["offset"] = 0
            args["order"] = order
            args["num_attacks"] = attack_num[dataset]
            args["guard_val"] = 1e-6
            args["round_digits"] = 20
            args["round_data"] = 6 
            args["weight"] = "No weight"
            args["out"] = os.path.join(output_dir, "{}_{}_eps{}_nt{}_d{}_order{}.csv".format(
                dataset, splitter, eps, nt, d, order
            ))
            args["adv"] = os.path.join(output_dir, "{}_{}_eps{}_nt{}_d{}_order{}.pickle".format(
                dataset, splitter, eps, nt, d, order
            ))
            args["threads"] = 8
            if zero_based[dataset]:
                fstart = 0
            else:
                fstart = 1
            args["feature_start"] = fstart
            args["initial_check"] = False
            args["no_shuffle"] = False
            attack_result = main(args)
            avg_norm.append(attack_result[idx + 1])
            attack_time.append(attack_result[-1])
            
        log_text = "{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
            "sklearn", dataset, splitter, eps, nt, d, test_acc, fpr, avg_norm[0], avg_norm[1],
            avg_norm[2], np.mean(attack_time)
        )
        log_file.write(log_text)

log_file.close()
