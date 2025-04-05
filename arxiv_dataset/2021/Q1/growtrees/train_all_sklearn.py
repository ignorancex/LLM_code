import os
from train_rf_one import main

num_trees = [20, 40, 60, 80, 100]
depth = [4, 6, 8, 10, 12, 14]
splitters = ["best", "heuristic", "robust"]
dataset = ["breast_cancer", "binary_mnist", "cod-rna", "ijcnn"]
data_path = {
    "breast_cancer": ["data/breast_cancer_scale0.train",
                      "data/breast_cancer_scale0.test"],
    "binary_mnist": ["data/binary_mnist0", "data/binary_mnist0.t"],
    "cod-rna": ["data/cod-rna_s", "data/cod-rna_s.t"],
    "ijcnn": ["data/ijcnn1s0", "data/ijcnn1s0.t"]
}
eps_vals = {
    ('robust', 'breast_cancer'): 0.3,
    ('robust', 'binary_mnist'): 0.3,
    ('robust', 'cod-rna'): 0.03,
    ('robust', 'ijcnn'): 0.03,
    ('heuristic', 'breast_cancer'): 0.3,
    ('heuristic', 'binary_mnist'): 0.3,
    ('heuristic', 'cod-rna'): 0.03,
    ('heuristic', 'ijcnn'): 0.03,
}
tree_vals = {
    ('best', 'breast_cancer'): (20, 4),
    ('best', 'binary_mnist'): (20, 14),
    ('best', 'cod-rna'): (40, 14),
    ('best', 'ijcnn'): (100, 14),
    ('robust', 'breast_cancer'): (80, 8),
    ('robust', 'binary_mnist'): (100, 14),
    ('robust', 'cod-rna'): (40, 14),
    ('robust', 'ijcnn'): (60, 8),
    ('heuristic', 'breast_cancer'): (20, 4),
    ('heuristic', 'binary_mnist'): (100, 12),
    ('heuristic', 'cod-rna'): (20, 14),
    ('heuristic', 'ijcnn'): (100, 12),
}
n_feat = {"binary_mnist": 784,
          "breast_cancer": 10,
          "cod-rna": 8,
          "ijcnn": 22}
zero_based = {"binary_mnist": True,
              "breast_cancer": False,
              "cod-rna": True,
              "ijcnn": False}

out_dir = "models/sk-rf"
training_log = open("logs/sklearn_training.csv", "w")
training_log.write("type,dataset,splitter,eps,nt,d,test acc,fpr\n")

count = 0

for ds in dataset:
    for splitter in splitters:
        nt, d = tree_vals[splitter, ds]
        print('-------')
        print("Training {} with {} splitter...".format(ds, splitter))
        eps = eps_vals.get((splitter, ds), 0.0)
        train, test = data_path[ds]
        is_robust = bool(splitter != "best")
        fout = os.path.join(out_dir, "sklearn_{}_{}_eps{}_nt{}_d{}.pickle".format(ds, splitter, eps, nt, d))
        acc, fpr = main(train=train, validation=None, test=test, model_path=fout, binary_class=True, nfeat=n_feat[ds],
                zero_start=zero_based[ds], robust=is_robust, splitter=splitter, eps=eps, criterion="gini",
                nt=nt, max_depth=d, max_features=0.5, verbose=0, round_data=6)

        log_text = "{},{},{},{},{},{},{},{}\n".format(
            "sklearn", ds, splitter, eps, nt, d, acc, fpr
        )
        training_log.write(log_text)
training_log.close()
