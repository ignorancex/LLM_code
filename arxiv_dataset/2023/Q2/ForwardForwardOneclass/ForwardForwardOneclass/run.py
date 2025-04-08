import os
import numpy as np

from base.utils import prepare_dataset, scoring
from forwardforward_general_oneclass_autodiff import GeneralOneclass

import matplotlib.pyplot as plt

def pipeline(model, train, test, val, viz_opt_landscape=False):
    name = model.name
    trainX, trainY = train[:, :-1], train[:, -1]
    valX, valY = val[:, :-1], val[:, -1]
    print("")
    print("Model: %s\n" % name + "-" * 40)
    model.fit(trainX, trainY, valX, valY, viz_opt_landscape=viz_opt_landscape)

    train_pred, train_proba = model.predict(trainX, return_probs=True)
    acc_tr, f1_tr, auc_tr, cm_tr = scoring(trainY, train_pred)
    print("Train Accuracy=%.4f | F1=%.4f | AUC=%.4f" % (acc_tr, f1_tr, auc_tr))
    print(cm_tr)

    plt.figure()
    _,bins = np.histogram(train_proba)
    plt.hist(train_proba[trainY==0], alpha=0.6, bins=bins, label='0')
    plt.hist(train_proba[trainY==1], alpha=0.6, bins=bins, label='1')
    plt.axvline(model.threshold, lw=3, linestyle='--', color='k')
    plt.legend()
    plt.savefig(os.path.join(model.save_folder, 'train_prob_histogram.png'))
    plt.close()
    
    testY = test[:, -1]
    test_pred, test_proba = model.predict(test[:, :-1], return_probs=True)
    acc_te, f1_te, auc_te, cm_te = scoring(testY, test_pred)
    print("Test Accuracy=%.4f | F1=%.4f | AUC=%.4f" % (acc_te, f1_te, auc_te))
    print(cm_te)

    plt.figure()
    _,bins = np.histogram(test_proba)
    plt.hist(test_proba[testY==0], bins=bins, alpha=0.6, label='0')
    plt.hist(test_proba[testY==1], bins=bins, alpha=0.6, label='1')
    plt.axvline(model.threshold, lw=3, linestyle='--', color='k')
    plt.legend()
    plt.savefig(os.path.join(model.save_folder, 'test_prob_histogram.png'))
    plt.close()

    return (acc_tr, f1_tr, auc_tr, cm_tr), (acc_te, f1_te, auc_te, cm_te)


def evaluate(datafile, m, nn_size, n_iters=50, **kwargs):
    acc_train, acc_test = [], []
    f1_train, f1_test = [], []
    auc_train, auc_test = [], []
    for seed in range(n_iters):
        train, test, val, labels = prepare_dataset(f"scripts/Forward-Forward-Network-main/{datafile}", do_normalize=True)
        model = m(nn_size, seed=seed, **kwargs)

        try:
            (acc_tr, f1_tr, auc_tr, cm_tr), (acc_te, f1_te, auc_te, cm_te) = pipeline(model, train, test, val)
        except:
            continue

        acc_train.append(acc_tr)
        acc_test.append(acc_te)
        f1_train.append(f1_tr)
        f1_test.append(f1_te)
        auc_train.append(auc_tr)
        auc_test.append(auc_te)

    print('acc_train')
    print( pd.Series(acc_train).describe() )
    print('acc_test')
    print( pd.Series(acc_test).describe() )
    print('f1_train')
    print( pd.Series(f1_train).describe() )
    print('f1_test')
    print( pd.Series(f1_test).describe() )
    print('auc_train')
    print( pd.Series(auc_train).describe() )
    print('auc_test')
    print( pd.Series(auc_test).describe() )

    np.savetxt(f'{model.save_folder}//acc_train.npy', acc_train)
    np.savetxt(f'{model.save_folder}//acc_test.npy', acc_test)
    np.savetxt(f'{model.save_folder}//f1_train.npy', f1_train)
    np.savetxt(f'{model.save_folder}//f1_test.npy', f1_test)
    np.savetxt(f'{model.save_folder}//auc_train.npy', auc_train)
    np.savetxt(f'{model.save_folder}//auc_test.npy', auc_test)

def read_summary(nm, folder):

    try:
        acc_train = np.loadtxt(f'{folder}//acc_train.npy')
        acc_test = np.loadtxt(f'{folder}//acc_test.npy')
        f1_train = np.loadtxt(f'{folder}//f1_train.npy')
        f1_test = np.loadtxt(f'{folder}//f1_test.npy')
        auc_train = np.loadtxt(f'{folder}//auc_train.npy')
        auc_test = np.loadtxt(f'{folder}//auc_test.npy')

        acc_test = pd.Series(acc_test)
        f1_test = pd.Series(f1_test)
        auc_test = pd.Series(auc_test)

        print(nm + " & \multicolumn{1}{c|}{" f"{round(100*acc_test.mean(),2)} ($\pm$ {round(100*acc_test.std(),2)} )" "} & " f"{round(100*acc_test.max(),2)}" " & \multicolumn{1}{c|}{" f"{round(f1_test.mean(),4)} ($\pm$ {round(f1_test.std(),4)})" "} & " f"{round(f1_test.max(),4)} " "& \multicolumn{1}{c|}{" f"{round(auc_test.mean(),4)} ($\pm$ {round(auc_test.std(),4)}) " "} & " f"{round(auc_test.max(),4)}" r" \\" )

        # print( f"{nm} {round(100*acc_test.mean(),2)}(\pm{round(100*acc_test.std(),2)}) {round(100*acc_test.max(),2)} {round(f1_test.mean(),4)}(\pm{round(f1_test.std(),4)}) {round(f1_test.max(),4)} {round(auc_test.mean(),4)} (\pm{round(auc_test.std(),4)}) {round(auc_test.max(),4)}" )
    except:
        pass

    return acc_test

if __name__ == "__main__":
    import pandas as pd

    dataname = "data_banknote_authentication"
    datafile = dataname + '.txt'

    loss_type = 'goodness'

    dims = [4,40,40,40]

    main_folder = 'results'
    save_folder = os.path.join(main_folder, dataname)
    train, test, val, labels = prepare_dataset(f"ForwardForwardOneclass/{datafile}", do_normalize=True)
    pipeline(GeneralOneclass(dims, save_folder=save_folder, nntype='ff', seed=1, loss_type=loss_type), train, test, val, viz_opt_landscape=False)
    main_folder = 'results-bp'
    save_folder = os.path.join(main_folder, dataname)
    train, test, val, labels = prepare_dataset(f"ForwardForwardOneclass/{datafile}", do_normalize=True)
    pipeline(GeneralOneclass(dims, save_folder=save_folder, nntype='bp', seed=1, loss_type=loss_type), train, test, val, viz_opt_landscape=False)
