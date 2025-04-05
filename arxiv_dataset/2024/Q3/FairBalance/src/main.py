from exp import Exp
import pandas as pd
import numpy as np
from stats import is_larger
import time

def run(data, repeat = 30, seed = 1):
    np.random.seed(seed)
    treatments = ["None", "Reweighing", "FairBalance", "FairBalanceVariant", "fairsmote", "fairsmote-situation",
                  "Reweighing+fairway", "FairBalance+fairway", "FairBalanceVariant+fairway", "Reweighing+fairsituation",
                  "FairBalance+fairsituation", "FairBalanceVariant+fairsituation", "None+fairway", "None+fairsituation"]

    metrics = ["Accuracy", "AUC", "mEOD", "mAOD", "smEOD", "smAOD", "Runtime"]
    columns = ["Treatment"] + metrics
    test_result = {column: [] for column in columns}
    train_result = {column: [] for column in columns}
    for treat in treatments:
        test_result["Treatment"].append(treat)
        train_result["Treatment"].append(treat)
        twopart = treat.split("+")
        if len(twopart)<2:
            if treat.split("-")[0]=="fairsmote":
                treat_predata = treat
                treat_preprocess = "None"
            else:
                treat_preprocess = treat
                treat_predata = "None"
        else:
            treat_preprocess = twopart[0]
            treat_predata = twopart[1]
        test_accuracy = []
        test_auc = []
        test_eod = []
        test_aod = []
        test_smeod = []
        test_smaod = []
        test_time = []
        train_accuracy = []
        train_auc = []
        train_eod = []
        train_aod = []
        train_smeod = []
        train_smaod = []
        train_time = []
        for i in range(repeat):
            t1 = time.time()
            exp = Exp(data = data, treatment = treat_preprocess, predata = treat_predata)
            m_train, m_test = exp.one_exp()
            runtime = time.time() - t1

            test_accuracy.append(m_test.accuracy())
            test_auc.append(m_test.auc())
            test_eod.append(m_test.eod())
            test_aod.append(m_test.aod())
            test_smeod.append(m_test.seod())
            test_smaod.append(m_test.saod())
            test_time.append(runtime)

            train_accuracy.append(m_train.accuracy())
            train_auc.append(m_train.auc())
            train_eod.append(m_train.eod())
            train_aod.append(m_train.aod())
            train_smeod.append(m_train.seod())
            train_smaod.append(m_train.saod())
            train_time.append(runtime)

        test_result["Accuracy"].append(test_accuracy)
        test_result["AUC"].append(test_auc)
        test_result["mEOD"].append(test_eod)
        test_result["mAOD"].append(test_aod)
        test_result["smEOD"].append(test_smeod)
        test_result["smAOD"].append(test_smaod)
        test_result["Runtime"].append(test_time)

        train_result["Accuracy"].append(train_accuracy)
        train_result["AUC"].append(train_auc)
        train_result["mEOD"].append(train_eod)
        train_result["mAOD"].append(train_aod)
        train_result["smEOD"].append(train_smeod)
        train_result["smAOD"].append(train_smaod)
        train_result["Runtime"].append(train_time)

    for key in metrics:
        if key == "Accuracy" or key == "AUC":
            rank = ranking(test_result[key], better="higher")
        else:
            rank = ranking(test_result[key], better="lower")
        test_result[key] = ["r%d: %.2f (%.2f)" % (rank[i], np.median(test_result[key][i]),
                                                  np.quantile(test_result[key][i], 0.75) - np.quantile(
                                                      test_result[key][i], 0.25)) for i in range(len(test_result[key]))]

    for key in metrics:
        if key == "Accuracy" or key == "AUC":
            rank = ranking(train_result[key], better="higher")
        else:
            rank = ranking(train_result[key], better="lower")
        train_result[key] = ["r%d: %.2f (%.2f)" % (rank[i], np.median(train_result[key][i]),
                                                  np.quantile(train_result[key][i], 0.75) - np.quantile(
                                                      train_result[key][i], 0.25)) for i in range(len(train_result[key]))]

    df_test = pd.DataFrame(test_result, columns=columns)
    df_test.to_csv("../results/test/" + data + ".csv", index=False)

    df_train = pd.DataFrame(train_result, columns=columns)
    df_train.to_csv("../results/train/" + data + ".csv", index=False)



def ranking(result, better="lower"):
    if better == "higher":
        result = (1.0 - np.array(result)).tolist()
    medians = [np.median(r) for r in result]
    order = np.argsort(medians)
    rankings = [0]*len(order)
    rank = 0
    pre = []
    for i, id in enumerate(order):
        if i==0:
            pre.extend(result[id])
        else:
            diff = is_larger(result[id], pre)
            if diff > 1:
                rank += 1
                pre = result[id]
        rankings[id] = rank
    return rankings

if __name__ == "__main__":
    datasets = ["adult", "compas", "heart", "bank", "german", "default", "student-mat", "student-por"]
    for data in datasets:
        run(data, seed=1)