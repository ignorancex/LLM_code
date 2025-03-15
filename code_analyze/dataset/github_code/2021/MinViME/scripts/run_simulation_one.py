import sys
import pandas as pd

sys.path.append('../minvime')
import estimator_classification as esti # The file ../minvime/estimator_classification.py

tps = [20000,10000,8000,6000,4000,2000,1000]
fps = [-900,-800,-600,-500,-400,-200,-100]

tn = 0
fn = 0
minroi = 100000
cases = 1000000
baserate = 0.001

rez = pd.DataFrame()

for tp in tps:
    for fp in fps:
        auc, prec, recall, fprs, tprs = esti.estimate_binary_model_requirements(
            tp=tp, fp=fp, tn=tn, fn=fn, cases=cases, 
            baserate=baserate, minroi=minroi
        )
        rez = rez.append({'TP Benefit':tp, 'FP Cost':fp, 'AUC':auc, 'precision':prec, 'recall':recall}, ignore_index=True)
print("Model Requirements")
print("Criteria | Min ROI $%i | Cases %i | Base Churn Rate %f " % (minroi,cases,baserate) )
print(rez)

