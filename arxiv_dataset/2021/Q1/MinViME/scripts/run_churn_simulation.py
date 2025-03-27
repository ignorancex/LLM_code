"""
Churn Simulation 

We explore properties of a minimum viable model for churn problems with different criteria.

In a churn situation every False Negative involves the cost of losing a customer.
The cost of a False Positive is the cost of an uneccessary intervention.

A True Positive means that you have a chance of mitigating the churn.
Therefore the impact of a model making a True Positive is a reduction in the expected
loss of a Churn. 

For example, if churn involves a $10,000 loss per customer, and the
probability of an intervention working is 20%. Then True Positive reduces the expected
loss from $10,000 to $8,000. Or the model delivered $2,000 worth of value. 

This script will produce a table of model performance criteria for problems of 
varying definitions in terms of the costs of both churn and intervention.
"""

import sys
import pandas as pd

sys.path.append('../minvime')
import estimator_classification as esti 

fns = [-10000,-8000,-6000,-4000,-2000,-1000]
fps = [-500,-400,-300,-200,-100]
success_rate = 0.2

tn = 0
roi = 500000
cases = 1000000
baserate = 0.005

rez = pd.DataFrame()

for fn in fns:
    for fp in fps:
        # A true positive is a reduction in expected loss (after considering intervention cost)
        tp = fn - (fn * success_rate) + fp
        current_churn_loss = cases * baserate * fn
        # To deliver ROI we need to lift ROI from the current churn losses.
        minroi = current_churn_loss + roi
        auc, prec, recall, fprs, tprs = esti.estimate_binary_model_requirements(
            tp=tp, fp=fp, tn=tn, fn=fn, cases=cases, 
            baserate=baserate, minroi=minroi
        )
        rez = rez.append({'Total Loss':current_churn_loss, 'Loss p Cust':fn, 'Intvn Cost':fp, 'auc':auc, 'precision':prec, 'recall':recall}, ignore_index=True)

print("Churn Model Requirements")
print("Criteria | Min ROI $%i | Cases %i | Base Churn Rate %f " % (roi,cases,baserate) )
print(rez)

