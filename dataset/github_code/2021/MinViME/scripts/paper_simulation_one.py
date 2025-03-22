import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../minvime')
import estimator_classification as esti # The file ../minvime/estimator_classification.py

fprates = [0.0, 0.00001, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.015,
               0.02,0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07,
               0.075, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
               0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93, 0.94,
               0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 1.0]
tp = 1000
fp = -140
tn = 0
fn = 0
minroi = 10000
cases = 1000000
baserate = 0.0015

auc, prec, recall, x3, y3 = esti.estimate_binary_model_requirements(
     tp=tp, fp=fp, tn=tn, fn=fn, cases=cases, 
     baserate=baserate, minroi=minroi
)

auc, x1, y1 = esti.generate_roc_auc(fprates, 0.5, 2)
auc, x2, y2 = esti.generate_roc_auc(fprates, 0.9, 6)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  sharey=True,figsize=(12,4))
fig.suptitle('Synthetically Generated ROC Plots')
ax1.plot(x1, y1)
ax2.plot(x2, y2)
ax3.plot(x3, y3)
ax1.set(xlabel='FPR', ylabel='TPR')
ax2.set(xlabel='FPR')
ax2.label_outer()
ax3.set(xlabel='FPR')
ax3.label_outer()

plt.savefig("../paper/images/roc_plots.png")



