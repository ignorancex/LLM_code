import sys
import pandas as pd
from mpl_toolkits import mplot3d 
import matplotlib.pyplot as plt

sys.path.append('../minvime')
import estimator_classification as esti # The file ../minvime/estimator_classification.py

tp = 0
fp = 0
tn = 0
fn = 0
minroi = 100000
cases = 1000000

baserates = [0.00001, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.003, 0.004, 0.008, 0.01, 0.012, 0.016, 0.02, 0.03, 0.05, 0.07, 0.1 ]
benefit_cost_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0001 ]
benefit_total_ratios = [0.2, 0.1, 0.05, 0.03, 0.01, 0.008, 0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0003, 0.0001]

rez = pd.DataFrame()

for baser in baserates:
    for bcr in benefit_cost_ratios:
        for btr in benefit_total_ratios:
            tp = btr * minroi
            fp = -(tp * bcr)
            auc, prec, recall, fprs, tprs = esti.estimate_binary_model_requirements(
                tp=tp, fp=fp, tn=tn, fn=fn, cases=cases, 
                baserate=baser, minroi=minroi
            )
            rez = rez.append({'baserate':baser, 'benefit_cost_ratio':bcr, 'benefit_total_ratio':btr, 'AUC':auc }, ignore_index=True)


def q1(x):
    return x.quantile(0.25)

def q2(x):
    return x.quantile(0.75)

f = {'AUC': ['mean', 'median', 'std', q1,q2]}

def plot_with_bands(x, y, lower, upper, title, xlabel, ylabel, fname):
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    ax.fill_between(x, upper, lower, color='grey', alpha=0.3, label="Min Viable AUC - 25% to 75%")
    ax.plot(x, y, color='blue', lw=2, label="Mean Min Viable AUC")
    plt.xscale('log')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.savefig(fname)


def generate_plot_over_group(groupby_col, title, xlabel, ylabel, filename):
    brdf = rez.groupby([groupby_col])
    df1 = brdf.agg(f).reset_index()
    x = df1[groupby_col].tolist()
    y = df1['AUC']['mean'].tolist()
    lower_q = df1['AUC']['q1'].tolist()
    upper_q = df1['AUC']['q2'].tolist()
    plot_with_bands(x, y, lower_q, upper_q, title, xlabel, ylabel, filename )

######################################################################################
generate_plot_over_group('baserate', "Minimum Viable AUC by Baserate", "Baserate", "AUC", "../paper/images/AUC_by_Baserate.png")
generate_plot_over_group('benefit_cost_ratio', "Minimum Viable AUC by Cost to Benefit Ratio", "Cost/Benefit Ratio", "AUC", "../paper/images/AUC_by_Cost_to_Benefit.png")
generate_plot_over_group('benefit_total_ratio', "Minimum Viable AUC by Case Benefit to Total ROI Ratio", "Benefit/Total ROI Ratio", "AUC", "../paper/images/AUC_by_Benefit_to_Total.png")

#################################################################################################
# NOW A SURFACE PLOT

oneink = rez[rez["benefit_total_ratio"]==0.0001]

x = oneink["baserate"].tolist() 
y = oneink["benefit_cost_ratio"].tolist() 
z = oneink["AUC"].tolist() 

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,6))
ax = Axes3D(fig)
surf = ax.plot_trisurf(x, y, z, cmap=cm.coolwarm, linewidth=0.2)
ax.set_xlabel('Baserate', fontsize=18)
ax.set_ylabel('Cost/Benefit Ratio', fontsize=18)
ax.set_zlabel('AUC', fontsize=14)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.zticks(fontsize=10)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Minimum Viable AUC for $10^{-4}$ Benefit/Total ROI Ratio', fontsize=20)
plt.savefig("../paper/images/Baserate_vs_Cost_to_Benefit.png")


