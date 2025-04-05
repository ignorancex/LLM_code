import numpy as np


#task1: create plots for main paper
#task2:Appendix I.1.2 Query strategies ablation comparison
#task4:Appendix I.1.6 Outperform the best expert
#task5:Appendix I.1.5 Robustly recover from complete malicious experts environment
#task6:Appendix I.1.3 Comparing CAMS with each individual expert
#task7:Appendix I.1.4 Comparing CAMS and Model Picker in a context-free environment

'''selecting one of tasks["task1","task2","task4","task5","task6","task7","task9"]'''

#task="task2"
task="task1"
#task="task4"
#task="task5"
#task="task7"
#task="task6"
#task="task8"
#task="task9"

'''selecting one of dataset["cifar10","drift","hiv","vertebral"]'''
dataset="cifar10"
# dataset="drift"
# dataset="hiv"
# dataset="vertebral"


############################
############################

n_RS="RS"
n_CAMS_best_policy="Oracle"
n_qbc = "QBC"
n_iwal = "IWAL"
n_mp = "MP"
n_contextual_qbc = "CQBC"
n_contextual_iwal = "CIWAL"
n_CAMS_identity = "CAMS"
n_CAMS_test = "test"
n_entropy ="entropy"
n_EXP4 ="EXP4"

hue_order=[n_RS,n_CAMS_best_policy,n_qbc,n_iwal,n_mp,n_contextual_qbc,n_contextual_iwal,n_CAMS_identity]

COLOR = {n_mp: "tab:blue", n_qbc: "black", n_CAMS_identity: "r", n_CAMS_best_policy: "purple",
          "sqbc": "y", n_RS: "tab:green", n_iwal: "darkgray", "efal": "tab:brown", n_contextual_qbc: "orange",n_EXP4:"tab:blue",
          n_contextual_iwal: "tab:brown", n_CAMS_test: "b", "policy_0": "mediumpurple", "policy_1": "bisque", "policy_2":"m", "policy_3":"y",
          "policy_4":"tab:green","policy_5":"tab:pink","policy_6":"tab:brown","policy_7":"grey",
          "policy_8":"royalblue","policy_9": "plum","policy_10":"yellowgreen","policy_11":"lightsteelblue",
          "policy_12":"silver","policy_13": "tan","policy_14":"violet","policy_15":"darkslateblue",
          "policy_16":"indigo","policy_17": "limegreen","policy_18": "tab:orange","entrophy":"r","random":"purple","variance": "tab:orange",n_entropy:"r"}

MARKER = {n_mp: "s", n_qbc: "x", n_CAMS_identity: "*", n_CAMS_best_policy: "+",
          "sqbc": "v", n_RS: "P", n_iwal: "D", "efal": "o", n_contextual_qbc: "X",
          n_contextual_iwal: "d", n_CAMS_test: "b"}


#######(not used)######
#q="weighted"
#q="arg"
#q="random"
q="weighted_can_E_forward"