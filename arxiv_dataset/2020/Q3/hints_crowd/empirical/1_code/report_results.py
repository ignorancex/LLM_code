import pandas as pd

def report_acc_and_loss(history, filename):
    hist_df = pd.DataFrame(history.history)
    with open(filename, mode='w') as f:
        hist_df.to_csv(f)

def report_auc(aucs, filename):
    aucs_df = pd.DataFrame(aucs)
    with open(filename, mode='w') as f:
        aucs_df.to_csv(f)