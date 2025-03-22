'''
To draw figures in the paper
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

def draw_acc_all(acc_scores_all, labels):
    # ACL 2024 Figure 1
    plt.figure()
    for data, label in zip(acc_scores_all, labels):
        plt.plot(np.array(data)*100, label=label, marker="o", markersize=4, linestyle='dashed')
        plt.scatter(np.argmax(data), np.max(data)*100, marker="*", s=80)
    plt.gca().yaxis.set_major_formatter('{:.1f}'.format)
    plt.legend()
    plt.xlabel("Layer Index")
    plt.ylabel("Accuracy (%)")
    plt.savefig("output/ACL_figures/layer_prediction.pdf")

if __name__ == "__main__":
    acc_path = "output/test_acc.csv"
    marks = ['prompt2', 'repeat', 'repeat_prev'] # Fig1: ['base', 'repeat'] # Fig2: 
    df = pd.read_csv(acc_path)
    acc_scores_all = [df[i].to_list() for i in marks]
    marks[0] = 'prompt'

    '''For Fig2 only
    # get BERT scores
    bert_path = "output/bert_comp.txt"
    mark = "BERT_large"
    df_Bert = pd.read_csv(bert_path, sep="\t")
    acc_Bert = df_Bert['bert ACC'].to_list()[:25]
    marks.append(mark)
    acc_scores_all.append(acc_Bert)
    '''

    draw_acc_all(acc_scores_all, marks)