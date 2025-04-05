import pandas as pd
import numpy as np


columns = ['labels_per_class', 'verb_type', 'representation_type', 'accuracy']
exp_data_gold = pd.read_csv('generated/experiments_gold.csv', names=columns)
exp_data_pred = pd.read_csv('generated/experiments_pred.csv', names=columns)


def main(exp_data, eval_type): 
    filtered_data = exp_data[exp_data['labels_per_class'] == 1]
    motions = filtered_data[filtered_data['verb_type'] == 'motions']
    non_motions = filtered_data[filtered_data['verb_type'] == 'non_motions']

    motions_cap = motions[motions['representation_type'] == 'e_caption']['accuracy']
    non_motions_cap = non_motions[non_motions['representation_type'] == 'e_caption']['accuracy']

    motions_obj = motions[motions['representation_type'] == 'e_object']['accuracy']
    non_motions_obj = non_motions[non_motions['representation_type'] == 'e_object']['accuracy']

    motions_comb = motions[motions['representation_type'] == 'e_combined']['accuracy']
    non_motions_comb = non_motions[non_motions['representation_type'] == 'e_combined']['accuracy']

    motions_img = motions[motions['representation_type'] == 'e_image']['accuracy']
    non_motions_img = non_motions[non_motions['representation_type'] == 'e_image']['accuracy']

    motions_img_cap = motions[motions['representation_type'] == 'concat_image_caption']['accuracy']
    non_motions_img_cap = non_motions[non_motions['representation_type'] == 'concat_image_caption']['accuracy']

    motions_img_obj = motions[motions['representation_type'] == 'concat_image_object']['accuracy']
    non_motions_img_obj = non_motions[non_motions['representation_type'] == 'concat_image_object']['accuracy']

    motions_img_text = motions[motions['representation_type'] == 'concat_image_text']['accuracy']
    non_motions_img_text = non_motions[non_motions['representation_type'] == 'concat_image_text']['accuracy']


    if eval_type == 'gold':
        table = "\\begin{table*}[t!]\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lllllllll}\n\\hline\n \\multirow{2}{*}{GOLD} & \\multirow{2}{*}{Verb type} & \\multicolumn{3}{c}{Textual} & \\multirow{2}{*}{VIS (CNN)} & \\multicolumn{3}{c}{Concat (CNN+)} \\\\\n &  & O & C & Combined (O+C) &  & Object (O) & Captions (C)  & Combined (O+C)  \\\\ \\hline\n \\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}Paper results\\\\ PAMI 2019\\end{tabular}} & Motion & 54.60  & 73.30  & 75.60  & 58.30 & 66.60  & 74.70 & 73.80\\\\\n & Non-Motion & 57.00  & 72.70 & 72.60  & 56.10 & 66.00 & 72.20 & 71.30 \\\\\n \\hline \\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}Semi-supervised\\\\ GTG 1 Label x class\\end{tabular}} & Motion  & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f  & %.2f $\\pm$ %.1f   \\\\\n & Non-Motion  & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f  & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f \\\\\n \\hline\n \\end{tabular}\n }\n \\end{table*}" % (motions_obj.mean() * 100, motions_obj.std() * 100, motions_cap.mean() * 100, motions_cap.std() * 100, motions_comb.mean() * 100, motions_comb.std() * 100, motions_img.mean() * 100, motions_img.std() * 100, motions_img_obj.mean() * 100, motions_img_obj.std() * 100, motions_img_cap.mean() * 100, motions_img_cap.std() * 100, motions_img_text.mean() * 100, motions_img_text.std() * 100, non_motions_obj.mean() * 100, non_motions_obj.std() * 100, non_motions_cap.mean() * 100, non_motions_cap.std() * 100, non_motions_comb.mean() * 100, non_motions_comb.std() * 100, non_motions_img.mean() * 100, non_motions_img.std() * 100, non_motions_img_obj.mean() * 100, non_motions_img_obj.std() * 100, non_motions_img_cap.mean() * 100, non_motions_img_cap.std() * 100, non_motions_img_text.mean() * 100, non_motions_img_text.std() * 100) 
    elif eval_type == 'pred':
        table = "\\begin{table*}[t!]\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lllllllll}\n\\hline\n \\multirow{2}{*}{PRED} & \\multirow{2}{*}{Verb type} & \\multicolumn{3}{c}{Textual} & \\multirow{2}{*}{VIS (CNN)} & \\multicolumn{3}{c}{Concat (CNN+)} \\\\\n &  & O & C & Combined (O+C) &  & Object (O) & Captions (C)  & Combined (O+C)  \\\\ \\hline\n \\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}Paper results\\\\ PAMI 2019\\end{tabular}} & Motion & 65.1  & 54.9  & 61.6  & 58.3 & 72.6  & 63.6 & 66.5\\\\\n & Non-Motion & 59.0  & 64.3 & 65.0  & 56.1 & 63.8 & 66.3 & 66.1 \\\\\n \\hline \\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}Semi-supervised\\\\ GTG 1 Label x class\\end{tabular}} & Motion  & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f  & %.2f $\\pm$ %.1f   \\\\\n & Non-Motion  & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f  & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f & %.2f $\\pm$ %.1f \\\\\n \\hline\n \\end{tabular}\n }\n \\end{table*}" % (motions_obj.mean() * 100, motions_obj.std() * 100, motions_cap.mean() * 100, motions_cap.std() * 100, motions_comb.mean() * 100, motions_comb.std() * 100, motions_img.mean() * 100, motions_img.std() * 100, motions_img_obj.mean() * 100, motions_img_obj.std() * 100, motions_img_cap.mean() * 100, motions_img_cap.std() * 100, motions_img_text.mean() * 100, motions_img_text.std() * 100, non_motions_obj.mean() * 100, non_motions_obj.std() * 100, non_motions_cap.mean() * 100, non_motions_cap.std() * 100, non_motions_comb.mean() * 100, non_motions_comb.std() * 100, non_motions_img.mean() * 100, non_motions_img.std() * 100, non_motions_img_obj.mean() * 100, non_motions_img_obj.std() * 100, non_motions_img_cap.mean() * 100, non_motions_img_cap.std() * 100, non_motions_img_text.mean() * 100, non_motions_img_text.std() * 100) 
    else:
        raise ValueError('Unknown evaluation type')

    return table

def filter_verbs(exp_data, table_type):
    filtered_data = exp_data#.query("labels_per_class == 1 or labels_per_class == 2 or labels_per_class == 3 or labels_per_class == 5 or labels_per_class == 7")
    motions = filtered_data[filtered_data['verb_type'] == 'motions']
    non_motions = filtered_data[filtered_data['verb_type'] == 'non_motions']

    motions_cap = motions[motions['representation_type'] == 'e_caption']
    non_motions_cap = non_motions[non_motions['representation_type'] == 'e_caption']
    motions_cap_means = motions_cap.groupby('labels_per_class').mean() * 100
    non_motions_cap_means = non_motions_cap.groupby('labels_per_class').mean() * 100
    motions_cap_std = motions_cap.groupby('labels_per_class').std() * 100
    non_motions_cap_std = non_motions_cap.groupby('labels_per_class').std() * 100

    motions_obj = motions[motions['representation_type'] == 'e_object']
    non_motions_obj = non_motions[non_motions['representation_type'] == 'e_object']
    motions_obj_means = motions_obj.groupby('labels_per_class').mean() * 100
    non_motions_obj_means = non_motions_obj.groupby('labels_per_class').mean() * 100
    motions_obj_std = motions_obj.groupby('labels_per_class').std() * 100
    non_motions_obj_std = non_motions_obj.groupby('labels_per_class').std() * 100

    motions_comb = motions[motions['representation_type'] == 'e_combined']
    non_motions_comb = non_motions[non_motions['representation_type'] == 'e_combined']
    motions_comb_means = motions_comb.groupby('labels_per_class').mean() * 100
    non_motions_comb_means = non_motions_comb.groupby('labels_per_class').mean() * 100
    motions_comb_std = motions_comb.groupby('labels_per_class').std() * 100
    non_motions_comb_std = non_motions_comb.groupby('labels_per_class').std() * 100

    motions_img = motions[motions['representation_type'] == 'e_image']
    non_motions_img = non_motions[non_motions['representation_type'] == 'e_image']
    motions_img_means = motions_img.groupby('labels_per_class').mean() * 100
    non_motions_img_means = non_motions_img.groupby('labels_per_class').mean() * 100
    motions_img_std = motions_img.groupby('labels_per_class').std() * 100
    non_motions_img_std = non_motions_img.groupby('labels_per_class').std() * 100

    motions_img_cap = motions[motions['representation_type'] == 'concat_image_caption']
    non_motions_img_cap = non_motions[non_motions['representation_type'] == 'concat_image_caption']
    motions_img_cap_means = motions_img_cap.groupby('labels_per_class').mean() * 100
    non_motions_img_cap_means = non_motions_img_cap.groupby('labels_per_class').mean() * 100
    motions_img_cap_std = motions_img_cap.groupby('labels_per_class').std() * 100
    non_motions_img_cap_std = non_motions_img_cap.groupby('labels_per_class').std() * 100

    motions_img_obj = motions[motions['representation_type'] == 'concat_image_object']
    non_motions_img_obj = non_motions[non_motions['representation_type'] == 'concat_image_object']
    motions_img_obj_means = motions_img_obj.groupby('labels_per_class').mean() * 100
    non_motions_img_obj_means = non_motions_img_obj.groupby('labels_per_class').mean() * 100
    motions_img_obj_std = motions_img_obj.groupby('labels_per_class').std() * 100
    non_motions_img_obj_std = non_motions_img_obj.groupby('labels_per_class').std() * 100

    motions_img_text = motions[motions['representation_type'] == 'concat_image_text']
    non_motions_img_text = non_motions[non_motions['representation_type'] == 'concat_image_text']
    motions_img_text_means = motions_img_text.groupby('labels_per_class').mean() * 100
    non_motions_img_text_means = non_motions_img_text.groupby('labels_per_class').mean() * 100
    motions_img_text_std = motions_img_text.groupby('labels_per_class').std() * 100
    non_motions_img_text_std = non_motions_img_text.groupby('labels_per_class').std() * 100



    table_head = "\\begin{table*}[t!]\n\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{lllllllll}\n\\hline\n \\multirow{2}{*}{%s} & \\multirow{2}{*}{Verb type} & \\multicolumn{3}{c}{Textual} & \\multirow{2}{*}{VIS (CNN)} & \\multicolumn{3}{c}{Concat (CNN+)} \\\\\n &  & O & C & Combined (O+C) &  & Object (O) & Captions (C)  & Combined (O+C)  \\\\ \\hline\n" % (table_type)

    table_gold = "\\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}Paper results\\\\ PAMI 2019\\end{tabular}} & Motion & 54.60  & 73.30  & 75.60  & 58.30 & 66.60  & 74.70 & 73.80\\\\\n & Non-Motion & 57.00  & 72.70 & 72.60  & 56.10 & 66.00 & 72.20 & 71.30 \\\\\hline\n"
    
    table_pred = "\\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}Paper results\\\\ PAMI 2019\\end{tabular}} & Motion & 65.1  & 54.9  & 61.6  & 58.3 & 72.6  & 63.6 & 66.5\\\\\n & Non-Motion & 59.0  & 64.3 & 65.0  & 56.1 & 63.8 & 66.3 & 66.1 \\\\\hline\n"


    table_supervised_gold = "\\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}PAMI Supervised \\end{tabular}} & Motion & 82.3 & 78.4 & 80.0  & 82.3  & 83.0  & 82.3  & 83.0\\\\\n & Non Motion & 79.1 & 79.1  & 79.1  & 80.0 & 80.0  & 80.0  & 80.0  \\\\\\hline\n"

    table_unsupervised_gold = "\\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}PAMI Unsupervised \\end{tabular}} & Motion & 35.3 & 53.8 & 55.3  & 58.4  & 48.4  & 66.9  & 58.4\\\\\n & Non Motion & 48.6 & 53.9  & 66.0  & 55.6 & 56.5  & 56.5  & 59.1  \\\\\\hline\n"

    table_supervised_pred = "\\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}PAMI Supervised \\end{tabular}} & Motion & 80.0 & 69.2 & 70.7  & 82.3  & 83.0  & 82.3  & 83.0\\\\\n & Non Motion & 79.1 & 79.1  & 79.1  & 80.0 & 80.0  & 80.0  & 80.0  \\\\\\hline\n"

    table_unsupervised_pred = "\\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}PAMI Unsupervised \\end{tabular}} & Motion & 43.8 & 41.5 & 45.3  & 58.4  & 60.0  & 53.0  & 55.3\\\\\n & Non Motion & 46.0 & 61.7  & 55.6  & 55.6 & 52.1  & 60.0  & 55.6  \\\\\\hline\n"


    table_body = "\\multirow{2}{*}{\\begin{tabular}[c]{@{}l@{}}%s label \\end{tabular}} & Motion & %.2f $\\pm$ %.1f  & %.2f $\pm$ %.1f   & %.2f $\pm$ %.1f  & %.2f $\pm$ %.1f  & %.2f $\pm$ %.1f  & %.2f $\pm$ %.1f  & %.2f $\pm$ %.1f\\\\\n & Non Motion & %.2f $\pm$ %.1f & %.2f $\pm$ %.1f  & %.2f $\pm$ %.1f  & %.2f $\pm$ %.1f   & %.2f $\pm$ %.1f  & %.2f $\pm$ %.1f  & %.2f $\pm$ %.1f  \\\\\\hline\n"
    
    table_tail = "\\end{tabular}}\n \\end{table*}"

    table = table_head

    if table_type == 'GOLD':
        table += table_gold#table_unsupervised_gold
    else:
        table += table_pred#table_unsupervised_pred

    for i in range(len(motions_cap_means)):
        table += (table_body % (motions_cap_means.index[i], motions_obj_means['accuracy'].iloc[i], motions_obj_std['accuracy'].iloc[i], motions_cap_means['accuracy'].iloc[i], motions_cap_std['accuracy'].iloc[i], motions_comb_means['accuracy'].iloc[i], motions_comb_std['accuracy'].iloc[i], motions_img_means['accuracy'].iloc[i], motions_img_std['accuracy'].iloc[i], motions_img_obj_means['accuracy'].iloc[i], motions_img_obj_std['accuracy'].iloc[i], motions_img_cap_means['accuracy'].iloc[i],
            motions_img_cap_std['accuracy'].iloc[i], motions_img_text_means['accuracy'].iloc[i], motions_img_text_std['accuracy'].iloc[i], non_motions_obj_means['accuracy'].iloc[i], non_motions_obj_std['accuracy'].iloc[i], non_motions_cap_means['accuracy'].iloc[i], non_motions_cap_std['accuracy'].iloc[i], non_motions_comb_means['accuracy'].iloc[i], non_motions_comb_std['accuracy'].iloc[i], non_motions_img_means['accuracy'].iloc[i],
            non_motions_img_std['accuracy'].iloc[i], non_motions_img_obj_means['accuracy'].iloc[i], non_motions_img_obj_std['accuracy'].iloc[i], non_motions_img_cap_means['accuracy'].iloc[i], non_motions_img_cap_std['accuracy'].iloc[i], non_motions_img_text_means['accuracy'].iloc[i], non_motions_img_text_std['accuracy'].iloc[i]))


    # if table_type == 'GOLD':
        # table += table_supervised_gold
    # else:
        # table += table_supervised_pred

    table += table_tail
    return table
