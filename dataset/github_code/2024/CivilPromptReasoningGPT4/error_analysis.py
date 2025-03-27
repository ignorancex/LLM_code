# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:42:52 2024

@author: dansc
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
from sklearn.metrics import confusion_matrix
import re
import seaborn as sns
import matplotlib.pyplot as plt
from functions.load_data_dict import load_data_dict
# from calculations import (
#     get_results,
#     scores,
#     extract_output_values_from_json_file,
#     read_json_lines_and_extract_output,
#     re_scan,
#     cot_process_output,
#     count_true_false,
#     list_to_csv,
#     look_at_nones,
#     ensemble,
#     ensemble_w_cutoff
# )

# =============================================================================
# LOAD THE DATA
# =============================================================================
# GET LIST OF THE ACTUAL LABELED VALUES AND TURN INTO A LIST
dev = load_data_dict('./data/dev.csv')

# GET LIST OF OUR FINAL PREDICTIONS AND TURN INTO LIST
dev_preds = pd.read_csv('./dev/data/dev_final.csv')
# ADD PREDICTIONS TO DICTIONARY
dev['actual'] = [1 if label == 'TRUE' else 0 for label in dev['label']]
dev['preds']  = dev_preds['baseline'].tolist()

df_dev = pd.DataFrame(dev)

append_me = []
for act, pred in zip(dev['actual'], dev['preds']):
    if (act == 0) and (pred == 1):
        append_me.append('False Positive')
    elif (act == 1) and (pred == 0):
        append_me.append('False Negative')
    elif (act == 1) and (pred == 1):
        append_me.append('True Positive')
    else:
        append_me.append('True Negative')

df_dev['type'] = append_me

dev = df_dev.to_dict(orient='list')
# =============================================================================
# E.D.A.
# =============================================================================
# CONFUSION MATRIX
print(pd.DataFrame(
    confusion_matrix(dev['actual'], dev['preds']),
    index = ['actFalse', 'actTrue'],
    columns = ['predFalse', 'predTrue']
    ))

# FILTER THE DATAFRAM FOR FALSE NEGATIVES
false_neg_df = df_dev[(df_dev['actual'] == 1) & (df_dev['preds'] == 0)]
false_neg_df.reset_index(drop=True, inplace=True)
false_neg = false_neg_df.to_dict(orient='list')

# FILTER THE DATAFRAM FOR FALSE POSITIVES
false_pos_df = df_dev[(df_dev['actual'] == 0) & (df_dev['preds'] == 1)]
false_pos_df.reset_index(drop=True, inplace=True)
false_pos = false_pos_df.to_dict(orient='list')

# ALL ERRORS
errors_df = df_dev[((df_dev['actual'] == 0) & (df_dev['preds'] == 1)) |(df_dev['actual'] == 1) & (df_dev['preds'] == 0)]
errors_df.reset_index(drop=True, inplace=True)
errors = errors_df.to_dict(orient='list')
errors['idx'] = [int(error) for error in errors['idx']]


# =============================================================================
# Classes and functions
# =============================================================================
# MAKE AN OBJECT THAT CAN COUNT NEGATIVE WORDS
class NegationLexiconClassifier():
    def __init__(self):
        self.neg_words = set()
        with open('./data/negation_lexicon.txt', encoding='utf-8') as iFile:
            for row in iFile:
                words = row.strip().split()
                self.neg_words.update(map(str.lower, words))
                

    def count_up(self, cqaal):
        num_neg_words = 0
        num_non_neg_words = 0

        # SPLIT THE TEXT INTO INDIVIDUAL WORDS
        words_in_cqaal = cqaal.lower().split()

        for word in words_in_cqaal:
            if word in self.neg_words:
                num_neg_words += 1
            else:
                num_non_neg_words += 1

        return [num_neg_words, num_non_neg_words]

def count_numbers(text):
    num_numbers = 0
    
    words_in_text = text.split()  # No need to convert to lower case
    
    for word in words_in_text:
        if re.search(r'\d', word):  # Using regex to match any word containing numerical characters
            num_numbers += 1
    
    return num_numbers


# make a function that counts sections signs
def count_section_signs(text):
    num_section_signs = 0
    words_in_text = text.lower().split()
    
    for word in words_in_text:
        if '§' in word:
            num_section_signs += 1
    
    return num_section_signs

def plot_2x2_subplots(y1, y2, y3, y4, data):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # Plotting y1
    sns.barplot(x='idx', y=y1, hue='type', data=errors_df, ax=axs[0, 0], order=errors_df['idx'])
    axs[0, 0].set_xlabel('Observation')
    axs[0, 0].set_ylabel(y1)
    axs[0, 0].set_title(f'{y1} Plot')

    # Plotting y2
    sns.barplot(x='idx', y=y2, hue='type', data=errors_df, ax=axs[0, 1], order=errors_df['idx'])
    axs[0, 1].set_xlabel('Observation')
    axs[0, 1].set_ylabel(y2)
    axs[0, 1].set_title(f'{y2} Plot')

    # Plotting y3
    sns.barplot(x='idx', y=y3, hue='type', data=errors_df, ax=axs[1, 0], order=errors_df['idx'])
    axs[1, 0].set_xlabel('Observation')
    axs[1, 0].set_ylabel(y3)
    axs[1, 0].set_title(f'{y3} Plot')

    # Plotting y4
    sns.barplot(x='idx', y=y4, hue='type', data=errors_df, ax=axs[1, 1], order=errors_df['idx'])
    axs[1, 1].set_xlabel('Observation')
    axs[1, 1].set_ylabel(y4)
    axs[1, 1].set_title(f'{y4} Plot')

    plt.tight_layout()
    plt.show()
    
def plot_1x3_subplots(y1, y2, y3, data):
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))

    # Plotting y1
    sns.barplot(x='idx', y=y1, hue='type', data=errors_df, ax=axs[0], order=errors_df['idx'])
    axs[0].set_xlabel('Observation')
    axs[0].set_ylabel(y1)
    axs[0].set_title(f'{y1} Plot')

    # Plotting y2
    sns.barplot(x='idx', y=y2, hue='type', data=errors_df, ax=axs[1], order=errors_df['idx'])
    axs[1].set_xlabel('Observation')
    axs[1].set_ylabel(y2)
    axs[1].set_title(f'{y2} Plot')

    # Plotting y3
    sns.barplot(x='idx', y=y3, hue='type', data=errors_df, ax=axs[2], order=errors_df['idx'])
    axs[2].set_xlabel('Observation')
    axs[2].set_ylabel(y3)
    axs[2].set_title(f'{y3} Plot')

    plt.tight_layout()
    plt.show()

# =============================================================================
# IMPLEMENTATION  CQAAL
# =============================================================================
                
# INSTANTIATE THAT OBJECT
neg_lc = NegationLexiconClassifier()

# IN ERRORS, COUNT NEGATIVE WORDS AND TOTAL WORDS
neg_counts = []
total_words = []
for cqaal in errors['cqaal']:
    neg_counts.append(neg_lc.count_up(cqaal)[0])
    total_words.append(neg_lc.count_up(cqaal)[0] + neg_lc.count_up(cqaal)[1])
errors['neg_counts'] = neg_counts
errors['total_words'] = total_words

numbers_count = []
for cqaal in errors['cqaal']:
    numbers_count.append(count_numbers(cqaal))
errors['number_count'] = numbers_count


section_signs_counts = []
for cqaal in errors['cqaal']:
    section_signs_counts.append(count_section_signs(cqaal))
errors['section_signs_count'] = section_signs_counts

# Calculate proportions

neg_prop = []
for neg, tot in zip(errors['neg_counts'],errors['total_words']):
    neg_prop.append(neg/tot)
errors['neg_prop'] = neg_prop  
    
ssc_prop = []
for ssc, tot in zip(errors['section_signs_count'],errors['total_words']):
    ssc_prop.append(ssc/tot)
errors['ssc_prop'] = ssc_prop      

number_prop = []
for nc, tot in zip(errors['number_count'],errors['total_words']):
    number_prop.append(nc/tot)
errors['number_prop'] = number_prop  

# =============================================================================
# IMPLEMENTATION OTHER
# =============================================================================

# Define the sections and count types
sections = ['context', 'question', 'answer', 'analysis']
count_types = ['neg', 'tot', 'neg_prop', 'num', 'num_prop', 'sign', 'sign_prop']

# Iterate over sections and count types
for section in sections:
    for count_type in count_types:
        # Initialize the list for the current count type
        errors[f"{section}_{count_type}"] = []

# Iterate over the data and calculate counts and proportions
for c, q, a, ex, neg_count, ssc_count, number_count, total_words in zip(errors['context'], errors['question'], errors['answer'], errors['analysis'], errors['neg_counts'], errors['section_signs_count'], errors['number_count'], errors['total_words']):
    # CONTEXT
    context_neg, context_nonneg = neg_lc.count_up(c)
    context_tot = context_neg + context_nonneg
    
    # QUESTION
    question_neg, question_nonneg = neg_lc.count_up(q)
    question_tot = question_neg + question_nonneg
    
    # ANSWER
    answer_neg, answer_nonneg = neg_lc.count_up(a)
    answer_tot = answer_neg + answer_nonneg
    
    # ANALYSIS
    analysis_neg, analysis_nonneg = neg_lc.count_up(ex)
    analysis_tot = analysis_neg + analysis_nonneg

    # Append counts and proportions to the corresponding lists
    for section, counts in zip(sections, [(context_neg, context_tot), (question_neg, question_tot), (answer_neg, answer_tot), (analysis_neg, analysis_tot)]):
        neg, tot = counts
        errors[f"{section}_neg"].append(neg)
        errors[f"{section}_tot"].append(tot)
        errors[f"{section}_neg_prop"].append(neg / tot)
        errors[f"{section}_num"].append(number_count)
        errors[f"{section}_num_prop"].append(number_count / tot)
        errors[f"{section}_sign"].append(ssc_count)
        errors[f"{section}_sign_prop"].append(ssc_count / tot)

# =============================================================================
# PLOTTING VALUES
# =============================================================================

# SET UP
errors['type'] = ['False Positive' if label == 'FALSE' else 'False Negative' for label in errors['label']]
errors_df = pd.DataFrame(errors)
errors_df.sort_values(by='type',ascending=False, inplace=True)

# PLOT ON ENTIRE
plot_2x2_subplots(
    y1= 'neg_counts',
    y2= 'total_words',
    y3= 'number_count',
    y4= 'section_signs_count',
    data = errors_df
    )

plot_1x3_subplots(
    y1 = 'neg_prop',
    y2 = 'number_prop',
    y3 ='ssc_prop',
    data = errors_df
    )

# ITERATE OVER SUBSECTIONS
# VALUES
for item in ['context','question','answer','analysis']:
    plot_2x2_subplots(
        y1= f'{item}_neg',
        y2= f'{item}_tot',
        y3= f'{item}_num',
        y4= f'{item}_sign',
        data = errors_df
        )

# PROPORTION
for item in ['context','question','answer','analysis']:
    plot_1x3_subplots(
        y1= f'{item}_neg_prop',
        y2= f'{item}_num_prop',
        y3= f'{item}_sign_prop',
        data = errors_df
        )

for item in errors_df[errors_df['type'] == 'False Positive']['cqaal'][5:]:
    print(item)

len(errors_df[errors_df['type'] == 'False Positive']['cqaal'])
