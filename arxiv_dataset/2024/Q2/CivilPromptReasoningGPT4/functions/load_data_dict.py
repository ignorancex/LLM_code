# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:56:13 2024

@author: dansc
"""
import csv

def load_data_dict(file_path, separated=False):
    '''
    Takes in a file path.
    Returns a dictionary with 'idx','question','answer','label','analysis','explanation','context', and 'cqa' labels
    '''
    with  open(file_path, encoding='utf-8')  as inFile :
        iCSV = csv.reader(inFile)
        
        # Skip the header
        next(iCSV)
        
        idx_list = []
        question_list = []
        answer_list = []
        label_list = []
        analysis_list = []
        explanation_list = []
        context_list = []
        
        for line in iCSV:
            idx = line[0]
            idx_list.append(idx)
        
            question = line[1]
            question_list.append(question)
        
            answer = line[2]
            answer_list.append(answer)
        
            label = int(line[3])
            label_list.append(label)
        
            analysis = line[4]
            analysis_list.append(analysis)
        
            explanation = line[5]
            explanation_list.append(explanation)
        
            context = line[6]
            context_list.append(context)
            
            
            
        data_dict = {}
        data_dict['idx'] = idx_list
        data_dict['question'] = question_list
        data_dict['answer'] = answer_list
        tf_list = []
        for item in label_list:
            if item == 1:
               tf_list.append('TRUE')
            else:
                tf_list.append('FALSE')
        data_dict['label'] = tf_list
        data_dict['analysis'] = analysis_list
        data_dict['explanation'] = explanation_list
        data_dict['context'] = context_list
        data_dict['cqaal'] = [f"Introduction:\n{context}\n\nQuestion:\n{question}\n\nAnswer Candidate:\n{answer}\n\nAnalysis:\n{analysis}\n\nLabel:\n{label}" for context, question, answer, analysis, label in zip(context_list, question_list, answer_list, analysis_list, label_list)]
        data_dict['cqal'] = [f"Introduction:\n{context}\n\nQuestion:\n{question}\n\nAnswer Candidate:\n{answer}\n\nLabel:\n{label}" for context, question, answer, label in zip(context_list, question_list, answer_list, tf_list)]
        data_dict['qa'] = [f'Question:\n{question}\n\nAnswer Candidate:\n{answer}\n\nOutput:\n' for question, answer in zip(question_list, answer_list)]
        
        if separated: # generation begins at "Label:"
            data_dict['cqa'] = [f"Introduction:\n{context}\n\nQuestion:\n{question}\n\nAnswer Candidate:\n{answer}\n\nAnalysis:" for context, question, answer in zip(context_list, question_list, answer_list)]
        else: # generation begins at "Analysis:"
            data_dict['cqa'] = [f"Introduction:\n{context}\n\nQuestion:\n{question}\n\nAnswer Candidate:\n{answer}\n\nLabel:" for context, question, answer in zip(context_list, question_list, answer_list)]

    return data_dict


def load_test_dict(file_path, separated=False):
    '''
    Takes in a file path.
    Returns a dictionary with 'idx','question','answer','label','analysis','explanation','context', and 'cqa' labels
    '''
    with  open(file_path, encoding='utf-8')  as inFile :
        iCSV = csv.reader(inFile)
        
        # Skip the header
        next(iCSV)
        
        idx_list = []
        question_list = []
        answer_list = []
        context_list = []
        
        for line in iCSV:
            idx = line[0]
            idx_list.append(idx)
        
            question = line[1]
            question_list.append(question)
        
            answer = line[2]
            answer_list.append(answer)
        
            context = line[3]
            context_list.append(context)
            
            
            
        test_dict = {}
        test_dict['idx'] = idx_list
        test_dict['question'] = question_list
        test_dict['answer'] = answer_list
        test_dict['context'] = context_list
        test_dict['qa'] = [f'Question:\n{question}\n\nAnswer Candidate:\n{answer}\n\nOutput:\n' for question, answer in zip(question_list, answer_list)]
        
        if separated: # generation begins at "Label:"
            test_dict['cqa'] = [f"Introduction:\n{context}\n\nQuestion:\n{question}\n\nAnswer Candidate:\n{answer}\n\nAnalysis:" for context, question, answer in zip(context_list, question_list, answer_list)]
        else: # generation begins at "Analysis:"
            test_dict['cqa'] = [f"Introduction:\n{context}\n\nQuestion:\n{question}\n\nAnswer Candidate:\n{answer}\n\nLabel:" for context, question, answer in zip(context_list, question_list, answer_list)]

    return test_dict
