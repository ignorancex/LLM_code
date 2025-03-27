# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:46:45 2023

@author: dansc
"""

# =============================================================================
# LOAD PACKAGES
# =============================================================================
import csv
from datasets import Dataset
import numpy as np
import math

# =============================================================================
# LOAD DATA
# =============================================================================
def load_sliding_window_complex(csv_file, remainder_divider = 1):
    
    '''
    parses csv with sliding window complex
    

    Parameters
    ----------
    csv_file : csv file to munch on

    Returns: Dataset object
    '''
    
    with  open(csv_file, encoding='utf-8')  as inFile :
        iCSV = csv.reader(inFile)
        
        # Skip the header
        next(iCSV)
        
        idxs = []
        questions = []
        labels = []
        cqas = []
        
        for line in iCSV:
           
            
            idx = int(line[0])
            
            
            question = line[1]
            questions.append(question)
            
            answer = line[2]
            #answers.append(answer)
            
            label = int(line[3])
            
            
            context = line[6]
            
            
            # =============================================================================
            # Understanding Sliding Window Processing
            # =============================================================================
            """
            Sliding window processing involves dividing a text sequence (context, question, answer) into
            segments or "windows" to fit within a fixed length, in our case: 512 tokens. The following steps
            outline the process:
            
            1. Calculate the number of bins needed by dividing the total length of the text sequence
               by the fixed window size 512. Round up to ensure all content is covered.
            
            2. Determine the length of the question and answer independently from the context. Subtract
               the combined length of the question and answer from the window size to find available
               slots for the context.
            
            3. Divide the length of the context by the available slots to estimate the number of windows
               required for the context.
            
            4. Use np.array_split() to evenly distribute the words of the context into the calculated
               number of windows, ensuring efficient segmentation.
            
            5. Assemble each window along with the question and answer, and represent them as strings
               to create the final sliding window representation.
            
            This process ensures that the information in the text sequence is divided into manageable
            segments for further analysis or model input.
            """
            # ============================================================================           
            
            
            # generate tags, question, and answer
            qa = '<Q> ' + question + ' </Q> ' + '<A> ' + answer + ' </A>'
            qa_length = len(qa.split())
            
            #generate tags and context
            c = '<C> ' + context + ' </C> '
            c_length = len(c.split())
            
            # How many words context can we fit? (we must include all of the question and answer)
            num_slots_avail = (512 - qa_length)/remainder_divider # we are only using half the space to leave room for subtokenization
            num_windows = math.ceil(c_length/num_slots_avail)
            
            # Create a list of lists that are: context1 + qa, context2 + qa etc. for however many windows we need.
            windows = np.array_split(c.split(), num_windows)
            windows = [window.tolist() for window in windows]
            for window in windows:
                window = ' '.join(window)
                window += qa
                cqas.append(window)
                idxs.append(idx)
                labels.append(label)
            
        
        # Build dictionary
        data_dict = {}
        data_dict['idx'] = idxs
        data_dict['cqa'] = cqas
       # data_dict['answer'] = answers
        data_dict['label'] = labels
        
        # Build Dataset object
        data_set = Dataset.from_dict(data_dict)
        data_set = data_set.with_format("torch")
        return data_set

# =============================================================================
# TESTING          
# =============================================================================

# train = load_sliding_window_complex('./../data/train.csv',2)           
# dev = load_sliding_window_complex('./../data/dev.csv',2)      


# train.column_names
# print(type(train['idx'][0]), #list of ints
#       type(train['cqa'][0]), #list of lists
#       type(train['label'][0])) #list of ints

# print(len(train['idx']), #list of ints
#       len(train['cqa']), #list of lists
#       len(train['label'])) #list of ints

# print(type(train))

# max_length = 0
# min_length = 512
# mean_length = 0
# for  lst in train['cqa']:
#     if len(lst.split())> max_length:
#         max_length = len(lst.split())
#     if len(lst.split())< min_length:
#         min_length = len(lst.split())
#     mean_length += len(lst.split())
# mean_length /= len(train['cqa'])

# print('max_length:',max_length) # 511 < 512 YAY!
# print('min_length:',min_length) # 511 < 0 YAY!
# print('mean_length:', round(mean_length,2))
