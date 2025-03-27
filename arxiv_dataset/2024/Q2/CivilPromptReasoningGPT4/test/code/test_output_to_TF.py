# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:49:30 2024

@author: dansc
"""
# =============================================================================
# Imports
# =============================================================================
# EXTERNAL PACKAGES
import json
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

import sys
sys.path.append('../../functions')

# HOMEBREWED PACKAGES
from load_data_dict import load_test_dict
from calculations import extract_output_values_from_json_file

# =============================================================================
# Load the data
# =============================================================================
# output = extract_output_values_from_json_file('./json_results/TEST/test_one-shot_COT.out')
output = extract_output_values_from_json_file('../data/test_few-shot_COT.out')
test_dict = load_test_dict('../../data/test.csv')
test_dict['output'] = output

# =============================================================================
# API CONFIG
# =============================================================================

# Load environment variables from the .env file
load_dotenv('./data/.env')

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
    raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY in your environment.")

# Set the API key for the OpenAI client
openai.api_key = api_key
client = OpenAI(api_key=api_key)
# =============================================================================
# Prompt
# =============================================================================
true_false_list = []

for idx, qa, output in zip(test_dict['idx'], test_dict['qa'], test_dict['output']):
   
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
        {
            "role": "system",
            "content": '''Given a "Question", "Answer Candidate" and "Output", determine if the "Answer Candidate" is TRUE or FALSE by comparing the "Question" and "Answer Candidate" to the "Output". Only return TRUE or FALSE; do not return any other tokens.'''
        },
        {   
            "role": "user",
            "content": qa + output + '\n\nOutput:\n'
        }
   ],
   temperature=0.7,
   max_tokens=6
  )
    # Export the resulting list
    print(json.dumps({'idx': idx, 'output': response.choices[0].message.content}))



# =============================================================================
# FIX THE NONE TRUE FALSES
# =============================================================================
# 0, 9, 24, 26, 58, 76, 78
# true_false_list = []
# # These indexes didn't return simple true/false values
# indexes = [0, 9, 24, 26, 58, 76, 78]

# for i in indexes:
#     idx = test_dict['idx'][i]
#     qa = test_dict['qa'][i]
#     output = test_dict['output'][i]
    
#     response = client.chat.completions.create(
#         model="gpt-4-1106-preview",
#         messages=[
#             {
#                 "role": "system",
#                 "content": '''Given a "Question", "Answer Candidate" and "Output", determine if the "Answer Candidate" is TRUE or FALSE by comparing the "Question" and "Answer Candidate" to the "Output". Only return TRUE or FALSE; do not return any other tokens.'''
#                 },
#             {
#                 "role": "user",
#                 "content": qa + output + '\n\nLabel:\n'
#                 }
#             ],
#         temperature=0.7,
#         max_tokens=6
#     )
#       # Export the resulting list
#     print(json.dumps({'idx': idx, 'output': response.choices[0].message.content}))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    