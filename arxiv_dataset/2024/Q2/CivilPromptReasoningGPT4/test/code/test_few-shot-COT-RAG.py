# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:19:39 2023
@author: Dan Schumacher
"""

import openai
from openai import OpenAI
from dotenv import load_dotenv
import json
import os

# HOMEBREW FUNCTIONS 
import sys
sys.path.append('../../functions')

# =============================================================================
# LOAD DATA
# =============================================================================
file_path = '../../data/RAG_test.json'
with open(file_path, 'r', encoding = 'utf-8') as in_file:
    data_dict = json.load(in_file)
    
# =============================================================================
# API CONFIG
# =============================================================================

# Load environment variables from the .env file
load_dotenv('../../data/.env')

# Get the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if api_key is None:
  raise ValueError("API key is missing. Make sure to set OPENAI_API_KEY in your environment.")

# Set the API key for the OpenAI client
openai.api_key = api_key
client = OpenAI(api_key=api_key)

# =============================================================================
# PROMPTING
# =============================================================================
true_false_list = []
for idx, prompt_head, cqa in zip( data_dict['idx'], data_dict['prompt_head'], data_dict['cqa']):
   
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
        {
            "role": "system",
            "content": '''Given an "Introduction" to a topic, a "Question" and an "Answer Candidate", your job is to generate two sections of output. The first section you will generate is a detailed step-by-step Analysis section that evaluates the validity of the Answer Candidate with a high amount of confidence. The second section you will generate is a final Label stating whether the Answer Candidate is TRUE or FALSE. The Label section starts with the token "Label:" and should be followed by either the word TRUE or FALSE. DO NOT RETURN ANY OTHER TOKENS FOR THE Label SECTION!'''
        },
        {   
            "role": "user",
            "content": prompt_head
        },
        {
            "role": "user",
            "content": cqa
        }
   ],
   temperature=0.7,
   max_tokens=512
  )

    print(json.dumps({'idx': idx, 'output': response.choices[0].message.content}))