# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:19:39 2023
@author: Dan Schumacher
"""
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# HOMEBREW FUNCTIONS 
import sys
sys.path.append('../../functions')
from load_data_dict import load_data_dict

# =============================================================================
# LOAD DATA
# =============================================================================
data_dict = load_data_dict('../../data/dev.csv')

for i in range(len(data_dict['cqa'])):
    data_dict['cqa'][i] = data_dict['cqa'][i].replace("Label:", "Analysis:")
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
# PROMPTING
# =============================================================================
true_false_list = []
for i, cqa in enumerate(data_dict['cqa']):    
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
        {
            "role": "system",
            "content": '''
            Given an "Introduction" to a topic, a "Question" and an "Answer Candidate", your job is to generate two sections of output. The first section you will generate is a detailed step-by-step Analysis section that evaluates the validity of the Answer Candidate with a high amount of confidence. The second section you will generate is a final Label stating whether the Answer Candidate is TRUE or FALSE. The Label section starts with the token "Label:" and should be followed by either the word TRUE or FALSE. DO NOT RETURN ANY OTHER TOKENS FOR THE Label SECTION!
            """        
            '''
        },
        {
          "role": "user",
          "content": cqa
        }
      ],
      temperature=0.7,
      max_tokens=512,
      top_p=1
    )
        
    print(json.dumps({'index': i, 'output': response.choices[0].message.content}))  