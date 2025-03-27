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
for i, cqa in enumerate(data_dict['cqa']):    
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
        {
            "role": "system",
            "content": '''
            Given an “Introduction” to the topic, a “Question” and an “Answer Candidate”, label the given Answer Candidate as "TRUE" or "FALSE". Only return either “TRUE” or “FALSE”, do not return any other tokens.
            '''
        },
        {
          "role": "user",
          "content": cqa
        }
      ],
      temperature=0.7,
      max_tokens=6,
      top_p=1
    )
        
    print(json.dumps({'index': i, 'output': response.choices[0].message.content}))  