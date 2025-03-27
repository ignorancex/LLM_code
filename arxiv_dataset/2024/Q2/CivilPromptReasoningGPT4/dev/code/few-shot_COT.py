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
        model="gpt-3.5-turbo",
        messages=[
        { 
            "role": "system",
            "content": '''
Given an "Introduction" to the topic, a "Question" and an "Answer Candidate", first state your reasoning before classifying the given Answer Candidate as TRUE or FALSE. State your answer after your reasoning in the format "The Answer Candidate is {mask}" where mask is either TRUE or FALSE. If you find yourself uncertain based on the available information, classify the given Answer Candidate as FALSE.
 
Text: """
Introduction:
Rule 12 lays several traps for the unwary defendant. She may waive the disfavored defenses by filing a pre-answer motion and leaving some out. But she may also waive them by answering, without filing a pre-answer motion at all, and leaving them out. See Rule 12(h)(1)(B). Consider whether Goliath has waived his objection in this example. 
   
Question:
Among the missing. Goliath sues David for slander. David answers on the merits, denying that he made the offending statement. Six weeks later, he realizes that Goliath has filed suit in an improper venue. 
   
Answer Candidate:
Service is a formal act by which a court asserts its authority over the defendant. The cases require that it be done by the book in order to acquire that authority. Knowledge of the action by the defendant or someone who works for it will not substitute for formal submission to the court’s authority through service (or a proper waiver). So we have to see if the defendant has properly waived service in any of these cases. In A, Darrow has attested to telling Lorenzo’s counsel about the case. But he hasn’t served him with process, and he hasn’t complied with the Rule 4(d) procedure for waiving it. The court has not acquired jurisdiction over Lorenzo.  The answer candidate is FALSE.
 
Label:
It is true that Chem-Pro would very likely be subject to general in personam jurisdiction in the Southern District of New York based on its principal place of business in the Western District. And a defendant that is subject to personal jurisdiction in a state is subject to jurisdiction in any court within that state, not just in a particular district. But even if Chem-Pro is subject to personal jurisdiction in the Southern District based on having its principal place of business in the state, that does not mean that it ‘‘resides’’ in the Southern District for venue purposes under §1391(d). For a multi-district state, §1391(d) focuses on the defendant’s contacts in the particular district. We must ask whether those contacts would suffice to support personal jurisdiction over it if that district were a state. Here, Chem-Pro has no contacts in the Southern District, when viewed as a state, so it does not ‘‘reside’’ there under the venue statute. Chem-Pro does ‘‘reside’’ in the Western District of New York, because its contacts there—having its principal place of business there— would support general in personam jurisdiction over it if the Northern District were a separate state. Under §1391(b)(1), venue is proper in ‘‘a judicial district in which any defendant resides, if all defendants are residents of the State in which the district is located.’’ Since Compagna resides in the Northern District of New York, both she and Chem-Pro ‘‘reside’’ in New York. Thus, venue is proper in a judicial district within New York where either resides. Either the Northern or Western District of New York is a proper venue. The Answer Candidate is TRUE.
 
Introduction:
As the previous section explains, the scope of diversity jurisdiction in Article III and that conveyed to the federal district courts by Congress aren’t the same. The Strawbridge rule illustrates one situation in which the statutory grant is narrower than the constitutional authority. Another example is the amount-in-controversy requirement. Article III, §2 contains no monetary restriction on diversity jurisdiction; it broadly authorizes jurisdiction over all cases between citizens of different states. Congress’s grant of diversity jurisdiction to the federal district courts, however, includes an amount-in-controversy requirement, in order to keep small diversity cases out of federal court. See 28 U.S.C. §1332(a) (granting jurisdiction over diversity cases in which ‘‘the amount in controversy exceeds the sum or value of $75,000, exclusive of interest or costs’’). Here’s a mediocre question, included to make a point.  
 
Question:
By the numbers. A diversity case cannot be heard in federal court unless the amount in controversy is at least 
 
Answer Candidate:
$75,000.01, exclusive of interest and costs.
 
Label:
Service is a formal act by which a court asserts its authority over the defendant. The cases require that it be done by the book in order to acquire that authority. Knowledge of the action by the defendant or someone who works for it will not substitute for formal submission to the court’s authority through service (or a proper waiver). So we have to see if the defendant has properly waived service in any of these cases. In A, Darrow has attested to telling Lorenzo’s counsel about the case. But he hasn’t served him with process, and he hasn’t complied with the Rule 4(d) procedure for waiving it. The court has not acquired jurisdiction over Lorenzo.  The answer candidate is TRUE
"""        
'''
     
        },
        {
        "role": "user",
        "content": cqa
        }
    ],
    temperature=0.7,
    max_tokens=256
  )

    print(json.dumps({'index': i, 'output': response.choices[0].message.content}))