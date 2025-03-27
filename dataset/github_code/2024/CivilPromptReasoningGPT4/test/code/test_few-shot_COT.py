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
from load_data_dict import load_test_dict

# =============================================================================
# LOAD DATA
# =============================================================================
data_dict = load_test_dict('../../data/test.csv')
# I need to change label to analysis
for i in range(len(data_dict['cqa'])):
    data_dict['cqa'][i] = data_dict['cqa'][i].replace("Label:", "Analysis:")

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
Given an "Introduction" to a topic, a "Question" and an "Answer Candidate", your job is to generate two sections of output. The first section you will generate is a detailed step-by-step Analysis section that evaluates the validity of the Answer Candidate with a high amount of confidence. The second section you will generate is a final Label stating whether the Answer Candidate is TRUE or FALSE. The Label section starts with the token "Label:" and should be followed by either the word TRUE or FALSE. DO NOT RETURN ANY OTHER TOKENS FOR THE Label SECTION!

Text: “””
Introduction:
Rule 12 lays several traps for the unwary defendant. She may waive the disfavored defenses by filing a pre-answer motion and leaving some out. But she may also waive them by answering, without filing a pre-answer motion at all, and leaving them out. See Rule 12(h)(1)(B). Consider whether Goliath has waived his objection in this example. 

Question:
5. Among the missing. Goliath sues David for slander. David answers on the merits, denying that he made the offending statement. Six weeks later, he realizes that Goliath has filed suit in an improper venue. 
   
Answer Candidate:
David may file a motion for judgment on the pleadings, claiming that venue is improper, since he did not raise any of the four disfavored defenses in either a pre-answer motion or his answer.

Analysis:
The rule-makers saw this one coming, and averted it by the language in Rule 12(h)(1)(B). That subsection provides that the defense is waived if it is omitted from ''a responsive pleading or an amendment allowed by Rule 15(a)(1) as a matter of course.'' Under Rule 15(a)(1), an amendment as a matter of course (i.e., without leave of court) to a responsive pleading must be made within twenty-one days after serving it (if no further pleading is allowed). The clear implication of this reference is that an omitted defense may only be added by an amendment as of right. David's motion is too late, since the time for an amendment without leave of court has passed. Under Rule 12(h)(1)(B), he has waived his venue objection, and cannot revive it, even if the judge is willing to grant leave to amend to raise it.

Label:
FALSE

Introduction:
As the previous section explains, the scope of diversity jurisdiction in Article III and that conveyed to the federal district courts by Congress aren't the same. The Strawbridge rule illustrates one situation in which the statutory grant is narrower than the constitutional authority. Another example is the amount-in-controversy requirement. Article III, §2 contains no monetary restriction on diversity jurisdiction; it broadly authorizes jurisdiction over all cases between citizens of different states. Congress's grant of diversity jurisdiction to the federal district courts, however, includes an amount-in-controversy requirement, in order to keep small diversity cases out of federal court. See 28 U.S.C. §1332(a) (granting jurisdiction over diversity cases in which ''the amount in controversy exceeds the sum or value of $75,000, exclusive of interest or costs''). Here's a mediocre question, included to make a point. 

Question:
8. By the numbers. A diversity case cannot be heard in federal court unless the amount in controversy is at least 

Answer Candidate:
$75,000.01, exclusive of interest and costs.

Analysis:
This is what I would call a ''pure knowledge'’ multiple- choice question. It simply asks a factual question and gives you four choices, one of them right (D). To answer it, you simply need to remember a fact, not analyze a problem. This is easier, and less imaginative, than questions that require you to apply a concept to a subtle set of facts. Analytical questions call for a higher order of reasoning than this amount-in-controversy hypo. Most professors will test your analytical skills, since they are trying to teach you to think, not memorize. So I think you will find the more analytical examples in this book more representative of law school exams—either multiple-choice or essay exams. Similarly, the multi-state bar exam also focuses on proper application of legal rules to facts, not memorization of black letter law.

Label:
TRUE
"""
'''
     
        },
        {
        "role": "user",
        "content": cqa
        }
    ],
    temperature=0.7,
    max_tokens=512
  )

    print(json.dumps({'index': i, 'output': response.choices[0].message.content}))