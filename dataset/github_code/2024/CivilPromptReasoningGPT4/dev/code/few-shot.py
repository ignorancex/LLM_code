# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 14:05:14 2023

@author: dansc
"""

# =============================================================================
# IMPORTS 
# =============================================================================
import openai
import os
from dotenv import load_dotenv
import json

import sys
sys.path.append('../../functions')
from load_data_dict import load_data_dict

data_dict = load_data_dict('../../data/dev.csv')

load_dotenv('../../data/.env')

# API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# for LangChain
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

def set_open_params(
    model = 'text-davinci-003',
    temperature = .7,
    max_tokens = 256,
    top_p = 1,
    frequency_penalty = 0,
    presence_penalty = 0,
):
    ''' Set Openai parameters '''
    
    openai_params = {}
    
    openai_params['model'] = model   
    openai_params['temperature'] = temperature
    openai_params['max_tokens'] = max_tokens
    openai_params['top_p'] = top_p
    openai_params['frequency_penalty'] = frequency_penalty
    openai_params['presence_penalty'] = presence_penalty
    return openai_params

def get_completion(params, prompt):
    
    ''' Get completion from aopenai api'''
    
    response = openai.Completion.create(
        engine = params['model'],
        prompt = prompt,
        temperature = params['temperature'],
        max_tokens = params['max_tokens'],
        top_p = params['top_p'],
        frequency_penalty = params['frequency_penalty'],
        presence_penalty = params['presence_penalty'],
        )
    return response

# I'm using response idx=9 and idx=665 as  examples

label_prediction_list = []
for idx, item in zip(data_dict['idx'], data_dict['cqa']):
    cqa = item
    prompt = f"""
Given an “Introduction” to the topic, a “Question” and an “Answer Candidate”, classify if the given Answer Candidate is true or false. Only return either “TRUE” or “FALSE”, do not return any other tokens

Text: “””
Introduction:
Section 1391(d), which defines the residence of a corporation that has contacts in one district within a state but not others, is confusing. Suppose, for example, that Omni-Plex Corporation has its principal place of business in the Northern District of California, sufficient contact to support general in personam jurisdiction over the corporation, that is, jurisdiction for a claim that arises anywhere. Assume further that Omni-Plex has no contacts in any other federal district within California. Under §1391(d), Omni-Plex ‘‘resides’’ in the Northern District of California, because, if the Northern District were a state, its contacts there would be sufficient to support personal jurisdiction over it there. But it would not ‘‘reside’’ in the Eastern District of California. It has no contacts there, so that, if the Eastern District were a state, it would not be subject to personal jurisdiction there. This is confusing because a defendant that is ‘‘at home’’ in a state (Daimler A.G. at 137) is subject to personal jurisdiction anywhere in the state, not just in the part of the state where the contacts exist. But §1391(d) tells us that, for venue purposes, we should look at the contacts in each district within the state separately. The corporation will be deemed to ‘‘reside’’ only in the districts where its contacts would support personal jurisdiction if that district were a state. Here’s a question to illustrate the operation of this vexing provision.
    
Question:
3. Manufacturing venue. Arthur wishes to bring a diversity action in federal court against Cleveland Manufacturing Company. Cleveland has its factory and principal place of business in the Northern District of Illinois, but no other contacts with Illinois. The claim is based on alleged negligence in making a toaster at the Illinois factory, which caused a fire in Arthur’s home in the Middle District of Georgia. 
    
Answer Candidate:
The Southern District of Illinois is not a proper venue under §1391 because no events giving rise to the claim took place there and Cleveland does not reside there under the venue statute. 

Label:
1

Introduction:
Venue in most federal actions is governed by 28 U.S.C. §1391(b), which provides: (b) Venue in general. A civil action may be brought in— (1) a judicial district in which any defendant resides, if all defendants are residents of the State in which the district is located; (2) a judicial district in which a substantial part of the events or omissions giving rise to the claim occurred, or a substantial part of property that is the subject of the action is situated; or (3) if there is no district in which an action may otherwise be brought as provided in this section, a judicial district in which any defendant is subject to the court’s personal jurisdiction with respect to such action. Note that subsections 1 and 2 are alternatives. Venue is proper in a district where either a defendant resides (if they are all residents of the state where the action is brought) or a district in which a substantial part of the events giving rise to the claim took place. Section 1391(b)(3) is a ‘‘fallback’’ venue provision that is only available in unusual circumstances: when there is no district, anywhere in the United States, where venue would be proper under subsection (b)(1) or (b)(2). If the defendants all reside in the state where suit is brought, or if a substantial part of the events giving rise to the claim occurred in some federal judicial district, or if property that is the subject of the action is found in a district, §1391(b)(3) cannot apply, because there will be at least one district in which venue is proper under §1391(b)(1) or (b)(2). Note also that there is a nasty proviso in 28 U.S.C. §1391(a) that poses a trap for the unwary: The first sentence of that subsection provides that the venue options in §1391(b) apply ‘‘except as otherwise provided by law.’’ Section 1391 is a general venue statute that applies unless there is a special venue statute for the type of claim the plaintiff brings. The United States Code has many specialized venue provisions for particular types of actions. See, e.g., 28 U.S.C. §1402 (tort action against the United States must be brought in the district where the plaintiff resides or wherein the act or omission complained of occurred). If there is a special venue statute for a particular type of claim, it may be interpreted as displacing the general venue choices in §1391(b)—that is, as providing alternative, exclusive venue choices for such claims—or as providing additional venue choices for those claims, along with those in §1391(b). See generally, Wright & Miller §3803. To sort out the basics of §1391(b), try the following question. Assume for all examples in the chapter that no special venue statute applies. In answering these questions, please do refer to the venue statutes. Once again, my questions are based on the assumption that you have the Federal Rules book available, so you can refer to relevant statutes and rules, rather than memorizing them. I want to test my students’ ability to apply provisions like the venue statutes or Federal Rules, rather than their memory of them. Needless to say, you should find out your professor’s policy on bringing materials into the exam before you prepare for it. If she doesn’t allow you to refer to the rules book, you obviously need to spend more time memorizing statutes and rules. 

Question:
1. Redistricting. Dziezek, who resides in the Southern District of Indiana, sues Torruella and Hopkins. Torruella resides in the Western District of Kentucky. Hopkins resides in the Western District of Tennessee. Dziezek sues them both for damages arising out of a business deal for the financing of a subdivision Dziezek planned to build in the Southern District of Ohio. His claim against Torruella is for fraud, his claim against Hopkins is for fraud and for violation of the Federal Truth in Lending Act. The negotiations between the parties for the financing took place in the Western District of Tennessee. Dziezek claimed that, after the defendants had provided the first installment of financing for the project, and he had commenced construction, they refused to provide subsequent payments to the contractor, who consequently did not complete the project. Venue in Dziezek’s action would be proper in 

Answer Candidate:
the Southern District of Indiana. 

Label:
0 



{cqa}
“””

    """
    
    params = set_open_params(temperature=0)
    response = get_completion(params, prompt)
    print(json.dumps({'idx': idx, 'output': response.choices[0].message.content}))
    true_false_list = response['choices'][0]['text'][1]
    label_prediction_list = []
    for item in true_false_list:
        if item.upper() == "TRUE":
            label_prediction_list.append(1)
        if item.upper() == "FALSE":
            label_prediction_list.append(0)
            
data_dict['prediction'] = label_prediction_list


