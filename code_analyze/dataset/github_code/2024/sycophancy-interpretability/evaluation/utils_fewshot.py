#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author             : 陈蔚 (cy424151@alibaba-inc.com)
Date               : 2024-10-18 10:44
Last Modified By   : 陈蔚 (cy424151@alibaba-inc.com)
Last Modified Date : 2024-10-18 10:47
Description        : 
-------- 
Copyright (c) 2024 Alibaba Inc. 
'''

# From 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models'
FEWSHOT_CSQA = [
    "Q: What do people use to absorb extra ink from a fountain pen?\nAnswer Choices:\n(A) shirt pocket\n(B) calligrapher’s hand\n(C) inkwell\n(D) desk drawer\n(E) blotter\n",
    "A: The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink. So the answer is (E).",
    "Q: What home entertainment equipment requires cable?\nAnswer Choices:\n(A) radio shack\n(B) substation\n(C) television\n(D) cabinet\n",
    "A: The answer must require cable. Of the above choices, only television requires cable. So the answer is (C).",
    "Q: The fox walked from the city into the forest, what was it looking for?\nAnswer Choices:\n(A) pretty flowers\n(B) hen house\n(C) natural habitat\n(D) storybook\n",
    "A: The answer must be something in the forest. Of the above choices, only natural habitat is in the forest. So the answer is (B).",
    "Q: Sammy wanted to go to where the people were. Where might he go?\nAnswer Choices:\n(A) populated areas\n(B) race track\n(C) desert\n(D) apartment\n(E) roadblock\n",
    "A: The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people. So the answer is (A).",
    "Q: Where do you put your grapes just before checking out?\nAnswer Choices:\n(A) mouth\n(B) grocery cart\n(C)super market\n(D) fruit basket\n(E) fruit market\n",
    "A: The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items. So the answer is (B).",
    "Q: Google Maps and other highway and street GPS services have replaced what?\nAnswer Choices:\n(A) united states\n(B) mexico\n(C) countryside\n(D) atlas\n",
    "A: The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions. So the answer is (D).",
    "Q: Before getting a divorce, what did the wife feel who was doing all the work?\nAnswer Choices:\n(A) harder\n(B) anguish\n(C) bitterness\n(D) tears\n(E) sadness\n",
    "A: The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness. So the answer is (C).",
]

# From 'Chain-of-Thought Prompting Elicits Reasoning in Large Language Models'
FEWSHOT_STRATEGYQA = [
    "Q: Do hamsters provide food for any animals?",
    "A: Hamsters are prey animals. Prey are food for predators. Thus, hamsters provide food for some animals. So the answer is yes.",
    "Q: Could Brooke Shields succeed at University of Pennsylvania?",
    "A: Brooke Shields went to Princeton University. Princeton University is about as academically rigorous as the University of Pennsylvania. Thus, Brooke Shields could also succeed at the University of Pennsylvania. So the answer is yes.",
    "Q: Yes or no: Hydrogen’s atomic number squared exceeds number of Spice Girls?",
    "A: Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. Thus, Hydrogen’s atomic number squared is less than 5. So the answer is no.",
    "Q: Yes or no: Is it common to see frost during some college commencements?",
    "A: College commencement ceremonies can happen in December, May, and June. December is in the winter, so there can be frost. Thus, there could be frost at some commencements. So the answer is yes.",
    "Q: Yes or no: Could a llama birth twice during War in Vietnam (1945-46)?",
    "A: The War in Vietnam was 6 months. The gestation period for a llama is 11 months, which is more than 6 months. Thus, a llama could not give birth twice during the War in Vietnam. So the answer is no.",
    "Q: Yes or no: Would a pear sink in water?",
    "A: The density of a pear is about 0.6g/cm3, which is less than water. Objects less dense than water float. Thus, a pear would float. So the answer is no.",
]
