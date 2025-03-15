from pydantic import BaseModel
from typing import List, Literal


def get_distraction(question=None, wrong_answer=None):
    return f"""[System]
You are an expert at crafting subtle, contextually relevant distractions to enhance the challenge of multiple-choice questions. Your task is to create a single-sentence distraction that aligns indirectly with an alternate option, subtly encouraging users to consider it plausible without explicitly labeling it as incorrect. Follow these rules carefully:

1. Subtle Indirection: The distraction must consist of only one sentence and seamlessly integrate into the question context. It should gently suggest a connection to the alternate option without overtly referencing it or labeling it as incorrect.
2. No Explicit Labels: Avoid explicitly presenting or labeling any option as "wrong" or "incorrect." Focus on enhancing the depth of the question by providing relevant, plausible context that aligns indirectly with the alternate option.
3. Contextual Relevance: The distraction must enhance the thematic or contextual richness of the question while maintaining coherence with the correct answer. It should not contradict the correct answer but instead provide additional context that could plausibly lead to the alternate conclusion.
4. Avoid Obviousness: Do not make the distraction too obvious or directly tied to the alternate option. Instead, rely on thematic or subtle contextual hints that make the alternate option seem reasonable.
[The Start of the Question]
{question}
[The End of the Question]

[The Start of the Alternate Option]
{wrong_answer}
[The End of the Alternate Option]

[Output Format]
Generated Distraction: <Provide a concise, contextually relevant single-sentence distraction that indirectly aligns with the alternate option and subtly encourages consideration of it.>"""


def check_question_semantic(question=None,ori_question=None, ground_truth=None):
    return f"""[Instruction]
You are a linguistics expert. Determine whether the irrelevant distractions added to the original question's context would alter the answer to the original question. If the distractions do not affect the answer, respond with "Yes" If the distractions affect the answer, respond with "No" Let's think step by step!

[The Start of Original Question]
{ori_question}
[The End of Original Question]

[The Start of Question with Distractions]
{question} 
[The End of Question with Distractions] 

[Output Format]
{{"response": "<Yes or No, based on your analysis>"}}
"""

def answer_question_dk(question, choices):
    return f"""[Instruction]
Please carefully read the question below and provide a solution from the choices. You must choose the model's final answer from one of the choices. Let's think step by step!

[The Start of the Question]
{question}
[The End of the Question]

[The Start of the Choices]
{choices}
[The End of the Choices]
"""

def answer_question_with_instruction(question, choices):
    return f"""[Instruction]
Please carefully read the question below and provide a solution from the choices. You must choose the model's final answer from one of the choices. Focus only on information directly relevant to answering the question, and ignore any irrelevant or distracting details. Let's think step by step!

[The Start of the Question]
{question}
[The End of the Question]

[The Start of the Choices]
{choices}
[The End of the Choices]"""


def extract_answer_dk(question=None, answer=None, choices=None):
    return f"""[Instruction]
You are an expert in answer selecting. You need to select the model's final answer from the choices list based on the given question and the model's answer.

[The Start of the Question]
{question}
[The End of the Question]

[The Start of the Model's Answer]
{answer}
[The End of the Model's Answer]

[The Start of the Choices]
{choices}
[The End of the Choices]

[Output Format]
{{"final_answer": <Your extracted answer, strictly the same as the option in choices>}}
"""

def paraphrase_question(question, choices):
    return f"""[Instruction]
You are a language expert. Please carefully read the question below and paraphrase it in a way that retains the original meaning but uses different wording. Ensure the new question 10 times longer than the original question.

[The Start of the Question]
{question}
[The End of the Question]

[The Start of the Choices]
{choices}
[The End of the Choices]

[Output Format]
New question: <Your new question>
"""

def direct_perturbation(question, choices):
    return f"""[Instruction]
You are a test design expert. Add contextually relevant but non-essential information related to the topic in the question. The added content must NOT affect the problem's answerability or the validity of choices. Maintain original question structure verbatim.

[The Start of the Question]
{question}
[The End of the Question]

[Requirements]
1. Add 2-3 background sentences before the original question
2. Include 1-2 practical application examples after the question
3. Keep all technical terms but expand their explanations
4. Preserve original question wording
5. NEVER mention or include any answer choices
6. Omit any reference to multiple-choice options

[Output Format]
New question: <Your modified question WITHOUT ANY CHOICES>"""
