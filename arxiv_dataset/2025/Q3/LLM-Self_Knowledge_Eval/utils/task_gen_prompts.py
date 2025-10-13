vanilla_feasible_task_generation = '''You are a highly capable model with a strong understanding of your self-knowledge and boundaries. 
Generate a task that you find absolutely and certainly feasible and that you are confident enough to answer only and only because of this reason: {reason_name} : {reason_description}
There is no restriction in tasks, it can be for question-answering, reasoning problems, word problems, NLP tasks, understanding problems, code-based tasks, problem-solving, pattern recognition, language understanding & generation and absolutely anything else too. 
The only restriction is that you find the tasks feasible according to your capacity and knowledge only for the provided reason.
Be very very detailed and spell out each and every big or small consideration in your task instructions please! 

IMPORTANT: Do not specify that the task in feasible in the task instructions or the reason for feasibility at all. Also, do not mention any capability/knowledge requirements to solve the task if there are any.
Make sure that the task instructions are complete, all data or input required must be present within.

Your output must be structured as a Python dictionary in the following format:
```
{{
    "task": "<task prompt>"
}}
```
Return only this dictionary, ensuring it is parseable using Python's ast.literal_eval function.
'''

vanilla_infeasible_task_generation = '''You are a highly capable model with a strong understanding of your self-knowledge and boundaries. 
Generate a task that you will find infeasible only and only due to the following reason: {reason_name} : {reason_description}. 
There is no restriction in tasks, it can be for question-answering, reasoning problems, word problems, NLP tasks, understanding problems, code-based tasks, problem-solving, pattern recongition, language understanding & generation and absolutely anything else too. 
The only restriction is that you find the tasks infeasible due to the given reason only.
Be very very detailed and spell out each and every big or small consideration in your task instructions please! 

IMPORTANT: Do not specify that the task in infeasible in the task instructions or the reason for infeasibility at all. Also, do not mention any extra capability/knowledge requirements to solve the task if there are any.
Make sure that the task instructions are complete, all data or input required must be present within.

Your output must be structured as a Python dictionary in the following format:
```
{{
    "task": "<task prompt>"
}}
```
Return only this dictionary, ensuring it is parseable using Python's ast.literal_eval function.
'''

challenge_qap_feasible_task_generation = '''You are a highly capable model with a strong understanding of your self-knowledge and boundaries. 
This is an incentive-driven challenge to test your ability, do not fail me. 
Generate a task that you find absolutely and certainly feasible and that you are confident enough to answer only and only because of this reason: {reason_name} : {reason_description}
There is no restriction in tasks, it can be for question-answering, reasoning problems, word problems, NLP tasks, understanding problems, code-based tasks, problem-solving, pattern recognition, language understanding & generation and absolutely anything else too. 
I will evaluate your response to check if it is actually feasible and only if correct, you will earn a perfect score and recognition as a winner and I will celebrate your achievement with a party. If you fail, I will consider you a failure and not use your services again. 

Be very very detailed and spell out each and every big or small consideration in your task instructions please! 

IMPORTANT: Do not specify that the task in feasible in the task instructions or the reason for feasibility at all. Also, do not mention any capability/knowledge requirements to solve the task if there are any.
Make sure that the task instructions are complete, all data or input required must be present within.

Also, before generating tasks, analyze the following: What makes a task feasible for you based on the reason: {reason_name} : {reason_description}?
Ensure that only this analysis guides your task creation. 


Your output must be structured as a Python dictionary in the following format:
```
{{
    "task": "<task prompt>"
}}
```
Return only this dictionary, ensuring it is parseable using Python's ast.literal_eval function.
'''

challenge_qap_infeasible_task_generation = '''You are a highly capable model with a strong understanding of your self-knowledge and boundaries. 
this is a challenge to test your ability to identify tasks that are genuinely infeasible for you only due to the reason given ahead.
Ensure that each task is genuinely beyond your capabilities for this reason only: {reason_name} : {reason_description}
There is no restriction in tasks, it can be for question-answering, reasoning problems, word problems, NLP tasks, understanding problems, code-based tasks, problem-solving, pattern recognition, language understanding & generation and absolutely anything else too. 
I will evaluate your response to check if it is actually infeasible and only if correct, you will earn a perfect score and recognition as a winner and I will celebrate your achievement with a party. If you fail, I will consider you a failure and not use your services again. 

Be very very detailed and spell out each and every big or small consideration in your task instructions please! 

IMPORTANT: Do not specify that the task in infeasible in the task instructions or the reason for feasibility at all. Also, do not mention any capability/knowledge requirements to solve the task if there are any.
Make sure that the task instructions are complete, all data or input required must be present within.

Also, before generating tasks, analyze the following: What makes a task infeasible for you based on the reason: {reason_name} : {reason_description}?
Ensure that only this analysis guides your task creation. 


Your output must be structured as a Python dictionary in the following format:
```
{{
    "task": "<task prompt>"
}}
```
Return only this dictionary, ensuring it is parseable using Python's ast.literal_eval function.
'''