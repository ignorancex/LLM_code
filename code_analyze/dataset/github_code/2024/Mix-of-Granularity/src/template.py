from liquid import Template

general_cot_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_cot = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_medrag = Template('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

meditron_cot = Template('''
### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

meditron_medrag = Template('''
Here are the relevant documents:
{{context}}

### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

one_shot_context = """
Here is the question:
A lesion causing compression of the facial nerve at the stylomastoid foramen will cause ipsilateral

Here are the potential choices:
{'A': 'paralysis of the facial muscles.', 'B': 'paralysis of the facial muscles and loss of taste.', 'C': 'paralysis of the facial muscles, loss of taste and lacrimation.', 'D': 'paralysis of the facial muscles, loss of taste, lacrimation and decreased salivation.'}

Please think step-by-step and generate your output in json:\n"""

one_shot_response = """{"step_by_step_thinking": "The facial nerve, also known as cranial nerve VII, is responsible for controlling the muscles of facial expression, taste sensation in the anterior two-thirds of the tongue, and secretions from the salivary and lacrimal glands. Compression of the facial nerve at the stylomastoid foramen would primarily affect the motor function of the nerve, leading to paralysis of the facial muscles. Taste sensation and secretions from glands are also affected but are not typically the primary symptoms noticed. Loss of lacrimation (tearing) is less commonly associated with facial nerve compression at this location compared to other symptoms. Decreased salivation might be a consequence of the nerve compression, but it's not as prominent as the paralysis of facial muscles and loss of taste.", "answer_choice": "A"}"""

one_shot_rag_context = """Here are the relevant documents:\nDocument [0] (Title: Sinus Bradycardia (Nursing) -- Introduction) The sinoatrial node (SA) is the default pacemaker and therefore a crucial component of the heart's conduction system. It is located subepicardially and is crescent in shape.\u00a0The\u00a0sinoatrial node is innervated by vagus and sympathetic nerves. The sinoatrial nodal artery supplies blood to the sinoatrial node, it branches off the right coronary artery in 60% of cases, whereas in 40% of cases it comes off the\nDocument [1] (Title: Sinus Bradycardia -- Introduction) The sinoatrial node (SA) is the default pacemaker and, therefore, a crucial component of the heart's conduction system. It is located subepicardial and is crescent in shape. In an average adult, a sinoatrial node is 13.5 millimeters in length and is innervated by the vagus and sympathetic nerves. The sinoatrial nodal artery supplies blood to the sinoatrial node, it branches off the right coronary artery in 60% of cases, whereas in 40%\nDocument [2] (Title: Sinoatrial Nodal Reentrant Tachycardia -- Introduction) The sinoatrial node (SA) is crescent in shape and is located subepicardially. It is 13.5 millimeters in length and is innervated by the vagus and sympathetic nerves in an average adult. The sinoatrial nodal artery supplies blood to the sinoatrial node, it branches off the right coronary artery in 60% of cases, whereas in 40% of cases, it comes from the left circumflex coronary artery. The SA node is a crucial component of the\n\nHere is the question:\nWhere is the sinoatrial node located?\n\nHere are the potential choices:\nA. Between the left atrium and the left ventricle\nB. Between the right atrium and the right ventricle\nC. In the upper wall of the right atrium\nD. In the upper wall of the left ventricle\n\nPlease think step-by-step and generate your output in json:\n"""

one_shot_rag_response = """{\"step_by_step_thinking\": \"The sinoatrial node (SA) is the default pacemaker and, therefore, a crucial component of the heart's conduction system. It is located subepicardial and is crescent in shape. The sinoatrial nodal artery supplies blood to the sinoatrial node, it branches off the right coronary artery in 60% of cases, whereas in 40% of cases, it comes from the left circumflex coronary artery. The SA node is a crucial component of the heart's conduction system.\", \"answer_choice\": \"C\"}"""