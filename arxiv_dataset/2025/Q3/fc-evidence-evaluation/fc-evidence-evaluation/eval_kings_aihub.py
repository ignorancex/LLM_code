from kings_aihub_api import AIHub as AI

test_prompt = """
You will get as input a claim and evidence. 
Please verify the correctness of the claim following the following steps.
1. Break down the claim in independent facts. Each fact should be a separate sentence. 
2. Only break down the claim into facts, not the evidence!
3. Evaluate each fact individually using the given evidence only. Do not use additional sources or background knowledge. Explain the evaluation.
4. Finally summarise how many facts are (1.) supported by the evidence, (2.) clearly are contradicted by the evidence, (3.) how many facts you have in total.

Generate the output in form of a json as shown in the example below.
-----
Examples:

Claim: Mukesh Ambani, richest man in Asia had surgery for pancreatic cancer at Sloan Kettering, New York, US cancer speciality hospital on October 30, 2020.
Evidence: When was the photograph taken of Mukesh Ambani on the Facebook post claiming he had been diagnosed with pancreatic cancer and had undergone surgery? The photograph was taken on September 5, 2020. When was a video filmed of  Mukesh Ambani at the virtual launch of NK Singh's book Portrait of Power? The video was filmed on October 19, 2020. What date was the  Facebook post which confirmed Mukesh Ambani had lost 30 kgs, been diagnosed with pancreatic cancer and had had liver transplant surgery? The Facebook post was dated November 2, 2020. Where was Mukesh's photo of him supposedly recieving surgery actually taken? It was taken by Manushree Vijayvergiya who shared her experience of meeting Mukesh and Isha Ambani in a cafe in Liechtenstein.
Output: {{
            "facts": '1. Mukesh Ambani is the richest man in Asia. 2. Mukesh Ambani had surgery for pancreatic cancer. 3. The surgery took place at Sloan Kettering, a cancer specialty hospital in New York, US. 4. The surgery occurred on October 30, 2020.',
            "fact check": '1. Mukesh Ambani is the richest man in Asia. Not enough information given as the evidence does not mention anything about Ambani's wealth. 2. Mukesh Ambani had surgery for pancreatic cancer. Not enough information given as the evidence mentions a Facebook post but shortly after that he was seen at a launch event. 3. The surgery took place at Sloan Kettering, a cancer specialty hospital in New York, US. Not enough information given as the evidence does not mention anything about a hospital location. 4. The surgery occurred on October 30, 2020. The evidence shows other appearances by Ambani shortly before and after October 30, 2020. This contradicts with the fact that the surgery occurred on October 30, 2020.',            
            "support": 0, 
            "contradict": 1, 
            "facts count": 4 
        }}

Claim: Millions of jobs in the US were lost during Donald Trump's US presidency.
Evidence: How many people were in employment in 2017? 145,627,000 people as of January 2017. How many people were in employment in 2020? 141,735,000 people in September 2020. How many people in employment did the economy lose under Trump's presidency? The economy lost an estimate of 3,892,000 people in employment.
Output: {{
            "facts": '1. Donald Trump was US president. 2. Millions of jobs in the US were lost during his US presidency.',
            "fact check": '1. Donald Trump was US president. Supported, the evidence mentions Trump’s presidency indicating that he was president of the US. 2. Millions of jobs in the US were lost during his US presidency. The evidence supports this statement.',
            "support": 2, 
            "contradict": 0, 
            "facts count": 2 
        }}
-----
Input: 

Claim: The Nevada attorney general admits to changing signature verifications manually for over 200,000 votes.
Evidence: Question: 'Did the Nevada attorney general admit to manually changing signature verifications for over 200,000 votes?'
Answer: The claim that Nevada's attorney general admitted to changing signature verifications manually is FALSE. The statement regarding allegations that votes were changed by a signature verification machine was made by the former attorney general of Nevada, not by current Attorney General Aaron Ford.

Output:"""

ai = AI()

# One off inference.
print(ai.ask(test_prompt))

# Chat with history.
# print(ai.chat("What do cats eat?"))
# print(ai.chat("Why do they eat that?"))
# print(ai.chat("Do dogs like that too?"))

# Query an image.
# with open("/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/cat.jpg", "rb") as f:
#     print(ai.ask("What is this a picture of?", images=[f.read()]))