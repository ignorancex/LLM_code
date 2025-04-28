import os
import math
import openai
import config

openai.api_key = config.chatbotParm['openai.api_key']
ASprompt = config.ASprompt
AFprompt = config.AFprompt

def is_not_missing(val):
    return val is not None and val != "" and not (isinstance(val, float) and math.isnan(val))

def resChat(Sysmessage, prompt):

    messages = []
    messages.append({"role": "system", "content": Sysmessage})

    question = {}
    question['role'] = 'user'
    question['content'] = prompt
    messages.append(question)

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      messages=messages,
      temperature=1,
      max_tokens=2048,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
	)

    try:
        answer = response['choices'][0]['message']['content'].replace("\n", "<br>").lstrip()
    except:
        answer = 'Oops... Report generation failed.'

    return answer

def choose_sysmessage(prompt):

     if not all(is_not_missing(val) for val in prompt.values()):
        answer = "MulMarker doesn't work properly"

     else:
          train_pval = float(prompt['train_pVal'])
          test_pval = float(prompt['test_pVal'])
          total_pval = float(prompt['total_pVal'])

          prompt = str(prompt)

          if (train_pval < 0.05) and (test_pval < 0.05) and (total_pval < 0.05):
               answer = resChat(ASprompt, prompt)
          else:
               answer = resChat(AFprompt, prompt)

     return answer
