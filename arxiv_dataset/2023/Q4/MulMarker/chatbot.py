import os
import redis
import json
import openai
import config

openai.api_key = config.chatbotParm['openai.api_key']
system_prompt = config.Qprompt

r = redis.Redis(host='localhost', port=6379, db=0)

def chat(user_id, prompt):
    # Try to get past conversation history from Redis
    conversation_history = r.get(user_id)
    if conversation_history is None:
        conversation_history = [
            {"role": "system", "content": system_prompt},
        ]
    else:
        conversation_history = json.loads(conversation_history)

    # Add user's current message to conversation history
    conversation_history.append({"role": "user", "content": prompt})

    # Use conversation history with OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        temperature=0.2,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    ) 

    try:
        answer = response['choices'][0]['message']['content']
        # Add bot's current message to conversation history
        conversation_history.append({"role": "assistant", "content": answer})
        # Update conversation history in Redis, with an expiration time of one week
        r.set(user_id, json.dumps(conversation_history), ex=7*24*60*60)  # expires in 7 days
    except:
        answer = 'Oops... Please refresh the page...'

    return answer

