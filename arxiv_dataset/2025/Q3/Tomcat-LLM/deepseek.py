import os
import backoff
from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env file")

# Set up OpenAI-compatible client for DeepSeek
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

@backoff.on_exception(backoff.constant, RateLimitError, interval=30, max_tries=5)
def get_completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def get_response_R1(prompt):
    """Function to interact with DeepSeek-Reasoner and generate a response."""
    
    messages = [
        {"role": "system", "content": (
            "You are an intelligent assistant helping a human in a collaborative game to collect a gem the human desires. "
            "Your task is to interpret the instruction provided by the human and generate an appropriate response enabling the human retrieve their desired gem."
        )},
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        max_tokens=512,
        temperature=0.2,
        stream=False,  # Set to True if you want streaming responses
        # top_p=0.9  # Optional
    )

    return response.choices[0].message.content.strip()

# Main test script
if __name__ == '__main__':
    if DEEPSEEK_API_KEY:
        print("DeepSeek Reasoner API key loaded successfully.")
        
        test_prompt = "What model are you using"
        reply = get_response_R1(test_prompt)
        print("DeepSeek Reasoner Response:", reply)
    else:
        print("API key not found. Please check your .env file.")
    
    # List models
    models = client.models.list()

    # Print model IDs
    print("Available models:")
    for model in models.data:
        print("-", model.id)
