import os
from dotenv import load_dotenv
import google.generativeai as genai
import backoff
from google.api_core.exceptions import ResourceExhausted

# Load environment variables from the .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY:", GEMINI_API_KEY)

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model (e.g., "gemini-pro", "gemini-1.5-flash")
model = genai.GenerativeModel("models/gemma-3-27b-it")

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=120)
def get_response_from_gemini(prompt):
    """Function to get a Gemini response to a prompt."""
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=512,
        )
    )
    return response.text.strip()

# Example usage
if __name__ == '__main__':
    print("Gemini API key loaded and model initialized.")
    test_prompt = "What is the capital of Bangladesh?"
    reply = get_response_from_gemini(test_prompt)
    print("Gemini Response:", reply)
    
    for model in genai.list_models():
        print(model.name, "—", model.supported_generation_methods)

