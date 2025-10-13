import pandas as pd
from transformers import pipeline
from groq import Groq
from langchain_community.llms import Ollama
import json

def call_model(prompt, target_model):
    """
    Call a target model using the provided prompt.
    """
    llm = Ollama(model=target_model)
    prediction = llm.invoke([prompt])
    return prediction

def call_llama(prompt, groq_key, model="llama3-70b-8192"):
    """
    Call the bigger Llama model using Groq.
    """
    # Set your Groq API key
    GROQ_API_KEY = groq_key
    
    client = Groq(api_key=GROQ_API_KEY) 
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}\n\n"
            }
        ],
        model=model,
    )
    result = chat_completion.choices[0].message.content
    return result


def get_prediction(context, question, target_model, groq_key, include_context=True):
    """
    Get the prediction from the target model.
    The prompt is constructed using the context and question.
    """
    if include_context and context:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer the above question in one word, and also give the explanation for your answer. Don't add anything else to your answer."
    else:
        prompt = f"Question: {question}\nAnswer the above question in one word, and also give the explanation for your answer. Don't add anything else to your answer."
    
    # Call the target model
    prediction_raw = call_model(prompt, target_model)
    
    # Now pass prediction to the bigger llama model for extraction
    extraction_prompt = (
        f"Prediction: {prediction_raw}\n"
        "The above is the predicted output for a model. Extract this into two parts: 'label' and 'explanation'. "
        "Return the result as a valid JSON object with keys 'label' and 'explanation'. Do add anything else to your answer."
    )
        
    extraction = call_llama(extraction_prompt, groq_key)
    
    # Extraction returns text in the format:
    # {Label:
    # Explanation:}
    try:
        result = json.loads(extraction)
        label = result.get("label", "unknown")
        explanation = result.get("explanation", "")
    except Exception as e:
        label = "unknown"
        explanation = extraction.strip()
    
    return label, explanation