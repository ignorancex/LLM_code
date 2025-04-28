from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import pandas as pd
import numpy as np

import os
from langchain_openai import ChatOpenAI
from prompt import Parse_user_input, Parse_user_input2, Summary_output, Determine_model, Analyzing_data

# Load the model
inference_server_url = "your url"
api_key = "your key"
model_name = "your model name"

model = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key,
    openai_api_base=inference_server_url,
    temperature=0.8,
)

# Define the conversation context
while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    uploaded_file_path = 'your file path'
    file_content = pd.read_csv(uploaded_file_path) if uploaded_file_path.endswith('.csv') else pd.read_excel(uploaded_file_path)
    if "AutoML" in user_input:
        user_input = Parse_user_input(user_input)
    response = model.generate_response(user_input)

    config = 
    print("model: " + response)