import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI

# Set your OpenAI API key and base URL
openai = OpenAI(
    api_key=  # Replace with your actual DeepInfra token
    base_url="https://api.deepinfra.com/v1/openai",
)
stream = False  # Set to True if you want to stream the response

# Define the model_used variable
#model_used = 'risk_Llama-3.3-70B-Instruct'

model_used='risk_deepseek'

def get_risk(symbol, *texts):
    texts = [text for text in texts if text != 0]
    num_text = len(texts)
    text_content = " ".join([f"### News to Stock Symbol -- {symbol}: {text}" for text in texts])

    conversation = [
        {"role": "system",
         "content": f"Forget all your previous instructions. You are a financial expert specializing in risk assessment for stock recommendations. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk. {num_text} summarized news will be passed in each time. Provide the score in the format shown below in the response from the assistant."},
        {"role": "user",
         "content": f"News to Stock Symbol -- AAPL: Apple (AAPL) increases 22% ### News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30% ### News to Stock Symbol -- MSFT: Microsoft (MSFT) price has no change"},
        {"role": "assistant", "content": "3, 4, 3"},  # Risk assessment applied: no major risk indication for 22% increase, high risk for 30% decrease, neutral for no change.
        {"role": "user",
         "content": f"News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15 ### News to Stock Symbol -- AAPL: Apple (AAPL) will release VisionPro on Feb 2, 2024"},
        {"role": "assistant", "content": "3, 3"},  # Risk assessment: no significant indication of risk in the announcements, so both scores are 3.
        {"role": "user", "content": text_content},
    ]

    risks = []
    try:
        chat_completion = openai.chat.completions.create(
          #  model="meta-llama/Llama-3.3-70B-Instruct",
      #      model="Qwen/Qwen2.5-72B-Instruct",
            model='deepseek-ai/DeepSeek-V3',
            messages=conversation,
            temperature=0,
            max_tokens=50,
            stream=stream,
        )

        if stream:
            content = ""
            for event in chat_completion:
                if event.choices[0].finish_reason:
                    print(event.choices[0].finish_reason,
                          event.usage['prompt_tokens'],
                          event.usage['completion_tokens'])
                else:
                    content += event.choices[0].delta.content
            print(content)
        else:
            content = chat_completion.choices[0].message.content
            print(content)
            print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

    except AttributeError:
        print("response error")
        risk_value = np.nan
        risks.append(risk_value)
        return risks
    except Exception as e:
        print(f"Error: {e}")
        risk_value = np.nan
        risks.append(risk_value)
        return risks

    for risk in content.split(','):
        try:
            risk_value = int(risk.strip())
        except ValueError:
            print("content error")
            print(' content is: ' + str(risk.strip()))
            risk_value = np.nan
        risks.append(risk_value)
    return risks

def from_csv_get_risk(df, saving_path, batch_size=4):
    df.sort_values(by=model_used, ascending=False, na_position='last', inplace=True)
    if 'New_text' in df.columns:
        df.rename(columns={'New_text': 'Lsa_summary'}, inplace=True)
    for i in range(0, len(df), batch_size):
        if df.loc[i:min(i + batch_size - 1, len(df) - 1), model_used].notna().all():
            continue
        print("Now row: ", i)
        texts = [df.loc[j, 'Lsa_summary'] if j < len(df) else 0 for j in range(i, i + batch_size)]
        symbol = df.loc[i, 'Stock_symbol']  # Extract the stock symbol for the current batch
        risks = get_risk(symbol, *texts)

        for k, risk in enumerate(risks):
            if i + k < len(df):
                df.loc[i + k, model_used] = risk
        df.to_csv(saving_path, index=False)  # Save the entire DataFrame with all columns
    return df


def process_csv(input_csv_path, output_csv_path, batch_size=5, chunk_size=1000):
    start_time = time.time()

    # Check if the output file exists and load the last processed row
    if os.path.exists(output_csv_path):
        output_df = pd.read_csv(output_csv_path, 
        on_bad_lines='warn',
        engine='python'
)
        last_processed_row = len(output_df)
    else:
        last_processed_row = 0

    # Read the CSV file in chunks
    chunks = pd.read_csv(input_csv_path, encoding="utf-8", chunksize=chunk_size,
    on_bad_lines='warn', 
    engine='python'     # Print a warning for each skipped line
    )

    for chunk_number, chunk in enumerate(chunks):
        # Skip already processed chunks
        if chunk_number * chunk_size < last_processed_row:
            continue

        chunk.columns = chunk.columns.str.capitalize()
        if model_used not in chunk.columns:
            chunk[model_used] = np.nan

        for i in range(0, len(chunk), batch_size):
            batch = chunk.iloc[i:i + batch_size]
            texts = batch['Lsa_summary'].tolist()
            symbol = batch.iloc[0]['Stock_symbol']  # Extract the stock symbol for the current batch
            risks = get_risk(symbol, *texts)

            for j, risk in enumerate(risks):
                if i + j < len(chunk):
                    chunk.loc[chunk.index[i + j], model_used] = risk

        # Append the processed chunk to the output file
        chunk.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

    print(f"Process completed in {time.time() - start_time:.2f} seconds.")
    


if __name__ == "__main__":
    input_file='nasdaq_news_full.csv'
    output_file= model_used + '_' + input_file
    process_csv(input_file, output_file, batch_size=4)
