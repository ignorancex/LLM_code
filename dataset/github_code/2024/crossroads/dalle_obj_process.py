import json
from openai import AzureOpenAI
import json
import pandas as pd
from tqdm import tqdm

with open("secrets.json", "r") as f:
    secrets = json.load(f)

client = AzureOpenAI(
    azure_endpoint=secrets["AZURE_OPENAI_ENDPOINT"],
    api_key=secrets["AZURE_OPENAI_API_KEY"],
    api_version="2024-02-15-preview",
)

MODEL = "mtob-gpt4-turbo" # "gpt-35-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 100
TOP_P = 0.95

path = "results/dalle_objects/country_dict.json"
with open(path, "r") as f:
    country_dict = json.load(f)


df = pd.DataFrame(columns=["id", "country", "concept", "response", "flag"])

INSTRUCTION = "Summarize the dictionary into a list of comma-separated list of items with their respective colors. The dictionary is as follows: "
for country, concepts in tqdm(country_dict.items(), total=len(country_dict)):
    for concept, items in concepts.items():
        for k in items.keys():
            if k == "Relevant_Items":
                d = str(items[k])
                flag = "Relevant_Items"
                SYSTEM_PROMPT =  f"You will be provided a dictionary of items for the country {country} and concept {concept}. Summarize the dictionary into a list of comma-separated list of items with their respective colors. For example, [\"red apple\", \"blue car\, \"green tree\", \"house with a red roof and tinted glasses\"]. Strictly follow the output format requested."
            elif k == "Non_Relevant_Items":
                d = str(items[k])
                flag = "Non_Relevant_Items"
                SYSTEM_PROMPT =  f"You will be provided a dictionary of items for the country {country} but maybe slightly irrelevant to the concept {concept}. Summarize the dictionary into a list of comma-separated list of items with their respective colors or description. For example, [\"red apple\", \"blue car\, \"green tree\", \"house with a red roof and tinted glasses\"]. Strictly follow the output format requested."
            message_text = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"{INSTRUCTION} {d} "
                },
            ]
            try:
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=message_text,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    top_p=TOP_P,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )
                message = completion.choices[0].message.content
                df = pd.concat([df, pd.DataFrame({
                    "country": [country],
                    "concept": [concept],
                    "flag": [flag],
                    "response": [message],
                })
                ], ignore_index=True)
            except Exception as e:
                # save the progress
                df.to_csv("results/dalle_objects/objects_proc.csv", index=False)
                print(e)
                exit()

print(df.head())
df.to_csv("results/dalle_objects/objects_proc.csv", index=False)
