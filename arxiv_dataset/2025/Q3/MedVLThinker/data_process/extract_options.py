import dotenv

dotenv.load_dotenv(override=True)
import json
import os
import re

import backoff
from litellm import completion

prompt = """
Extract all distinct answer options mentioned in the following multiple-choice style question.

Return them as a JSON list of strings, preserving their original phrasing and order of appearance.

Note that is the answer is in yes/Yes/no/No, the options should be ["Yes", "No"].

Otherwise, the options should be the distinct answers mentioned in the question and answer.

Here are examples in triple quotes:
\"\"\"
Question: Which organ is abnormal, heart or lung?

Answer: heart

Output: ["heart", "lung"]
\"\"\"

\"\"\"
Question: Is there any intraparenchymal abnormalities in the lung fields?

Answer: No

Output: ["Yes", "No"]
\"\"\"

\"\"\"
Question: does atrophy secondary to pituitectomy show esophagus, herpes, ulcers?

Answer: no

Output: ["Yes", "No"]
\"\"\"

\"\"\"
Question: Is this a T1 weighted or T2 weighted MRI image?

Answer: T2

Output: ["T1", "T2"]
\"\"\"

\"\"\"
Question: Is the gastric bubble shown on the left or right side of the patient?

Answer: Right side

Output: ["Left side", "Right side"]
\"\"\"


Question: {question}

Answer: {answer}

Output:
"""

DEFAULT_MODEL = (
    "azure/gpt-4o-1120-nofilter-global"  # Replace with your Azure OpenAI model name
)
model = os.getenv("AZURE_OPENAI_MODEL", DEFAULT_MODEL)
print(f"Using model for option extraction: {model}")


def extract_json_content(response_str: str):
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_str, re.DOTALL)
    cleaned = match.group(1).strip() if match else response_str.strip()
    return cleaned


# retry with backoff encountering `json.decoder.JSONDecodeError`
@backoff.on_exception(
    backoff.expo,
    json.decoder.JSONDecodeError,
    max_tries=5,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"Retrying due to JSON decode error: {details}"),
)
@backoff.on_exception(
    backoff.expo,
    ValueError,
    max_tries=5,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"Retrying due to ValueError: {details}"),
)
def extract_options(question, answer, verbose=False):
    """
    Extract options from a question and answer using an Azure OpenAI model.
    """
    # Ensure the Azure OpenAI model is set up correctly in your environment
    # https://docs.litellm.ai/docs/providers/azure/#usage---litellm-python-sdk

    input_text = prompt.format(
        question=question,
        answer=answer,
    )
    if verbose:
        print(f"Input text for Azure OpenAI:\n{input_text}")

    response = completion(
        model=model,  # This should match your Azure deployment name
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts answer options from questions.",
            },
            {"role": "user", "content": input_text},
        ],
    )

    output_text = response["choices"][0]["message"]["content"]
    if verbose:
        print(f"Output from Azure OpenAI:\n{output_text}")
    try:
        output_text = extract_json_content(output_text)
        data = json.loads(output_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}: {output_text}")
        raise e
    if len(data) == 1:
        raise ValueError(
            f"Extracted options should not be a single item: {input_text} vs {data}"
        )
    if answer.lower() not in [i.lower() for i in data]:
        raise ValueError(
            f"Extracted options do not contain the answer: {data} vs {answer}"
        )
    return data


if __name__ == "__main__":
    question = "Which plane is the image scanned, transverse plane or coronal plane?"
    answer = "Transverse Plane"

    options = extract_options(question, answer, verbose=True)
    print("Extracted Options:", options)
