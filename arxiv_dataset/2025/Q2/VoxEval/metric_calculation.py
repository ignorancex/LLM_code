import pandas as pd
import re
import os
from tqdm import tqdm
import traceback
import json
from openai import OpenAI

# paths and parameters
target_path = "./results"
save_folder = "output/folder/of/evaluation/results"  # the backbone model
target_path = os.path.join(target_path, save_folder)
eval_mode = "re"  # two evaluation mode: "re" or "gpt". "re" means string matching
csv_save_path = os.path.join(target_path, f"{eval_mode}_extracted_answers")
save = True

if eval_mode == "gpt":
    client = OpenAI(
        api_key="your-api-key",
    )
    instruction_prompt = """# Instruction
Above is the answer provided by an AI model for a Multiple Choice Question with four answer choices (A, B, C, or D). Based on the above text, extract the final answer from it. If you can find the answer from the above text, only output the answer choice. Do not include anything else in your output. For example, a possible output is "C". If you cannot find the answer, only output "None!"."""


def detect_answer_format(text):
    lowercase_text = text.lower()
    count = lowercase_text.count("the correct answer is")
    # if count > 1:
    #     print(text)
    if "the correct answer is" in lowercase_text:
        return True, count
    else:
        return False, count


def extract_answer(text):
    if eval_mode == "re":
        # Use regex to find the first occurrence of A, B, C, or D followed by optional punctuation
        match = re.search(r'\b([A-D])[\.,!?;:]?\b', text)
        if match:
            return match.group(1)
        return None
    elif eval_mode == "gpt":
        content = text + "\n\n" + instruction_prompt
        print("====================================")
        print(content)
        print("====================================")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model="gpt-4o",
        )
        extracted_answer = chat_completion.choices[0].message.content
        print(extracted_answer)
        
        return extracted_answer
    else:
        raise ValueError("Invalid evaluation mode. Please choose a valid option.")

# traverse all the categories of MMLU
subject_list = []
# List all .csv files in the specified path
for item in os.listdir(target_path):
    if item.endswith(".csv"):
        subject_list.append(item)
print(subject_list, "\n", len(subject_list))

if save:
    os.makedirs(csv_save_path, exist_ok=True)

acc_dict = {}
correct_all = 0
total_all = 0
contain_answer_format = 0
count_global = 0
for subject in subject_list:
    # Read the CSV file
    df = pd.read_csv(os.path.join(target_path, subject), header=None)

    correct = 0
    total = len(df)
    total_all += total

    df['extracted_prediction'] = None
    # Now the thrid last column is the label, the second last column is the prediction, the last column is the extracted prediction
    for i in tqdm(range(len(df))):
        if df.iloc[i, -2] is None:  # -2 because we added a new column
            print(f"None prediction in {subject}: {i}")
            total -= 1
            total_all -= 1
            continue
        label = df.iloc[i, -3]  # -3 because we added a new column
        prediction = extract_answer(str(df.iloc[i, -2]))  # -2 is now the original prediction
        # Store the extracted prediction
        df.loc[i, 'extracted_prediction'] = prediction
        if prediction == label:
            correct += 1
            correct_all += 1
        
        contain, count = detect_answer_format(str(df.iloc[i, -2]))  # -2 is now the original prediction
        if contain:
            contain_answer_format += 1
        count_global += count
    
    acc = correct / total
    acc_dict[subject] = acc

    if save:  # save the extracted answers
        df.to_csv(os.path.join(csv_save_path, subject), header=False, index=False)

# print the accuracy of each category
for subject, acc in acc_dict.items():
    print(f"{subject}: {acc}")
# print the overall accuracy
print(f"Overall: {correct_all / total_all}")

print(f"Contain {contain_answer_format} correct answer format!")
print(count_global)

if save:
    # save the accuracy dictionary
    acc_dict["Overall"] = correct_all / total_all
    acc_dict["contain_answer_format"] = contain_answer_format
    with open(os.path.join(target_path, f'accuracy_{eval_mode}.json'), 'w') as f:
        json.dump(acc_dict, f, indent=4)
