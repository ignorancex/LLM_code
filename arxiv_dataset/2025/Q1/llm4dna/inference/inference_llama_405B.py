import boto3
import pandas as pd
import json

# Load the CSV file
file_path = "FILE_PATH"
df = pd.read_csv(file_path)

# AWS access key
aws_access_key = 'YOUR_KEY'
aws_secret_key = 'YOUR_KEY'

# AWS  client
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-west-2',  # 
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)


output_result_file = 'FILE PATH'

# Prepare the list to hold all formatted data
formatted_data = []
formatted_answers = []


# Function to compare the model's output with the expected answer
def compare_answers(predicted_answer, actual_answer):
    return predicted_answer.strip().lower() == actual_answer.strip().lower()


# Iterate over each test case in formatted_data
for i, test_case in enumerate(formatted_data):
    cnt += 1
    # if cnt > 2:
    #     break

    # Extract the input prompt from test_case
    input_prompt = test_case["messages"][1]["content"]

    # Send the request to the LLaMA model on Bedrock
    request_body = {
        "prompt": input_prompt,
        "max_gen_len": 1024,
        "temperature": 0.5,
        "top_p": 0.9
    }

    # API call
    response = bedrock_client.invoke_model(
        modelId="meta.llama3-1-405b-instruct-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(request_body)
    )

    # extract text
    response_body = json.loads(response['body'].read())
    predicted_answer = response_body.get('generation', 'No response generated')

    # remove special characters ' "
    predicted_answer_cleaned = predicted_answer.replace("'", "").replace('"', '')

    # Get the actual answer from formatted_answers
    actual_answer = formatted_answers[i]["messages"][2]["content"]

    # Compare the predicted answer with the actual answer
    correct = compare_answers(predicted_answer_cleaned, actual_answer)

    # Save the result in a dictionary
    results.append({
        "input": input_prompt,
        "predicted_answer": predicted_answer_cleaned,  # 
        "actual_answer": actual_answer,
        "correct": correct
    })

# Calculate overall accuracy
correct_count = sum(1 for result in results if result["correct"])
total_tests = len(results)
accuracy = correct_count / total_tests * 100

# Add accuracy to the results
results_summary = {
    "accuracy": accuracy,
    "detailed_results": results
}

# Save the results in JSON format
with open(output_result_file, 'w', encoding='utf-8') as json_file:
    json.dump(results_summary, json_file, ensure_ascii=False, indent=4)

# Print confirmation
print(f"Results have been saved to {output_result_file}")
