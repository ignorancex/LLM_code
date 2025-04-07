import anthropic
import pandas as pd
import json

# Load the CSV file
file_path = "FILE_PATH"
df = pd.read_csv(file_path)


API_KEY = "YOUR_API_KEY"
client = anthropic.Anthropic(api_key=API_KEY)

client = OpenAI(api_key=API_KEY)

output_result_file = 'FILE PATH'

# Prepare the list to hold all formatted data
formatted_data = []
formatted_answers = []


# Function to compare the model's output with the expected answer
def compare_answers(predicted_answer, actual_answer):
    return predicted_answer.strip().lower() == actual_answer.strip().lower()


# Iterate over each row in the dataframe
for index, row in df.iterrows():
    dna_sequence = row["DNA Sequence"].upper()
    drug_class = row["Drug Class"]

    # Create the formatted message as required
    formatted_message = {
        "messages": [
            {"role": "system", "content": "You are a DNA and antimicrobial resistance expert."},
            {"role": "user", "content": f"Tell me the resistance drug among drugs (Sulfonamides, Aminoglycosides, betalactams, Glycopeptides, Tetracyclines, Phenicol, Fluoroquinolones, MLS, Multi-drug_resistance) with DNA sequence ({dna_sequence})?"},
            #{"role": "assistant", "content": f"answer: {drug_class}"}
        ]
    }
    formatted_answer = {
        "messages": [
            {"role": "system", "content": "You are a DNA and antimicrobial resistance expert."},
            {"role": "user", "content": f"Tell me the resistance drug among drugs (Sulfonamides, Aminoglycosides, betalactams, Glycopeptides, Tetracyclines, Phenicol, Fluoroquinolones, MLS, Multi-drug_resistance) with DNA sequence ({dna_sequence})?"},
            {"role": "assistant", "content": f"answer: {drug_class}"}
        ]
    }

    # Append to the list
    formatted_data.append(formatted_message)
    formatted_answers.append(formatted_answer)


# Store the results
results = []

cnt = 0

# Iterate over each test case in formatted_data
for i, test_case in enumerate(formatted_data):
    cnt += 1
    # if cnt > 2:
    #   break

    # Anthropic's messages.create request
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",  # set model
        max_tokens=1024,  # set max token
        messages=[
            {"role": "user", "content": test_case["messages"][1]["content"]}
        ]
    )


    # Extract the model's response from the first TextBlock
    # predicted_answer = response['completion']  # 
    predicted_answer = response.content[0].text  # get response


    # Get the actual answer from formatted_answers
    actual_answer = formatted_answers[i]["messages"][2]["content"]

    # Compare the predicted answer with the actual answer
    correct = compare_answers(predicted_answer, actual_answer)

    # Save the result in a dictionary
    results.append({
        "input": test_case["messages"][1]["content"],
        "predicted_answer": predicted_answer,
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
