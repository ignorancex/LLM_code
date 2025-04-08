import argparse
from src.tool import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', nargs='+', default=[])
    
    parser.add_argument('--dataset_type', type=str,
                        choices=['math', 'date'], required=True)
    parser.add_argument('--dataset_name', type=str, default="gsm8k")
    args = parser.parse_args()

    input_path = args.input_path
    dataset_type = args.dataset_type
    dataset_name = args.dataset_name
    
    if dataset_name == "gsm8k":
        dataset = jsonlines_load("dataset/gsm8K_test.jsonl")
    elif dataset_name == "svamp":
        dataset = jsonlines_load("dataset/svamp.jsonl")
    elif dataset_name == "asdiv":
        dataset = jsonlines_load("dataset/asdiv.jsonl")
    elif dataset_name == "singleeq":
        dataset = jsonlines_load("dataset/single_eq.jsonl")
    elif dataset_name == "singleop":
        dataset = jsonlines_load("dataset/single_op.jsonl")
    elif dataset_name == "singleaddsub":
        dataset = jsonlines_load("dataset/single_addsub.jsonl")
    elif dataset_name == "multiarith":
        dataset = jsonlines_load("dataset/multiarith.jsonl")

    # slice data based on start and end
    total_num = len(dataset)
    print("total data: ", total_num)

    # output_data = jsonlines_load(input_path)
    output_data = []
    for path in input_path:
        with open(path, "r") as f:
            output = json.load(f)
            output_data.extend(output)

    total = 0
    correct = 0
    error = 0

    for i in range(len(output_data)):
        if dataset_type == 'math':
            if output_data[i] and output_data[i]['majority_answer'] is not None:
                if abs(output_data[i]['majority_answer'] - output_data[i]['answer']) < 1e-3:
                    correct += 1
                else:
                    # print(f"Wrong Answer Index {i}: question: {output_data[i]['question']}\n answer: {output_data[i]['answer']}\n majority_answer: {output_data[i]['majority_answer']}")
                    pass
            else:
                error += 1

        else:
            if output_data[i]['final_ans'] == output_data[i]['answer']:
                correct += 1
            else:
                error += 1

        # total += 1

    print(
        f'Accuracy: {correct/total_num}, Total: {total_num}, Correct: {correct}, Error: {error}')