import re
import os
import json
import datetime
import argparse
from tqdm import tqdm
import ray
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

def extract_boxes(s):
    res = []
    # Regular expression to match the bounding box block
    pattern = r'<\|box_start\|>(.*?)<\|box_end\|>'
    bboxs = re.findall(pattern, s)
    for bbox in bboxs:
        try:
            (x1, y1), (x2, y2) = eval(bbox)
            res.append(((x1 + x2) / 2, (y1 + y2) / 2))
        except Exception:
            pass
    return res

def parse_output(output, data_item):
    output = output.replace("<|endoftext|>", "")
    boxes = extract_boxes(output)
    return {
        'question': data_item['question'],
        'ground_truth': data_item['ground_truth'],
        'model_output': output,
        'extracted_answer': len(boxes) if boxes else None
    }

@ray.remote(num_gpus=1)
def remote_inference(data, model_args, batch_size, processor, question_template):
    # Initialize the LLM model with the provided model path and GPU settings
    llm = LLM(
        model=model_args['model_path'],
        tensor_parallel_size=1,
        gpu_memory_utilization=model_args['gpu_memory_utilization'],
    )
    # Set sampling parameters
    sampling_params = SamplingParams(temperature=0, max_tokens=512, skip_special_tokens=False)

    # Construct the message list for each data item
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{d['image_path']}"},
            {"type": "text", "text": question_template.format(question=d['question'])}
        ]
    } for d in data]

    results = []
    # Process data in batches
    for i in tqdm(range(0, len(messages), batch_size), desc="Remote Inference"):
        batch_msgs = messages[i:i + batch_size]
        batch_data = data[i:i + batch_size]
        # Apply the processor to generate prompts for each message
        text_prompts = [processor.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)
                        for msg in batch_msgs]
        # Process image data using the vision processing utility
        image_inputs, _ = process_vision_info(batch_msgs)
        # Construct inputs for the model
        inputs = [{"prompt": p, "multi_modal_data": {"image": img}}
                  for p, img in zip(text_prompts, image_inputs)]
        # Generate outputs using the model
        outputs = llm.generate(inputs, sampling_params, use_tqdm=False)
        # Parse each output
        for j, output in enumerate(outputs):
            result = parse_output(output.outputs[0].text, batch_data[j])
            results.append(result)
    return results

class CountingEvaluator:
    def __init__(self, args):
        """
        Initialize the CountingEvaluator.
        
        Loads the evaluation datasets and sets up model and processor parameters.
        
        Args:
            args (Namespace): Parsed command-line arguments.
        """
        self.args = args
        self.model_args = {
            "model_path": args.model_path,
            "gpu_memory_utilization": 0.5
        }
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.QUESTION_TEMPLATE = "Output all the bounding boxes of the {question}"
        # Define the evaluation dataset names
        self.eval_sets = [
            "pixmo_count_test540",
            "pixmo_count_val540",
        ]
        self.dataset_dict = {}
        # Load evaluation data from JSONL files
        for eval_set in self.eval_sets:
            file_path = os.path.join(args.anno_dir, f"{eval_set}.jsonl")
            with open(file_path, "r") as f:
                data = [json.loads(line) for line in f]
            for d in data:
                d['image_path'] = os.path.join(args.image_dir, d['image_path'])
            self.dataset_dict[eval_set] = {
                'data': data,
            }

    def evaluate(self):
        """
        Evaluate the model on all datasets concurrently.
        
        For each evaluation dataset, a remote inference task is launched concurrently 
        (each requesting a GPU) using Ray. After all tasks finish, the results are collated,
        overall accuracy is computed, and detailed results are saved to a JSON file.
        
        Returns:
            float: The overall average accuracy (in percent).
        """
        output_path = os.path.join('logs', os.path.basename(self.args.model_path), 'counting')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        all_results = {}
        total_correct = 0
        total_samples = 0

        # Create a dictionary to store future tasks for each evaluation set
        futures = {}
        for eval_set in self.eval_sets:
            dataset = self.dataset_dict[eval_set]
            # Launch a remote task for each evaluation set concurrently.
            futures[eval_set] = remote_inference.remote(
                dataset['data'], self.model_args, self.args.batch_size,
                self.processor, self.QUESTION_TEMPLATE
            )

        # Retrieve the results from each remote task once they are finished
        for eval_set, future in futures.items():
            results = ray.get(future)
            # Calculate the number of correct answers for the current dataset
            correct = sum(1 for r in results if r['extracted_answer'] == r['ground_truth'])
            accuracy = correct / len(results) * 100
            all_results[eval_set] = {
                'accuracy': accuracy,
                'results': results
            }
            print(f"Dataset: {eval_set}, Accuracy: {accuracy:.2f}%")
            total_correct += correct
            total_samples += len(results)

        # Calculate overall average accuracy
        avg_accuracy = total_correct / total_samples * 100
        all_results['average_accuracy'] = avg_accuracy

        # Save the results to a JSON file
        self.all_results = all_results
        print(f"Saving results to {output_path}")
        with open(os.path.join(output_path, f"counting_scores_{datetime.datetime.now().strftime('%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(all_results, f, indent=2)

        return avg_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the model directory")
    parser.add_argument("--anno_dir", required=True, help="Directory of annotation files")
    parser.add_argument("--image_dir", required=True, help="Directory of image files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    args = parser.parse_args()

    # Initialize Ray
    ray.init()

    evaluator = CountingEvaluator(args)
    accuracy = evaluator.evaluate()
    print(f"Final Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()