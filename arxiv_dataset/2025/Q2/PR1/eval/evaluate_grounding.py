from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch
import json
import ray
from tqdm import tqdm
import re
import os
import argparse

def parse_float_sequence_within(input_str):
    """Extract the first sequence of four floating-point numbers from the string"""
    patterns = [
        r"\[\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\]",  # [x1,y1,x2,y2]
        r"\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\)",  # (x1,y1,x2,y2)
        r"\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)\s*,\s*\(\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\s*\)"  # (x1,y1),(x2,y2)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, input_str)
        if match:
            return [float(match.group(i)) for i in range(1, 5)]
    return [0.0, 0.0, 0.0, 0.0]

def extract_bbox_answer(content):
    """Extract bounding box from model output"""
    is_qwen2vl = "<|box_start|>" in content
    bbox = parse_float_sequence_within(content)
    return (bbox if is_qwen2vl else [int(x*1000) for x in bbox]), is_qwen2vl

def compute_iou(box1, box2):
    """Compute IoU"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-10)

def compute_accuracy(box1, box2, threshold=0.5):
    """Compute accuracy"""
    return compute_iou(box1, box2) >= threshold

def compute_center_accuracy(box1, box2):
    """Compute center accuracy"""
    cx = (box2[0] + box2[2]) / 2
    cy = (box2[1] + box2[3]) / 2
    return (box1[0] <= cx <= box1[2]) and (box1[1] <= cy <= box1[3])

@ray.remote(num_gpus=1)
class RefCOCOEvaluator:
    """Evaluator class encapsulating the evaluation logic"""
    def __init__(self, args):
        self.args = args
        self.model = LLM(
            model = args.model_path,
            gpu_memory_utilization = 0.7
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            repetition_penalty=1.05,
            max_tokens=512,
            skip_special_tokens=False
        )
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        
        # Evaluation metrics
        self.scorers = {
            "IoU": compute_iou,
            "ACC@0.1": lambda x,y: compute_accuracy(x,y,0.1),
            "ACC@0.3": lambda x,y: compute_accuracy(x,y,0.3),
            "ACC@0.5": lambda x,y: compute_accuracy(x,y,0.5),
            "ACC@0.75": lambda x,y: compute_accuracy(x,y,0.75),
            "ACC@0.95": lambda x,y: compute_accuracy(x,y,0.95),
            "Center_ACC": compute_center_accuracy,
        }
    
    def evaluate_batch(self, batch_data, batch_messages):
        """Evaluate a single batch of data"""
        # Prepare input data
        text = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
        
        # Process visual input
        image_inputs, video_inputs = process_vision_info(batch_messages)
        
        inputs = [{"prompt": prompt, "multi_modal_data": {"image": image}} 
                 for prompt, image in zip(text, image_inputs)]
        outputs = self.model.generate(
            inputs,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        outputs_decoded = [o.outputs[0].text for o in outputs]
        
        # Calculate metrics for this batch
        batch_results = []
        scores = {k: 0.0 for k in self.scorers}
        
        for example, output in zip(batch_data, outputs_decoded):
            pred_box, is_normalized = extract_bbox_answer(output)
            gt_box = example['normalized_solution'] if is_normalized else example['solution']
            
            result = {
                'question': example['problem'],
                'ground_truth': gt_box,
                'model_output': output,
                'extracted_answer': pred_box,
                'scores': {}
            }
            
            for name, scorer in self.scorers.items():
                score = scorer(gt_box, pred_box)
                result['scores'][name] = score
                scores[name] += score
            
            batch_results.append(result)
        
        return batch_results, scores, len(batch_data)

def main(args):
    """Main execution flow"""
    # Initialize Ray
    ray.init()
    
    # Determine the datasets to evaluate
    ALL_DATASETS = [
        'refcoco_val', 'refcoco_testA', 'refcoco_testB',
        'refcocop_val', 'refcocop_testA', 'refcocop_testB',
        'refcocog_val', 'refcocog_test'
    ]
    os.makedirs(f"{args.model_path}/evaluations", exist_ok=True)
    target_datasets = ALL_DATASETS
    
    # Create evaluator actors
    num_workers = torch.cuda.device_count() if torch.cuda.is_available() else 1
    evaluators = [RefCOCOEvaluator.remote(args) for _ in range(num_workers)]
    
    for ds in target_datasets:
        print(f"Processing {ds}...")
        ds_path = os.path.join(args.anno_dir, f"{ds}.json")
        data = json.load(open(ds_path, "r"))
        if args.sample_num > 0:
            data = data[:args.sample_num]
        
        # Prepare all messages
        messages = []
        for x in data:
            img_path = os.path.join(args.image_dir, x['image'])
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_path}"},
                    {"type": "text", "text": f"Output the bounding box of the {x['problem']} in the image."}
                ]
            }])
        
        # Distribute batches to workers
        batch_size = args.batch_size
        result_refs = []
        
        for i in tqdm(range(0, len(data), batch_size * num_workers)):
            # Assign batches to workers in round-robin fashion
            for worker_idx in range(num_workers):
                batch_start = i + worker_idx * batch_size
                if batch_start >= len(data):
                    continue
                
                batch_end = min(batch_start + batch_size, len(data))
                batch_data = data[batch_start:batch_end]
                batch_messages = messages[batch_start:batch_end]
                
                evaluator = evaluators[worker_idx % num_workers]
                result_refs.append(evaluator.evaluate_batch.remote(batch_data, batch_messages))
        
        # Collect results
        final_results = []
        total_samples = 0
        scores = {k: 0.0 for k in [
            "IoU", "ACC@0.1", "ACC@0.3", "ACC@0.5", 
            "ACC@0.75", "ACC@0.95", "Center_ACC"
        ]}
        
        for result_ref in tqdm(result_refs, desc="Collecting results"):
            batch_results, batch_scores, batch_size = ray.get(result_ref)
            final_results.extend(batch_results)
            for k in scores:
                scores[k] += batch_scores[k]
            total_samples += batch_size
        
        # Calculate average score
        avg_scores = {k: round(v/total_samples*100, 2) for k,v in scores.items()}
        result = {
            'dataset': ds,
            'average_scores': avg_scores,
            'details': final_results
        }
        
        # Save results
        output_path = os.path.join('logs', os.path.basename(args.model_path), 'grounding', ds)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(os.path.join(output_path, 'rec_results.json'), 'w') as f:
            json.dump({
                'model': args.model_path,
                'config': vars(args),
                **result
            }, f, indent=2)
        
        print(f"\nResults for {ds}:")
        for k,v in result['average_scores'].items():
            print(f"{k}: {v}%")
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual Localization Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--anno_dir", type=str, default="path/to/refcoco", help="Data root directory")
    parser.add_argument("--image_dir", type=str, default="path/to/coco/", help="Image root directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--sample_num", type=int, default=-1, help="Number of samples (for debugging)")
    args = parser.parse_args()
    main(args)