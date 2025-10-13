import torch
import time
import re
import json
import vllm
from vllm import SamplingParams
from utils import write_to_file, clean_text

def save_results(qns, predictions, output_file):
    results = []
    for idx, prediction in enumerate(predictions):
        qn = qns[idx]['question']
        qn_id = qns[idx]['id']
        ground_truth = qns[idx]['label']
        results.append({"id": qn_id, "question": qn, "ground_truth": ground_truth, "prediction": prediction})

    with open(output_file , 'w') as f:
        json.dump(results, f, indent=4)
    
    return 

class InferenceEngine:
    def __init__(self, model, seed, gpu):

        sampling_params = SamplingParams(temperature=0.90, top_p=0.9, max_tokens=1000, seed=seed, stop = '*** END')
        LLM = vllm.LLM(model=model, tensor_parallel_size=gpu, gpu_memory_utilization=0.9,trust_remote_code=True)

        self.LLM = LLM
        self.sampling_params = sampling_params
        self.predictions = []

    def infer(self, prompts, method, output_file):

        outputs = self.LLM.generate(prompts, self.sampling_params)

        if method == "DirectAnswer":
            for idx, output in enumerate(outputs):
                
                answer = clean_text(output.outputs[0].text).strip().lower()
                self.predictions.append(answer)

                results = f"prompt: {prompts[idx]}\n\n"
                results += f"Answer: {answer}\n"
                results += "\n\n-----------------------------------------------------\n\n"
                write_to_file(output_file, results)
        
        elif method == "Explain" or method == 'CoT':
            for idx, output in enumerate(outputs):
                result = clean_text(output.outputs[0].text).strip().lower()
                pattern = r"final answer:\s*(.*)"
                match = re.search(pattern, result)
                if match:
                    answer =  match.group(1).strip()
                else:
                    answer = ""
                self.predictions.append(answer)
                results = f"prompt: {prompts[idx]}\n\n"
                results += f"Final answer: {answer}\n\n"
                results += f"output: {result}\n"
                results += "\n\n-----------------------------------------------------\n\n"
                write_to_file(output_file, results)

        return self.predictions
