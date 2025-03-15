import sys
import os

# Debug: Print current directory and Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Remove one dirname call to point to Panacea-R1
# Add the project root to Python path
sys.path.insert(0, project_root)  # This will now add Panacea-R1 to the path


import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import re

# Add these at the top with other global variables
from pyserini.search.lucene import LuceneSearcher
from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer
from tqdm import tqdm


_searcher = None
_tokenizer = SimpleTokenizer()

def get_searcher():
    """Lazily initialize and return the searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = LuceneSearcher.from_prebuilt_index('wikipedia-dpr-100w')
    return _searcher

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
    model.eval()
    return tokenizer, model


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1].strip()
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1].strip()
    else:
        print("[Error] Failed to locate model response header")
        return None, processed_str

    # Regular expression to find the last occurrence of <answer>...</answer>
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, processed_str, re.DOTALL)  # Use re.DOTALL to match multiline content

    if matches:
        return matches[-1].strip(), processed_str  # Return the last matched answer
    else:
        print("[Error] No valid answer tags found")
        return None, processed_str
    
    
def extract_json_from_llm_output(text):
    pattern = r"```json\n([\s\S]+?)\n```"
    matched_jsons = re.findall(pattern, text)
    
    if matched_jsons:
        extracted_json = matched_jsons[-1]  # get the final one
        return json.loads(extracted_json)
    else:
        # backup plan
        pattern = r"\{.*?\}"
        matched_jsons = re.findall(pattern, text, re.DOTALL)
        
        if matched_jsons:
            extracted_json = matched_jsons[-1]  # get the final one
            return json.loads(extracted_json)
        else:
            raise ValueError('No JSON structure found.')
        
def run_index_search_bm25(search_query, topk=50):
    
    searcher = get_searcher()
    
    # Rate limit checking
    hits = searcher.search(search_query, k=topk)
    
    doc_list = [json.loads(hit.lucene_document.get('raw'))['contents'] for hit in hits]
    
    return doc_list


def evaluate_model(model, tokenizer, data_path, device, model_name, save_dir, batch_size=8, search_api=None):
    df = pd.read_parquet(data_path)
    
    inputs = [item[0]['content'] for item in df['prompt'].tolist()]
    targets = df['label'].tolist()
    
    model = model.to(device)
    error_count = 0
    
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    recall_at_20 = []
    recall_at_50 = []
    recall_at_100 = []
    
    for batch_start in tqdm(range(0, len(inputs), batch_size), desc="Evaluating"):
        batch_end = min(batch_start + batch_size, len(inputs))
        batch_inputs = inputs[batch_start:batch_end]
        
        tokenized_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(**tokenized_inputs, max_new_tokens=512)
        
        for i, output in enumerate(output_ids):
            try:
                generated_text = tokenizer.decode(output, skip_special_tokens=True)
                idx = batch_start + i
                # convert target from ndarray to list
                
                extracted_solution, processed_str = extract_solution(generated_text)
                query = json.loads(extracted_solution)['query']
                
                rank = 1001
                target = targets[idx].tolist()
                searched_doc_list = run_index_search_bm25(query, topk=100)
                
                for i in range(len(searched_doc_list)):
                    if has_answers(searched_doc_list[i], target, _tokenizer, regex=False):
                        rank = i + 1
                        break
                    
                recall_at_1.append(1) if rank <= 1 else recall_at_1.append(0)
                recall_at_5.append(1) if rank <= 5 else recall_at_5.append(0)
                recall_at_10.append(1) if rank <= 10 else recall_at_10.append(0)
                recall_at_20.append(1) if rank <= 20 else recall_at_20.append(0)
                recall_at_50.append(1) if rank <= 50 else recall_at_50.append(0)
                recall_at_100.append(1) if rank <= 100 else recall_at_100.append(0)
                
            except:
                recall_at_1.append(0)
                recall_at_5.append(0)
                recall_at_10.append(0)
                recall_at_20.append(0)
                recall_at_50.append(0)
                recall_at_100.append(0)
                error_count += 1
                print(f"Error: {generated_text}, Error count: {error_count}")
                continue
            
            
        try:
            print(f"Recall@1: {sum(recall_at_1) / len(recall_at_1)}")
            print(f"Recall@5: {sum(recall_at_5) / len(recall_at_5)}")
            print(f"Recall@20: {sum(recall_at_20) / len(recall_at_20)}")
            print(f"Recall@100: {sum(recall_at_100) / len(recall_at_100)}")
        except:
            continue

    
    print("Error count: ", error_count)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/shared/eng/pj20/lmr_model/nq_serini_3b/actor/global_step_400")
    parser.add_argument("--data_path", type=str, default="data/local_index_search/nq_serini/test.parquet")
    parser.add_argument("--model_name", type=str, default="nq-3b-step-400")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path)
    evaluate_model(model, tokenizer, args.data_path, device, args.model_name, args.save_dir, args.batch_size)

if __name__ == "__main__":
    main()
