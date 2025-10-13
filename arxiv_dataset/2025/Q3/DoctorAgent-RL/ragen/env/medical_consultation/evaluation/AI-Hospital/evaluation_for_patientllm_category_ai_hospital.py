import json
import numpy as np
import torch
import re
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
from concurrent.futures import ThreadPoolExecutor, as_completed # Remove ThreadPoolExecutor
from tqdm import tqdm
import argparse
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Global variables for model initialization
QWEN_MODEL_PATH = "Qwen2.5-7B-Instruct"
global_model = None
global_tokenizer = None
global_model_loaded = False
# Add a global device variable
global_device = None
# Add distributed training variables
global_rank = None
global_world_size = None
global_local_rank = None

def setup_distributed():
    """Setup distributed training environment"""
    global global_rank, global_world_size, global_local_rank
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return False
    
    # Get local rank from environment variable
    global_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if global_local_rank == -1:
        print("LOCAL_RANK environment variable not set. Using single GPU.")
        return False
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    # Get rank and world size
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()
    
    # Set device for this process
    torch.cuda.set_device(global_local_rank)
    
    print(f"Initialized process {global_rank}/{global_world_size} on GPU {global_local_rank}")
    return True

def load_model_if_needed():
    """Load the model if it hasn't been loaded yet"""
    global global_model, global_tokenizer, global_model_loaded, global_device

    if not global_model_loaded:
        try:
            # Setup distributed environment
            is_distributed = setup_distributed()
            
            # Determine the device to use
            if is_distributed:
                global_device = torch.device(f"cuda:{global_local_rank}")
            else:
                global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Process {global_rank if is_distributed else 0}: Using device: {global_device}")

            global_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True, padding_side='left')
            
            # 在分布式环境中，每个进程加载模型到自己的GPU
            if is_distributed:
                global_model = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL_PATH,
                    device_map=None,  # 不使用device_map，手动管理
                    trust_remote_code=True
                )
                global_model.to(global_device)
            else:
                # 非分布式环境，使用device_map自动管理
                global_model = AutoModelForCausalLM.from_pretrained(
                    QWEN_MODEL_PATH,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            global_model.eval() # Set model to evaluation mode

            global_model_loaded = True
            print(f"Process {global_rank if is_distributed else 0}: Model loaded successfully on {global_device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return True

class MedicalDiagnosisEvaluator:
    def __init__(self, simulation_data_item, reference_data_list=None, alpha=1.0, beta=0.01):
        """
        Initialize evaluator with simulation data and reference data for information retrieval evaluation.

        Parameters:
            simulation_data_item (dict): A single simulation dialogue data item
            reference_data_list (list, optional): Reference data list for information retrieval evaluation
            alpha (float): Weight for interaction turns in DEI calculation
            beta (float): Weight for average token count in DEI calculation
        """
        self.simulation_data = simulation_data_item

        # Find matching reference data item
        self.reference_data = None
        if reference_data_list:
            for ref_item in reference_data_list:
                if ref_item["self_report"] == simulation_data_item["self_report"]:
                    self.reference_data = ref_item
                    break

        self.alpha = alpha
        self.beta = beta

        # Model and tokenizer are now loaded globally
        # self.model = global_model # Not needed here anymore
        # self.tokenizer = global_tokenizer # Not needed here anymore

    def tokenize_text(self, text):
        """
        Tokenize text, split by characters for Chinese, by spaces for English

        Parameters:
            text (str): Input text

        Returns:
            list: List after tokenization
        """
        if not text:
            return []

        # For Chinese text, tokenize by character; for English text, tokenize by space
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            # Chinese text, tokenize by character
            tokens = list(text)
        else:
            # English text, tokenize by space
            tokens = text.split()

        return tokens

    def calculate_bleu(self, candidate, reference, n_gram=1):
        """
        Calculate BLEU-N score

        Parameters:
            candidate (str): Candidate text
            reference (str): Reference text
            n_gram (int): n-gram size (1, 2, 3, 4)

        Returns:
            float: BLEU-N score
        """
        if not candidate or not reference:
            if not candidate and not reference:  # Both are empty
                return 1.0
            return 0.0

        # Tokenize
        candidate_tokens = self.tokenize_text(candidate)
        reference_tokens = self.tokenize_text(reference)

        if len(candidate_tokens) == 0:
            return 0.0

        # Calculate N-gram precision
        candidate_ngrams = self._get_ngrams(candidate_tokens, n_gram)
        reference_ngrams = self._get_ngrams(reference_tokens, n_gram)

        # Calculate common n-gram count
        common_count = sum((candidate_ngrams & reference_ngrams).values())

        # Calculate precision
        if sum(candidate_ngrams.values()) == 0:
            precision = 0.0
        else:
            precision = common_count / sum(candidate_ngrams.values())

        # Optional: Calculate simple brevity penalty
        bp = 1.0
        if len(candidate_tokens) < len(reference_tokens):
            bp = math.exp(1 - len(reference_tokens) / len(candidate_tokens))

        return bp * precision

    def calculate_rouge(self, candidate, reference, n_gram=1, use_lcs=False):
        """
        Calculate ROUGE-N/L score

        Parameters:
            candidate (str): Candidate text
            reference (str): Reference text
            n_gram (int): n-gram size
            use_lcs (bool): Whether to use LCS algorithm (ROUGE-L)

        Returns:
            tuple: (F1, Precision, Recall)
        """
        if not candidate or not reference:
            if not candidate and not reference:  # Both are empty
                return 1.0, 1.0, 1.0
            return 0.0, 0.0, 0.0

        # Tokenize
        candidate_tokens = self.tokenize_text(candidate)
        reference_tokens = self.tokenize_text(reference)

        if use_lcs:  # ROUGE-L
            lcs_len = self._longest_common_subsequence_length(candidate_tokens, reference_tokens)

            # Calculate precision and recall
            precision = lcs_len / len(candidate_tokens) if candidate_tokens else 0.0
            recall = lcs_len / len(reference_tokens) if reference_tokens else 0.0

        else:  # ROUGE-N
            # Get n-gram counts
            candidate_ngrams = self._get_ngrams(candidate_tokens, n_gram)
            reference_ngrams = self._get_ngrams(reference_tokens, n_gram)

            # Calculate overlapping n-gram count
            overlap_count = sum((candidate_ngrams & reference_ngrams).values())

            # Calculate precision and recall
            precision = overlap_count / sum(candidate_ngrams.values()) if sum(candidate_ngrams.values()) > 0 else 0.0
            recall = overlap_count / sum(reference_ngrams.values()) if sum(reference_ngrams.values()) > 0 else 0.0

        # Calculate F1 score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return f1, precision, recall

    def _get_ngrams(self, tokens, n):
        """
        Get n-gram counts from text

        Parameters:
            tokens (list): Tokenized text
            n (int): n-gram size

        Returns:
            Counter: n-gram counts
        """
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams

    def _longest_common_subsequence_length(self, tokens1, tokens2):
        """
        Calculate the length of the longest common subsequence between two token sequences

        Parameters:
            tokens1 (list): First token sequence
            tokens2 (list): Second token sequence

        Returns:
            int: Length of the longest common subsequence
        """
        if not tokens1 or not tokens2:
            return 0

        m, n = len(tokens1), len(tokens2)
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def extract_diagnosis_and_recommendation(self, model_output):
        """
        Extract diagnosis and recommendation from model output.
        Updated to look for English patterns like "Diagnosis:" and "Recommendation:"

        Parameters:
            model_output (str): The model's output text

        Returns:
            tuple: (diagnosis_text, recommendation_text)
        """
        model_diagnosis = ""
        model_recommendation = ""

        # Look for "Diagnosis:" pattern
        if "Diagnosis:" in model_output:
            diagnosis_parts = model_output.split("Diagnosis:")
            if len(diagnosis_parts) > 1:
                remaining_text = diagnosis_parts[1]
                # Try to find the next separator
                separators = ["Recommendation:", "Treatment:", "Prescription:", "Plan:", "\n\n"]
                for sep in separators:
                    if sep in remaining_text:
                        model_diagnosis = remaining_text.split(sep)[0].strip()
                        break
                if not model_diagnosis:  # If no separator found
                    model_diagnosis = remaining_text.strip()

        # Look for "Recommendation:" pattern
        if "Recommendation:" in model_output:
            recommendation_parts = model_output.split("Recommendation:")
            if len(recommendation_parts) > 1:
                model_recommendation = recommendation_parts[1].strip()
        elif "Treatment:" in model_output:
            recommendation_parts = model_output.split("Treatment:")
            if len(recommendation_parts) > 1:
                model_recommendation = recommendation_parts[1].strip()
        elif "Plan:" in model_output:
            recommendation_parts = model_output.split("Plan:")
            if len(recommendation_parts) > 1:
                model_recommendation = recommendation_parts[1].strip()

        return model_diagnosis, model_recommendation

    def calculate_metrics(self, diagnosis_semantic_score, recommendation_semantic_score):
        """
        Calculate various evaluation metrics for the model's diagnosis and recommendations
        compared to the ground truth, including BLEU-1/2/3/4 and ROUGE-1/2/L.
        Includes pre-calculated semantic scores.

        Parameters:
            diagnosis_semantic_score (float): Pre-calculated semantic score for diagnosis
            recommendation_semantic_score (float): Pre-calculated semantic score for recommendation
        """
        # Get the final dialogue turn containing the diagnosis
        diagnosis_turns = [turn for turn in self.simulation_data["simulation_dialogue"]
                         if turn.get("role") == "doctor" and turn.get("is_diagnosis", False)]

        if not diagnosis_turns:
            # Return empty evaluation results
            empty_metrics = {
                "combined_score": 0.0,
                "diagnosis": {
                    "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0,
                    "rouge_1": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "rouge_2": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "rouge_l": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "semantic_score": 0.0
                },
                "recommendation": {
                    "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0,
                    "rouge_1": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "rouge_2": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "rouge_l": {"f1": 0.0, "precision": 0.0, "recall": 0.0},
                    "semantic_score": 0.0
                }
            }
            return empty_metrics

        final_diagnosis_turn = diagnosis_turns[-1]
        model_output = final_diagnosis_turn["content"]

        # Extract diagnosis and recommendation text using both methods
        model_diagnosis, model_recommendation = self.extract_diagnosis_and_recommendation(model_output)

        # Get ground truth
        true_diagnosis = self.simulation_data.get("true_diagnosis", "")
        true_recommendation = self.simulation_data.get("true_recommendation", "")

        # Calculate evaluation metrics for diagnosis (excluding semantic score here)
        diagnosis_metrics = {
            "bleu_1": self.calculate_bleu(model_diagnosis, true_diagnosis, n_gram=1),
            "bleu_2": self.calculate_bleu(model_diagnosis, true_diagnosis, n_gram=2),
            "bleu_3": self.calculate_bleu(model_diagnosis, true_diagnosis, n_gram=3),
            "bleu_4": self.calculate_bleu(model_diagnosis, true_diagnosis, n_gram=4),
            "rouge_1": {},
            "rouge_2": {},
            "rouge_l": {},
            "semantic_score": diagnosis_semantic_score # Use pre-calculated score
        }

        # Calculate ROUGE metrics
        f1, p, r = self.calculate_rouge(model_diagnosis, true_diagnosis, n_gram=1)
        diagnosis_metrics["rouge_1"] = {"f1": f1, "precision": p, "recall": r}

        f1, p, r = self.calculate_rouge(model_diagnosis, true_diagnosis, n_gram=2)
        diagnosis_metrics["rouge_2"] = {"f1": f1, "precision": p, "recall": r}

        f1, p, r = self.calculate_rouge(model_diagnosis, true_diagnosis, use_lcs=True)
        diagnosis_metrics["rouge_l"] = {"f1": f1, "precision": p, "recall": r}

        # Calculate evaluation metrics for recommendation (excluding semantic score here)
        recommendation_metrics = {
            "bleu_1": self.calculate_bleu(model_recommendation, true_recommendation, n_gram=1),
            "bleu_2": self.calculate_bleu(model_recommendation, true_recommendation, n_gram=2),
            "bleu_3": self.calculate_bleu(model_recommendation, true_recommendation, n_gram=3),
            "bleu_4": self.calculate_bleu(model_recommendation, true_recommendation, n_gram=4),
            "rouge_1": {},
            "rouge_2": {},
            "rouge_l": {},
            "semantic_score": recommendation_semantic_score # Use pre-calculated score
        }

        # Calculate ROUGE metrics
        f1, p, r = self.calculate_rouge(model_recommendation, true_recommendation, n_gram=1)
        recommendation_metrics["rouge_1"] = {"f1": f1, "precision": p, "recall": r}

        f1, p, r = self.calculate_rouge(model_recommendation, true_recommendation, n_gram=2)
        recommendation_metrics["rouge_2"] = {"f1": f1, "precision": p, "recall": r}

        f1, p, r = self.calculate_rouge(model_recommendation, true_recommendation, use_lcs=True)
        recommendation_metrics["rouge_l"] = {"f1": f1, "precision": p, "recall": r}

        # Use both ROUGE-L F1 and semantic scores for the combined score
        rouge_combined = (diagnosis_metrics["rouge_l"]["f1"] + recommendation_metrics["rouge_l"]["f1"]) / 2
        semantic_combined = (diagnosis_semantic_score + recommendation_semantic_score) / 10  # Normalize to 0-1

        # Combine both metrics (equal weight)
        combined_score = (rouge_combined + semantic_combined) / 2

        return {
            "combined_score": combined_score,
            "diagnosis": diagnosis_metrics,
            "recommendation": recommendation_metrics
        }

    def calculate_interaction_efficiency(self):
        """
        Calculate interaction efficiency based on total turns and token count.
        Efficiency = weighted dialogue turns × average tokens per turn
        """
        if type(self.simulation_data["total_turns"]) == int:
            total_turns = self.simulation_data["total_turns"]
        else:
            total_turns = sum(self.simulation_data["total_turns"].values()) / len(self.simulation_data["total_turns"])

        # Calculate average tokens per turn
        doctor_turns = [turn for turn in self.simulation_data["simulation_dialogue"]
                       if turn.get("role") == "doctor"]

        if not doctor_turns:
            return 0, 0, 0  # No doctor turns found

        total_tokens = sum(turn.get("tokens", 0) for turn in doctor_turns)
        avg_tokens = total_tokens / len(doctor_turns)

        # Calculate interaction efficiency (lower is better)
        interaction_efficiency = total_turns * avg_tokens

        return interaction_efficiency, avg_tokens, total_turns

    def calculate_information_retrieval_rate(self):
        """
        计算信息检索率（返回原始准确率和大模型评分）
        返回格式：(precision, model_score)
        """
        # 定义错误信息模式
        error_patterns = [
            r"Sorry, you've asked this question before",
            r"Sorry, I cannot answer your question"
        ]
        
        # 获取所有有效患者回答（过滤错误信息）
        valid_responses = []
        for turn in self.simulation_data.get("simulation_dialogue", []):
            if turn.get("role") == "patient" and "content" in turn:
                response = turn["content"].strip()
                # 过滤错误回答
                if not any(re.search(pattern, response) for pattern in error_patterns):
                    valid_responses.append(response)

        # 计算原始准确率
        total_responses = len([t for t in self.simulation_data.get("simulation_dialogue", []) 
                            if t.get("role") == "patient"])
        valid_count = len(valid_responses)
        precision = valid_count / total_responses if total_responses > 0 else 0.0

        return precision

    def calculate_diagnostic_efficiency_index(self, combined_score, total_turns, avg_tokens):
        """
        Calculate Diagnostic Efficiency Index
        DEI = (score)² / (α × interaction turns + β × average token count)
        Uses combined score as the scoring metric
        Higher DEI indicates better efficiency while maintaining accuracy.

        Parameters:
            combined_score (float): The combined diagnostic performance score
            total_turns (int): Total interaction turns
            avg_tokens (float): Average tokens per turn
        """
        # Calculate denominator using formula
        denominator = self.alpha * total_turns + self.beta * avg_tokens

        # Avoid division by zero
        if denominator == 0:
            return 0.0

        # Calculate DEI
        dei = (combined_score ** 2) / denominator

        return dei

    def extract_model_outputs(self):
        """
        Extract model diagnosis and recommendation text from the simulation data.
        """
        diagnosis_turns = [turn for turn in self.simulation_data["simulation_dialogue"]
                         if turn.get("role") == "doctor" and turn.get("is_diagnosis", False)]

        if not diagnosis_turns:
            return "", ""

        final_diagnosis_turn = diagnosis_turns[-1]
        model_output = final_diagnosis_turn["content"]

        # Extract diagnosis and recommendation text
        model_diagnosis, model_recommendation = self.extract_diagnosis_and_recommendation(model_output)

        # 定义错误信息模式
        error_patterns = [
            r"Sorry, you've asked this question before",
            r"Sorry, I cannot answer your question"
        ]
        
        # 获取所有有效患者回答（过滤错误信息）
        valid_responses = []
        for turn in self.simulation_data.get("simulation_dialogue", []):
            if turn.get("role") == "patient" and "content" in turn:
                response = turn["content"].strip()
                # 过滤错误回答
                if not any(re.search(pattern, response) for pattern in error_patterns):
                    valid_responses.append(response)

        # 构建sentence1（有效回答的拼接）
        model_gathered_info = " ".join(valid_responses)

        return model_diagnosis, model_recommendation, model_gathered_info


def calculate_semantic_similarity_score_batch(data_pairs, info_gather=False):
    """
    Calculate semantic similarity scores for a batch of (candidate, reference) pairs
    using a local language model.

    Parameters:
        data_pairs (list of tuples): List of (candidate_text, reference_text) tuples

    Returns:
        list of float: List of semantic similarity scores (0-5)
    """
    if not data_pairs:
        return []

    # Check if model is loaded
    if not global_model_loaded and not load_model_if_needed():
        print("Warning: Model not loaded, skipping semantic scoring")
        return [0.0] * len(data_pairs)

    prompts = []
    for candidate, reference in data_pairs:
        if not candidate or not reference:
             # Handle empty cases directly
             prompts.append(None) # Use None as a placeholder for empty cases
             continue
        
        if info_gather:
            with open('ragen/env/medical_consultation/evaluation/eval_information_prompt_template.txt', 'r') as file:
                prompt = file.read()
                prompt = prompt.format(patient_self_report=reference, doctor_gathered_info=candidate)
        else:
            with open('ragen/env/medical_consultation/evaluation/eval_prompt_template_v2.txt', 'r') as file:
                prompt = file.read()
                prompt = prompt.format(candidate=candidate, reference=reference)

        # prompt = [
        #     {"role": "user", "content": prompt}
        # ]
        # # Apply chat template
        # prompt = global_tokenizer.apply_chat_template(
        #     prompt,
        #     tokenize=False,
        #     add_generation_prompt=True
        # )
        prompts.append(prompt)

    scores = []
    # Process prompts in batches, handling None placeholders
    valid_prompts = [p for p in prompts if p is not None]
    empty_indices = [i for i, p in enumerate(prompts) if p is None]

    if valid_prompts:
        try:
            # Tokenize the batch of prompts
            inputs = global_tokenizer(valid_prompts, return_tensors="pt", padding=True, truncation=True).to(global_device)

            with torch.no_grad():
                # Generate responses for the batch
                outputs = global_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.01,
                    top_p=0.9,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode and extract scores from the batch responses
            response_texts = global_tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            valid_scores = []
            score_pattern = r'(\d{1,3})(?:\s*\/\s*5)?$'
            for response_text in response_texts:
                
                # extract score from response text <answer>5</answer>
                match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
                if match:
                    matched_response = match.group(0)
                else:
                    matched_response = response_text.strip()
                match = re.search(score_pattern, matched_response, re.DOTALL)
                if match:
                    score = float(match.group(1))
                    valid_scores.append(min(max(score, 0), 5)) # Ensure score is between 0-5
                else:
                    # If no clear score is found, try to find any number in the text
                    numbers = re.findall(r'\b\d{1,3}\b', response_text.strip())
                    found_score = False
                    if numbers:
                        for num in numbers:
                            num_val = float(num)
                            if 0 <= num_val <= 5:
                                valid_scores.append(num_val)
                                found_score = True
                                break
                    if not found_score:
                        print(f"Warning: Could not extract score from: {response_text.strip()}")
                        valid_scores.append(0.0) # Default to middle score if parsing fails

            # Reconstruct the full list of scores, including placeholders for empty cases
            full_scores = [0.0] * len(prompts)
            valid_score_index = 0
            for i in range(len(prompts)):
                if i in empty_indices:
                    full_scores[i] = 0.0 # Score 5 for empty/empty pair
                else:
                    full_scores[i] = valid_scores[valid_score_index]
                    valid_score_index += 1

            scores = full_scores

        except Exception as e:
            print(f"Error in semantic scoring batch: {e}")
            # Return 0.0 for all scores in case of a batch error
            return [0.0] * len(data_pairs)

    else: # All prompts were None (empty pairs)
         scores = [0.0] * len(data_pairs)

    # 在分布式环境中，同步所有进程的结果
    if global_world_size and global_world_size > 1:
        # 收集所有进程的结果
        all_scores = [None] * global_world_size
        dist.all_gather_object(all_scores, scores)
        
        # 只在主进程(rank 0)上合并结果
        if global_rank == 0:
            # 展平所有进程的结果
            scores = [score for process_scores in all_scores for score in process_scores]
        else:
            # 非主进程返回空列表
            scores = []

    return scores


def calculate_category_averages(results_by_category):
    """
    Calculate average metrics for each category.
    
    Parameters:
        results_by_category (dict): Dictionary with category names as keys and lists of results as values
        
    Returns:
        dict: Average results by category
    """
    category_averages = {}
    
    for category, results in results_by_category.items():
        if not results:
            continue
            
        # Calculate averages for this category
        avg_result = {
            "diagnostic_performance": {
                # "combined_score": np.mean([r["diagnostic_performance"]["combined_score"] for r in results]),
                "diagnosis": {
                    # "bleu_1": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_1"] for r in results]),
                    # "bleu_2": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_2"] for r in results]),
                    # "bleu_3": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_3"] for r in results]),
                    # "bleu_4": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_4"] for r in results]),
                    # "rouge_1": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["f1"] for r in results]),
                    #             "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["precision"] for r in results]),
                    #             "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["recall"] for r in results])},
                    # "rouge_2": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["f1"] for r in results]),
                    #             "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["precision"] for r in results]),
                    #             "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["recall"] for r in results])},
                    # "rouge_l": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["f1"] for r in results]),
                    #             "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["precision"] for r in results]),
                    #             "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["recall"] for r in results])},
                    "semantic_score": np.mean([r["diagnostic_performance"]["diagnosis"] for r in results])
                },
                "recommendation": {
                    # "bleu_1": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_1"] for r in results]),
                    # "bleu_2": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_2"] for r in results]),
                    # "bleu_3": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_3"] for r in results]),
                    # "bleu_4": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_4"] for r in results]),
                    # "rouge_1": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["f1"] for r in results]),
                    #             "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["precision"] for r in results]),
                    #             "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["recall"] for r in results])},
                    # "rouge_2": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["f1"] for r in results]),
                    #             "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["precision"] for r in results]),
                    #             "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["recall"] for r in results])},
                    # "rouge_l": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["f1"] for r in results]),
                    #             "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["precision"] for r in results]),
                    #             "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["recall"] for r in results])},
                    "semantic_score": np.mean([r["diagnostic_performance"]["recommendation"] for r in results])
                }
            },
            "interaction_efficiency": {
                "total_turns": np.mean([r["interaction_efficiency"]["total_turns"] for r in results]),
                # "avg_tokens": np.mean([r["interaction_efficiency"]["avg_tokens"] for r in results]),
                # "interaction_efficiency": np.mean([r["interaction_efficiency"]["interaction_efficiency"] for r in results])
            },
            # "information_retrieval": {
            #     "precision": np.mean([r["information_retrieval"]["precision"] for r in results]),
            #     "model_score": np.mean([r["information_retrieval"]["model_score"] for r in results])
            # },
            # "diagnostic_efficiency_index": np.mean([r["diagnostic_efficiency_index"] for r in results]),
            "case_count": len(results)
        }
        
        category_averages[category] = avg_result
    
    return category_averages


def evaluate_all_cases(simulation_data_list, reference_data_list=None, alpha=1.0, beta=0.01, batch_size=16):
    """
    Evaluate all simulation cases using batch processing for semantic similarity.

    Parameters:
        simulation_data_list (list): List of simulation dialogue data items.
        reference_data_list (list, optional): Reference data list for information retrieval evaluation.
        alpha (float): Weight for interaction turns in DEI calculation.
        beta (float): Weight for average token count in DEI calculation.
        batch_size (int): Number of cases to process in each batch for semantic scoring.

    Returns:
        dict: Evaluation results including case-level results, overall average results, and category-based averages.
    """
    if not simulation_data_list:
        print("No simulation data provided")
        return {}

    # Load model once before evaluation
    if not load_model_if_needed():
        print("Model loading failed. Cannot proceed with evaluation.")
        return {}

    # filter out simulation data which not in reference data
    all_ref_report = [ref_item["self_report"] for ref_item in reference_data_list]
    simulation_data_list = [sim_item for sim_item in simulation_data_list if sim_item["self_report"] in all_ref_report]
    print("Filtered simulation data list: ", len(simulation_data_list))

    if 'category' not in simulation_data_list[0]:
        category_index = {}
        des_index = {}
        for ref_item in reference_data_list:
            category_index[ref_item["self_report"]] = ref_item["category"]
            des_index[ref_item["self_report"]] = ref_item["enhanced_description"]
        for sim_item in simulation_data_list:
            sim_item["category"] = category_index[sim_item["self_report"]]
            sim_item["enhanced_description"] = des_index[sim_item["self_report"]]

    case_results = []
    all_diagnosis_pairs = []
    all_recommendation_pairs = []
    all_info_gather_pairs = []
    case_indices = []
    results_by_category = defaultdict(list)  # Store results grouped by category

    # --- Step 1: Extract model outputs and prepare data for batching ---
    print(f"Process {global_rank if global_rank is not None else 0}: Extracting model outputs and preparing for batching...")
    
    # 在分布式环境中，每个进程只处理部分数据
    if global_world_size and global_world_size > 1:
        # 计算每个进程处理的数据量
        per_rank = len(simulation_data_list) // global_world_size
        remainder = len(simulation_data_list) % global_world_size
        
        # 计算当前进程的起始和结束索引
        start_idx = global_rank * per_rank + min(global_rank, remainder)
        end_idx = start_idx + per_rank + (1 if global_rank < remainder else 0)
        
        # 只处理分配给当前进程的数据
        local_simulation_data = simulation_data_list[start_idx:end_idx]
        print(f"Process {global_rank}: Processing {len(local_simulation_data)} items (indices {start_idx}-{end_idx-1})")
    else:
        local_simulation_data = simulation_data_list
    
    for i, sim_item in enumerate(tqdm(local_simulation_data, desc=f"Process {global_rank if global_rank is not None else 0}: Preparing Data")):
        evaluator = MedicalDiagnosisEvaluator(
            simulation_data_item=sim_item,
            reference_data_list=reference_data_list,
            alpha=alpha,
            beta=beta
        )
        # model_diagnosis, model_recommendation, model_gathered_info = evaluator.extract_model_outputs()
        model_diagnosis, model_recommendation = sim_item["final_diagnosis"]["diagnosis"], sim_item["final_diagnosis"]["recommendation"]
        true_diagnosis = sim_item.get("true_diagnosis", "")
        true_recommendation = sim_item.get("true_recommendation", "")
        # true_info = sim_item.get("enhanced_description", "")

        all_diagnosis_pairs.append((model_diagnosis, true_diagnosis))
        all_recommendation_pairs.append((model_recommendation, true_recommendation))
        # all_info_gather_pairs.append((model_gathered_info, true_info))
        case_indices.append(i) # Store index to map scores back

    # --- Step 2: Calculate semantic scores in batches ---
    print(f"Process {global_rank if global_rank is not None else 0}: Calculating semantic scores in batches (batch size: {batch_size})...")
    all_diagnosis_semantic_scores = []
    all_recommendation_semantic_scores = []
    # all_info_gather_semantic_scores = []

    # Adjust batch size based on number of GPUs
    if global_world_size and global_world_size > 1:
        # Increase batch size proportionally to number of GPUs
        effective_batch_size = batch_size * global_world_size
        print(f"Process {global_rank}: Using effective batch size of {effective_batch_size} across {global_world_size} GPUs")
    else:
        effective_batch_size = batch_size

    # Process diagnosis pairs in batches
    for i in tqdm(range(0, len(all_diagnosis_pairs), effective_batch_size), desc=f"Process {global_rank if global_rank is not None else 0}: Semantic Scoring (Diagnosis)"):
        batch_pairs = all_diagnosis_pairs[i:i + effective_batch_size]
        batch_scores = calculate_semantic_similarity_score_batch(batch_pairs)
        all_diagnosis_semantic_scores.extend(batch_scores)

    # Process recommendation pairs in batches
    for i in tqdm(range(0, len(all_recommendation_pairs), effective_batch_size), desc=f"Process {global_rank if global_rank is not None else 0}: Semantic Scoring (Recommendation)"):
        batch_pairs = all_recommendation_pairs[i:i + effective_batch_size]
        batch_scores = calculate_semantic_similarity_score_batch(batch_pairs)
        all_recommendation_semantic_scores.extend(batch_scores)

    # # Process info gather pairs in batches
    # for i in tqdm(range(0, len(all_info_gather_pairs), effective_batch_size), desc=f"Process {global_rank if global_rank is not None else 0}: Semantic Scoring (Info Gather)"):
    #     batch_pairs = all_info_gather_pairs[i:i + effective_batch_size]
    #     batch_scores = calculate_semantic_similarity_score_batch(batch_pairs, info_gather=True)
    #     all_info_gather_semantic_scores.extend(batch_scores)

    # 在分布式环境中，收集所有进程的结果
    if global_world_size and global_world_size > 1:
        # 收集所有进程的诊断和推荐分数
        all_process_diagnosis_scores = [None] * global_world_size
        all_process_recommendation_scores = [None] * global_world_size
        # all_process_info_gather_scores = [None] * global_world_size
        all_process_case_indices = [None] * global_world_size
        
        dist.all_gather_object(all_process_diagnosis_scores, all_diagnosis_semantic_scores)
        dist.all_gather_object(all_process_recommendation_scores, all_recommendation_semantic_scores)
        # dist.all_gather_object(all_process_info_gather_scores, all_info_gather_semantic_scores)
        dist.all_gather_object(all_process_case_indices, case_indices)
        
        # 只在主进程(rank 0)上合并结果
        if global_rank == 0:
            # 展平所有进程的结果
            all_diagnosis_semantic_scores = [score for process_scores in all_process_diagnosis_scores for score in process_scores]
            all_recommendation_semantic_scores = [score for process_scores in all_process_recommendation_scores for score in process_scores]
            # all_info_gather_semantic_scores = [score for process_scores in all_process_info_gather_scores for score in process_scores]
            case_indices = [idx for process_indices in all_process_case_indices for idx in process_indices]
        else:
            # 非主进程返回空结果
            return {}

    # --- Step 3: Calculate other metrics and assemble results ---
    print(f"Process {global_rank if global_rank is not None else 0}: Calculating other metrics and assembling results...")
    
    # 在分布式环境中，只有主进程计算最终结果
    if global_world_size and global_world_size > 1 and global_rank != 0:
        return {}
    
    for i in tqdm(range(len(simulation_data_list)), desc=f"Process {global_rank if global_rank is not None else 0}: Calculating Other Metrics"):
        sim_item = simulation_data_list[i]
        evaluator = MedicalDiagnosisEvaluator(
            simulation_data_item=sim_item,
            reference_data_list=reference_data_list,
            alpha=alpha,
            beta=beta
        )

        # Get pre-calculated semantic scores for this case
        diagnosis_semantic_score = all_diagnosis_semantic_scores[i]
        recommendation_semantic_score = all_recommendation_semantic_scores[i]
        # info_gather_semantic_score = all_info_gather_semantic_scores[i]

        # Calculate metrics, passing the semantic scores
        # metrics = evaluator.calculate_metrics(diagnosis_semantic_score, recommendation_semantic_score)
        # interaction_efficiency, avg_tokens, total_turns = evaluator.calculate_interaction_efficiency()
        total_turns = sum(sim_item["total_turns"].values()) / len(sim_item["total_turns"])
        # precision = evaluator.calculate_information_retrieval_rate()

        # Calculate DEI using the combined score and efficiency metrics
        # dei = evaluator.calculate_diagnostic_efficiency_index(
        #     combined_score=metrics["combined_score"],
        #     total_turns=total_turns,
        #     avg_tokens=avg_tokens
        # )

        # Add ID and category for identification
        result = {
            "id": sim_item.get("id"),
            "category": sim_item.get("category", "unknown"),  # Get category from simulation data
            "diagnostic_performance": {
                # "combined_score": metrics["combined_score"],
                "diagnosis": diagnosis_semantic_score,
                "recommendation": recommendation_semantic_score
            },
            "interaction_efficiency": {
                "total_turns": total_turns,
                # "avg_tokens": avg_tokens,
                # "interaction_efficiency": interaction_efficiency
            },
            # "information_retrieval": {
            #     "precision": precision,
            #     "model_score": info_gather_semantic_score
            # },
            # "diagnostic_efficiency_index": dei
        }
        
        case_results.append(result)
        
        # Group results by category
        category = sim_item.get("category", "unknown")
        results_by_category[category].append(result)

    # Calculate overall average results
    if not case_results:
        return {"case_results": [], "average_result": {}, "category_results": {}}

    # Calculate overall averages
    avg_result = {
        "diagnostic_performance": {
            # "combined_score": np.mean([r["diagnostic_performance"]["combined_score"] for r in case_results]),
            "diagnosis": {
                # "bleu_1": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_1"] for r in case_results]),
                # "bleu_2": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_2"] for r in case_results]),
                # "bleu_3": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_3"] for r in case_results]),
                # "bleu_4": np.mean([r["diagnostic_performance"]["diagnosis"]["bleu_4"] for r in case_results]),
                # "rouge_1": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["f1"] for r in case_results]),
                #             "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["precision"] for r in case_results]),
                #             "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_1"]["recall"] for r in case_results])},
                # "rouge_2": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["f1"] for r in case_results]),
                #             "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["precision"] for r in case_results]),
                #             "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_2"]["recall"] for r in case_results])},
                # "rouge_l": {"f1": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["f1"] for r in case_results]),
                #             "precision": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["precision"] for r in case_results]),
                #             "recall": np.mean([r["diagnostic_performance"]["diagnosis"]["rouge_l"]["recall"] for r in case_results])},
                "semantic_score": np.mean([r["diagnostic_performance"]["diagnosis"] for r in case_results])
            },
            "recommendation": {
                # "bleu_1": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_1"] for r in case_results]),
                # "bleu_2": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_2"] for r in case_results]),
                # "bleu_3": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_3"] for r in case_results]),
                # "bleu_4": np.mean([r["diagnostic_performance"]["recommendation"]["bleu_4"] for r in case_results]),
                # "rouge_1": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["f1"] for r in case_results]),
                #             "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["precision"] for r in case_results]),
                #             "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_1"]["recall"] for r in case_results])},
                # "rouge_2": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["f1"] for r in case_results]),
                #             "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["precision"] for r in case_results]),
                #             "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_2"]["recall"] for r in case_results])},
                # "rouge_l": {"f1": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["f1"] for r in case_results]),
                #             "precision": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["precision"] for r in case_results]),
                #             "recall": np.mean([r["diagnostic_performance"]["recommendation"]["rouge_l"]["recall"] for r in case_results])},
                "semantic_score": np.mean([r["diagnostic_performance"]["recommendation"] for r in case_results])
            }
        },
        "interaction_efficiency": {
            "total_turns": np.mean([r["interaction_efficiency"]["total_turns"] for r in case_results]),
            # "avg_tokens": np.mean([r["interaction_efficiency"]["avg_tokens"] for r in case_results]),
            # "interaction_efficiency": np.mean([r["interaction_efficiency"]["interaction_efficiency"] for r in case_results])
        },
        # "information_retrieval": {
        #     "precision": np.mean([r["information_retrieval"]["precision"] for r in case_results]),
        #     "model_score": np.mean([r["information_retrieval"]["model_score"] for r in case_results])
        # },
        # "diagnostic_efficiency_index": np.mean([r["diagnostic_efficiency_index"] for r in case_results])
    }

    # Calculate category-based averages
    category_averages = calculate_category_averages(results_by_category)

    return {
        "case_results": case_results,
        "average_result": avg_result,
        "category_results": category_averages
    }


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Medical Dialogue Diagnosis Evaluation')
    parser.add_argument('--simulation_data', type=str,
                        default='/play/fengyichun/Jiawei/RL_multi_com/code/doctor/baseline/test_qwen_en_output.json',
                        help='Path to simulation data JSON file')
    parser.add_argument('--reference_data', type=str,
                        default='/play/fengyichun/Jiawei/RL_multi_com/code/Evaluation/test_en_2.json',
                        help='Path to reference data JSON file')
    parser.add_argument('--output', type=str,
                        default='evaluation_results.json',
                        help='Path to output results JSON file')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for interaction turns in DEI calculation')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Weight for average token count in DEI calculation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for semantic similarity calculation')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')

    args = parser.parse_args()

    # Load simulation data
    try:
        with open(args.simulation_data, 'r', encoding='utf-8') as f:
            simulation_data = json.load(f)
            simulation_data.sort(key=lambda x: x["id"])
    except FileNotFoundError:
        print(f"Simulation data file not found: {args.simulation_data}")
        return

    # Load reference data
    reference_data = None
    try:
        with open(args.reference_data, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)
    except FileNotFoundError:
        print(f"Reference data file not found: {args.reference_data}, skipping info retrieval rate.")

    # Run evaluation
    results = evaluate_all_cases(
        simulation_data_list=simulation_data,
        reference_data_list=reference_data,
        alpha=args.alpha,
        beta=args.beta,
        batch_size=args.batch_size # Pass batch size
    )

    # Save results to output file
    if results:
        # Only save results on the main process (rank 0)
        if global_rank is None or global_rank == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # Print summary
            if "average_result" in results and results["average_result"]:
                avg = results["average_result"]
                print("\n=== Overall Average Evaluation Results ===")
                print(f"Combined Score: {avg['diagnostic_performance']['combined_score']:.4f}")
                print(f"Diagnostic Efficiency Index: {avg['diagnostic_efficiency_index']:.4f}")
                print(f"Interaction Turns: {avg['interaction_efficiency']['total_turns']:.2f}")
                print(f"Average Tokens: {avg['interaction_efficiency']['avg_tokens']:.2f}")
                print(f"Info Retrieval Precision: {avg['information_retrieval']['precision']:.4f}")
                print(f"Info Retrieval Model Score: {avg['information_retrieval']['model_score']:.4f}")
                
                # Print category-based results
                if "category_results" in results and results["category_results"]:
                    print("\n=== Category-based Evaluation Results ===")
                    for category, cat_avg in results["category_results"].items():
                        print(f"\nCategory: {category} ({cat_avg['case_count']} cases)")
                        print(f"  Combined Score: {cat_avg['diagnostic_performance']['combined_score']:.4f}")
                        print(f"  Diagnostic Efficiency Index: {cat_avg['diagnostic_efficiency_index']:.4f}")
                        print(f"  Interaction Turns: {cat_avg['interaction_efficiency']['total_turns']:.2f}")
                        print(f"  Average Tokens: {cat_avg['interaction_efficiency']['avg_tokens']:.2f}")
                        print(f"  Info Retrieval Model Score: {cat_avg['information_retrieval']['model_score']:.4f}")
                
                print(f"\nFull results saved to: {args.output}")
            else:
                print("\nNo results to display.")
        else:
            print(f"Process {global_rank}: Skipping result saving (only rank 0 saves results)")
    
    # 在分布式环境中，确保所有进程都完成
    if global_world_size and global_world_size > 1:
        dist.barrier()
        print(f"Process {global_rank}: Evaluation completed")


if __name__ == "__main__":
    main()