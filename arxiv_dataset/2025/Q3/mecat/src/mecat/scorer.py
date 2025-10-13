import os
import sys
import json
import torch
from typing import Dict, List, Tuple, Optional
from loguru import logger
from tabulate import tabulate
from .utils import SilentStream, check_java_available
from collections import Counter
import math


class PyTorchBleuScorer:
    """PyTorch-based BLEU scorer implementation"""
    
    def __init__(self, max_n=4):
        self.max_n = max_n
    
    def _tokenize(self, text):
        """Simple tokenization by splitting on whitespace"""
        return text.lower().split()
    
    def _compute_ngrams(self, tokens, n):
        """Compute n-grams from tokens"""
        if n > len(tokens):
            return Counter()
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    
    def _compute_bleu_n(self, candidate_tokens, reference_tokens_list, n):
        """Compute BLEU-n score for a single candidate against multiple references"""
        candidate_ngrams = self._compute_ngrams(candidate_tokens, n)
        
        if len(candidate_ngrams) == 0:
            return 0.0
        
        # Compute maximum count for each n-gram across all references
        max_counts = Counter()
        for ref_tokens in reference_tokens_list:
            ref_ngrams = self._compute_ngrams(ref_tokens, n)
            for ngram in candidate_ngrams:
                max_counts[ngram] = max(max_counts[ngram], ref_ngrams[ngram])
        
        # Compute clipped counts
        clipped_counts = Counter()
        for ngram in candidate_ngrams:
            clipped_counts[ngram] = min(candidate_ngrams[ngram], max_counts[ngram])
        
        # Compute precision
        clipped_total = sum(clipped_counts.values())
        total = sum(candidate_ngrams.values())
        
        return clipped_total / total if total > 0 else 0.0
    
    def _compute_brevity_penalty(self, candidate_length, reference_lengths):
        """Compute brevity penalty"""
        # Find the reference length closest to candidate length
        closest_ref_len = min(reference_lengths, key=lambda x: abs(x - candidate_length))
        
        if candidate_length > closest_ref_len:
            return 1.0
        elif closest_ref_len == 0:
            return 0.0
        else:
            return math.exp(1 - closest_ref_len / candidate_length)
    
    def compute_score(self, gts_tokenized, res_tokenized):
        """
        Compute BLEU scores
        Args:
            gts_tokenized: Dict[str, List[Dict]] - ground truth tokenized captions
            res_tokenized: Dict[str, List[Dict]] - result tokenized captions
        Returns:
            Tuple[List[float], List[List[float]]] - (overall scores, individual scores)
        """
        sample_ids = list(gts_tokenized.keys())
        
        # Initialize score accumulation
        bleu_scores = [[] for _ in range(self.max_n)]
        
        for sample_id in sample_ids:
            # Get candidate and references
            candidate = res_tokenized[sample_id][0]['caption']  # Assuming single prediction
            references = [ref['caption'] for ref in gts_tokenized[sample_id]]
            
            # Tokenize
            candidate_tokens = self._tokenize(candidate)
            reference_tokens_list = [self._tokenize(ref) for ref in references]
            reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
            
            # Compute brevity penalty
            bp = self._compute_brevity_penalty(len(candidate_tokens), reference_lengths)
            
            # Compute BLEU-n scores
            sample_bleu_scores = []
            for n in range(1, self.max_n + 1):
                if len(candidate_tokens) < n:
                    bleu_n = 0.0
                else:
                    precision_n = self._compute_bleu_n(candidate_tokens, reference_tokens_list, n)
                    if precision_n == 0.0:
                        bleu_n = 0.0
                    else:
                        # Compute geometric mean of precisions
                        log_precisions = []
                        for i in range(1, n + 1):
                            p_i = self._compute_bleu_n(candidate_tokens, reference_tokens_list, i)
                            if p_i > 0:
                                log_precisions.append(math.log(p_i))
                            else:
                                log_precisions.append(float('-inf'))
                        
                        if any(p == float('-inf') for p in log_precisions):
                            bleu_n = 0.0
                        else:
                            geometric_mean = math.exp(sum(log_precisions) / len(log_precisions))
                            bleu_n = bp * geometric_mean
                
                sample_bleu_scores.append(bleu_n)
                bleu_scores[n-1].append(bleu_n)
        
        # Compute overall scores (average across all samples)
        overall_scores = [sum(scores) / len(scores) if scores else 0.0 for scores in bleu_scores]
        
        # Individual scores per sample
        individual_scores = [bleu_scores[i] for i in range(self.max_n)]
        
        return overall_scores, individual_scores


class BaseScorer:
    def __init__(self, metrics: List[str] = None, batch_size: int = 32, device: Optional[str] = None):
        """
        Initialize the scorer, optionally enable specific metrics
        Args:
            metrics: List[str], the metrics to enable, None means enable all metrics
            batch_size: int, batch size for FENSE metric
            device: str, device for FENSE metric (e.g., 'cpu', 'cuda')
        Returns:
            List[str]: all available metrics
        """
        self.metrics = metrics if metrics else self.get_available_metrics()
        self.batch_size = batch_size
        self.device = device
        self.fense_scorer = self._init_fense_scorer() if "fense" in self.metrics else None
        self.date_scorer = self._init_date_scorer() if "date" in self.metrics else None
        # Traditional scorers will be initialized on-demand in _evaluate method

    def get_available_metrics(self) -> List[str]:
        """
        Returns:
            List[str]: all available metrics
        """
        return ["bleu", "fense", "date"]

    def _init_fense_scorer(self):
        """
        Initialize the FENSE scorer only if fense metric is requested
        """
        if "fense" not in self.metrics:
            return None
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mecat.DATE import DATEEvaluator

        logger.info("Initializing FENSE scorer ...")
        device = self.device if self.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = self.batch_size if self.batch_size is not None else 32
        return DATEEvaluator(
            device=device,
            batch_size=batch_size,
            sbert_model="paraphrase-TinyBERT-L6-v2",
            echecker_model="echecker_clotho_audiocaps_base",
            return_type="fense",
        )

    def _init_date_scorer(self):
        """Initialize DATE scorer with CPU priority option."""
        if "date" not in self.metrics:
            return None
            
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from mecat.DATE import DATEEvaluator
        
        logger.info("Initializing DATE scorer with CPU priority...")
        
        device = self.device if self.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = self.batch_size if self.batch_size is not None else 32
        
        # Enable CPU priority by default to avoid GPU memory issues
        cpu_priority = True
        
        return DATEEvaluator(
            device=device,
            batch_size=batch_size,
            sbert_model="paraphrase-TinyBERT-L6-v2",
            echecker_model="echecker_clotho_audiocaps_base",
            return_type="date",
            cpu_priority=cpu_priority,  # Enable CPU priority by default
        )

    def _evaluate(
        self,
        group_name: str,
        gts: Dict[str, List[Dict[str, str]]],
        res: Dict[str, List[Dict[str, str]]],
        raw_data: List[Dict[str, str]],
        save_path: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], List[Dict[str, str]], dict]:
        # Extract group_name for DATE evaluation
        # For QA tasks, group_name is already the subtask (e.g., "direct_perception", "sound_characteristics")
        # For caption tasks, group_name format is "caption_long_default", "caption_short_default", etc.
        # For DATE evaluation, we need to map to the appropriate group_name from GROUP_CONFIGS
        
        # Check if this is a QA task by looking at the group_name
        # QA tasks have format like "qa_direct_perception_default", "qa_sound_characteristics_default", etc.
        qa_subtasks = [
            "direct_perception", "sound_characteristics", "quality_assessment",
            "environment_reasoning", "inference_judgement", "application_context"
        ]
        
        # Check if group_name starts with "qa_" and contains a QA subtask
        is_qa_task = group_name.startswith("qa_") and any(subtask in group_name for subtask in qa_subtasks)
        
        if is_qa_task:
            # Extract the actual subtask from the group_name (e.g., "qa_direct_perception_default" -> "direct_perception")
            for subtask in qa_subtasks:
                if subtask in group_name:
                    date_group_name = subtask
                    logger.info(f"QA task detected: '{group_name}' -> extracted subtask: {date_group_name}")
                    break
        else:
            # Handle caption tasks (existing logic)
            date_group_name = "content_long"  # default fallback
            
            # Map group_name to the appropriate GROUP_CONFIGS key
            if "_" in group_name:
                parts = group_name.split("_")
                if len(parts) >= 2:
                    # Extract the subtask part (e.g., "long", "short", "speech", etc.)
                    subtask_part = parts[1]
                    
                    # Map subtask to group_name based on CaptionEvaluationConfig.GROUP_CONFIGS
                    if subtask_part in ["long", "short", "environment"]:
                        # These subtasks don't have pure/mixed variants
                        subtask_to_group_mapping = {
                            "long": "content_long",
                            "short": "content_short",
                            "environment": "environment"
                        }
                        date_group_name = subtask_to_group_mapping.get(subtask_part, "content_long")
                    else:
                        # For speech, music, sound, we need to determine if it's pure or mixed
                        # Look for pure/mixed indicators in the group_name
                        is_pure = True  # Default to pure
                        
                        # Check if group_name contains indicators of mixed content
                        if "mixed" in group_name.lower():
                            is_pure = False
                        elif "pure" in group_name.lower():
                            is_pure = True
                        else:
                            # If no explicit indicator, try to infer from the data
                            mixed_indicators = ["mixed", "both", "combined", "multi"]
                            pure_indicators = ["pure", "only", "single"]
                            
                            group_name_lower = group_name.lower()
                            if any(indicator in group_name_lower for indicator in mixed_indicators):
                                is_pure = False
                            elif any(indicator in group_name_lower for indicator in pure_indicators):
                                is_pure = True
                            else:
                                # Default behavior: assume pure for backward compatibility
                                is_pure = True
                                logger.info(f"No explicit pure/mixed indicator found in group_name '{group_name}', defaulting to pure")
                        
                        # Map to the appropriate group_name
                        if subtask_part == "speech":
                            date_group_name = "pure_speech" if is_pure else "mixed_speech"
                        elif subtask_part == "music":
                            date_group_name = "pure_music" if is_pure else "mixed_music"
                        elif subtask_part == "sound":
                            date_group_name = "pure_sound" if is_pure else "mixed_sound"
                        else:
                            # Fallback for unknown subtasks
                            date_group_name = "content_long"
                        
                        logger.info(f"Caption task: '{group_name}' (subtask: {subtask_part}, is_pure: {is_pure}) mapped to DATE group_name: {date_group_name}")
        # Initialize scores
        overall_scores = {}
        sample_scores = {i: {} for i in range(len(raw_data))}

        # Calculate BLEU scores with PyTorch implementation
        if "bleu" in self.metrics:
            logger.info("Calculating BLEU scores...")
            
            # Initialize PyTorch BLEU scorer
            bleu_scorer = PyTorchBleuScorer(max_n=4)
            
            # Prepare data in the expected format for our BLEU scorer
            gts_for_bleu = {}
            res_for_bleu = {}
            
            for sample_id in gts.keys():
                # Convert ground truth format
                gts_for_bleu[sample_id] = [{'caption': ref['caption']} for ref in gts[sample_id]]
                # Convert result format
                res_for_bleu[sample_id] = [{'caption': res[sample_id][0]['caption']}]
            
            # Compute BLEU scores
            bleu_overall_scores, bleu_individual_scores = bleu_scorer.compute_score(gts_for_bleu, res_for_bleu)
            
            # Store results
            bleu_names = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
            for i, (name, score) in enumerate(zip(bleu_names, bleu_overall_scores)):
                overall_scores[name] = score
                for j, sample_score in enumerate(bleu_individual_scores[i]):
                    sample_scores[j][name] = sample_score
            
            logger.info(f"BLEU scores computed: {[f'{name}: {score:.4f}' for name, score in zip(bleu_names, bleu_overall_scores)]}")

        # Calculate FENSE scores only if requested
        if "fense" in self.metrics and self.fense_scorer is not None:
            logger.info("Calculating FENSE scores...")

            # Prepare batch data for FENSE
            eval_caps = [sample["prediction"] for sample in raw_data]
            ref_caps_list = []
            sample_keys = []
            for sample in raw_data:
                ref_caps = sample["references"] if "references" in sample else sample.get("reference", [])
                if len(ref_caps) == 0:
                    ref_caps = [""]
                ref_caps_list.append(ref_caps)
                # Extract sample key if available
                sample_key = sample.get("id", None)
                sample_keys.append(sample_key)

            # Prepare save paths for FENSE
            fense_save_path = None
            fense_dataframe_path = None
            if save_path is not None:
                fense_save_path = os.path.join(save_path, f"fense")
                fense_dataframe_path = os.path.join(save_path, f"fense_results_{group_name}.csv")

            # Use corpus_score for batch processing
            fense_scores = self.fense_scorer.corpus_score(
                eval_caps, ref_caps_list, agg_score="none", 
                save_path=fense_save_path, save_dataframe_path=fense_dataframe_path,
                sample_keys=sample_keys, group_name=date_group_name
            )

            # Store individual scores
            for i, score in enumerate(fense_scores):
                sample_scores[i]["fense"] = score

            overall_scores["fense"] = sum(fense_scores) / len(fense_scores) if fense_scores else 0.0
            logger.info(f"FENSE: {overall_scores['fense']:.4f}")

        # Calculate DATE scores only if requested
        if "date" in self.metrics and self.date_scorer is not None:
            logger.info("Calculating DATE scores...")
            # Prepare batch data for DATE
            eval_caps = [sample["prediction"] for sample in raw_data]
            ref_caps_list = []
            sample_keys = []
            for sample in raw_data:
                ref_caps = sample["references"] if "references" in sample else sample.get("reference", [])
                if len(ref_caps) == 0:
                    ref_caps = [""]
                ref_caps_list.append(ref_caps)
                # Extract sample key if available
                sample_key = sample.get("id", None)
                sample_keys.append(sample_key)

            # Prepare save paths for DATE
            date_save_path = None
            date_dataframe_path = None
            if save_path is not None:
                date_save_path = os.path.join(save_path, f"date")
                date_dataframe_path = os.path.join(save_path, f"date_results_{group_name}.csv")

            # Use corpus_score for batch processing
            date_scores = self.date_scorer.corpus_score(
                eval_caps, ref_caps_list, agg_score="none",
                save_path=date_save_path, save_dataframe_path=date_dataframe_path,
                sample_keys=sample_keys, group_name=date_group_name
            )

            # Store individual scores
            for i, score in enumerate(date_scores):
                sample_scores[i]["date"] = score

            overall_scores["date"] = sum(date_scores) / len(date_scores) if date_scores else 0.0
            logger.info(f"DATE: {overall_scores['date']:.4f}")

        return overall_scores, sample_scores, raw_data, {}

    def run_group_evaluation(
        self,
        group_name: str,
        gts: Dict[str, List[Dict[str, str]]],
        res: Dict[str, List[Dict[str, str]]],
        raw_data: List[Dict[str, str]],
        headers: List[str],
        output_dir: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], List[Dict[str, str]], Dict]:
        if len(raw_data) == 0:
            logger.info(f"No valid data found for {group_name}")
            # Return empty results with proper structure
            # Expand metrics to match what _evaluate actually produces
            expanded_metrics = []
            for metric in self.metrics:
                if metric == "bleu":
                    expanded_metrics.extend(["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
                elif metric == "fense":
                    expanded_metrics.append("fense")
                elif metric == "date":
                    expanded_metrics.append("date")
                else:
                    expanded_metrics.append(metric)
            
            empty_scores = {metric: 0.0 for metric in expanded_metrics}
            return empty_scores, {}, [], {}
        overall_scores, sample_scores, raw_data, jsonl_data_to_save = self._evaluate(group_name, gts, res, raw_data, save_path)

        if output_dir:
            result_row = [group_name, len(raw_data)]
            for metric in headers[2:]:
                result_row.append(f"{overall_scores.get(metric, 0.0):.4f}")

            output_file = os.path.join(output_dir, f"results_{group_name}.txt")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                f.write(tabulate([result_row], headers=headers, tablefmt="plain"))
            logger.info(f"Results saved to: {output_file}")

            for k in jsonl_data_to_save:
                if jsonl_data_to_save[k] is not None:
                    jsonl_filename = os.path.join(output_dir, f"jsonl_{k}_{group_name}.jsonl")
                    with open(jsonl_filename, "w", encoding="utf-8") as f:
                        for data in jsonl_data_to_save[k]:
                            json_string = json.dumps(data)
                            f.write(json_string + "\n")

        return overall_scores, sample_scores, raw_data, jsonl_data_to_save
