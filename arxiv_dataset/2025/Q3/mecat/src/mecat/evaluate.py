import argparse
import pandas as pd
import numpy as np
import json
import os
import glob
from typing import Dict, List, Literal, Union, Optional, Tuple
from pathlib import Path
from loguru import logger
from .utils import CaptionEvaluationConfig, QAEvaluationConfig
from .scorer import BaseScorer


def load_json(filename: Path | str, task: Optional[str] = None, subtask: Optional[str] = None):
    # parse the filename
    output = {}
    filenames = glob.glob(filename if isinstance(filename, str) else filename.__str__())

    for tmp_filename in filenames:
        with open(tmp_filename, "r") as fr:
            for line in fr:
                try:
                    tmp_json_obj = json.loads(line)
                    for tmp_key, tmp_value in tmp_json_obj.items():
                        fieldname = "answer" if task == "qa" else subtask
                        # skip the samples that are not in the given subtask
                        if task == "qa" and tmp_value["category"] != subtask:
                            continue
                        output[tmp_key] = tmp_value[fieldname]
                except:
                    logger.warning(f"Failed to parse the sample {line}")
                    pass
    return output


def load_csv(filename: Path | str):
    data = pd.read_csv(filename, header=None, quoting=1)
    # If there are more than 2 columns, raise error
    if data.shape[1] > 2:
        raise Exception("CSV file should have only 2 columns (key, value)")
    # Convert to {key: value} with both as string
    output = {str(row[0]): str(row[1]) for row in data.values}
    return output


def load_txt(filename: Path | str):
    """Load data from txt file with two columns (key and value)"""
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split("\t")  # Use tab as default separator
                if len(parts) == 1:  # If no tab, try space
                    parts = line.split(" ", 1)
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    data[key] = value
                else:
                    logger.warning(f"Skipping malformed line: {line}")
    return data


def load_jsonl_simple(filename: Path | str):
    """Load data from jsonl file with simple {key: value} format"""
    data = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    line_data = json.loads(line)
                    # Handle both {key: value} and {key: {field: value}} formats
                    for key, value in line_data.items():
                        if isinstance(value, dict):
                            # If value is a dict, try to extract the first string value
                            for field, field_value in value.items():
                                if isinstance(field_value, str):
                                    data[key] = field_value
                                    break
                        else:
                            data[key] = str(value)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON line: {line}, error: {e}")
                    continue
    return data


def load_prediction_file(file_path: Path | str) -> Dict[str, str]:
    """
    Load prediction data from various file formats
    Args:
        file_path: Path to the prediction file (can be absolute or relative)
    Returns:
        Dict[str, str]: Dictionary with {key: value} format
    """
    # Convert to absolute path using os commands
    if isinstance(file_path, str):
        # Use os.path.abspath to convert relative path to absolute path
        abs_path = os.path.abspath(file_path)
    else:
        abs_path = os.path.abspath(str(file_path))
    
    # Convert back to Path object for consistent handling
    file_path = Path(abs_path)
    
    # Check if file exists
    if not file_path.exists():
        # Provide helpful error message with current working directory
        current_dir = os.getcwd()
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}\n"
            f"Original input: {abs_path}\n"
            f"Current working directory: {current_dir}\n"
            f"Please check the file path and ensure the file exists."
        )

    # Verify the file is actually a file (not a directory)
    if not file_path.is_file():
        raise ValueError(f"Path exists but is not a file: {file_path}")

    # Log successful file loading
    logger.info(f"Loading prediction file: {file_path}")
    
    # Load file based on extension
    file_suffix = file_path.suffix.lower()
    
    try:
        if file_suffix == ".csv":
            return load_csv(file_path)
        elif file_suffix == ".jsonl":
            return load_jsonl_simple(file_path)
        elif file_suffix == ".txt":
            return load_txt(file_path)
        elif file_suffix == ".json":
            # For JSON files, try to load as a single JSON object
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Convert to simple {key: value} format
                    result = {}
                    for key, value in data.items():
                        if isinstance(value, dict):
                            # If value is a dict, try to extract the first string value
                            for field, field_value in value.items():
                                if isinstance(field_value, str):
                                    result[key] = field_value
                                    break
                        else:
                            result[key] = str(value)
                    return result
                else:
                    raise ValueError("JSON file must contain a dictionary object")
        else:
            raise ValueError(f"Unsupported file format: {file_suffix}. "
                           f"Supported formats: .csv, .jsonl, .txt, .json")
    except Exception as e:
        raise RuntimeError(f"Failed to load prediction file '{file_path}': {str(e)}")  


def expand_metrics_for_headers(metrics: List[str]) -> List[str]:
    """
    Expand high-level metric names to their actual output keys
    
    Args:
        metrics: List of high-level metric names (e.g., ['bleu', 'rouge'])
        
    Returns:
        List of actual metric keys that will be produced (e.g., ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'ROUGE_L'])
    """
    expanded = []
    for metric in metrics:
        if metric == "bleu":
            expanded.extend(["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])
        elif metric == "rouge":
            expanded.append("ROUGE_L")
        elif metric == "meteor":
            expanded.append("METEOR")
        elif metric == "cider":
            expanded.append("CIDEr")
        elif metric == "spice":
            expanded.append("SPICE")
        elif metric == "fense":
            expanded.append("fense")
        elif metric == "date":
            expanded.append("date")
        else:
            expanded.append(metric)
    return expanded


def load_data_with_config(
    predicted_data: Dict[str, str],
    task: Literal["caption", "qa"] = "caption",
    subtask: Optional[str] = None,
    level: Optional[str] = None,
    reference_dir: Optional[str] = None,
    force_pure: bool = False,
    force_mixed: bool = False,
    metrics: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
    """
    Load reference data using Config form from utils.py based on task and subtask,
    and match with provided predicted_data
    Args:
        predicted_data: Dict[str, str], the predicted data as {id: prediction}
        task: Literal['caption', 'qa'], the task type
        subtask: Optional[str], the specific field to load (e.g., 'long', 'short', 'direct_perception')
        level: Optional[str], the level for QA task (e.g., 'basic', 'intermediate')
        reference_dir: Optional[str], path to reference directory
        metrics: Optional[List[str]], the metrics to be used (for compatibility, not used in current implementation)
    Returns:
        Tuple: (gts, res, raw_data) in the format expected by scorers
    """
    if task == "caption":
        config = CaptionEvaluationConfig()
        if subtask is None:
            subtask = "long"  # default field

        # Find all groups that match the subtask (both pure and mixed)
        matching_groups = []
        for group_name, group_config in config.GROUP_CONFIGS.items():
            if group_config["subtask"] == subtask:
                matching_groups.append(group_name)

        if not matching_groups:
            raise ValueError(f"Field '{subtask}' not found in caption config")

        # For fields that have both pure and mixed variants, evaluate them separately
        pure_groups = [g for g in matching_groups if g.startswith("pure_")]
        mixed_groups = [g for g in matching_groups if g.startswith("mixed_")]

        if force_pure and pure_groups:
            # Force use pure data
            group_config = config.GROUP_CONFIGS[pure_groups[0]]
            subsets = group_config["subsets"]
            logger.info(f"Force using pure data for field '{subtask}': {pure_groups[0]} with subsets {subsets}")
        elif force_mixed and mixed_groups:
            # Force use mixed data
            group_config = config.GROUP_CONFIGS[mixed_groups[0]]
            subsets = group_config["subsets"]
            logger.info(f"Force using mixed data for field '{subtask}': {mixed_groups[0]} with subsets {subsets}")
        elif pure_groups and mixed_groups:
            # If we have both pure and mixed, we'll handle them separately in the evaluate function
            # For now, use pure data as the main evaluation
            group_config = config.GROUP_CONFIGS[pure_groups[0]]
            subsets = group_config["subsets"]
            logger.info(f"Using pure data for field '{subtask}': {pure_groups[0]} with subsets {subsets}")
        elif pure_groups:
            # Use pure data if available
            group_config = config.GROUP_CONFIGS[pure_groups[0]]
            subsets = group_config["subsets"]
            logger.info(f"Using pure data for field '{subtask}': {pure_groups[0]} with subsets {subsets}")
        elif mixed_groups:
            # Use mixed data if no pure data available
            group_config = config.GROUP_CONFIGS[mixed_groups[0]]
            subsets = group_config["subsets"]
            logger.info(f"Using mixed data for field '{subtask}': {mixed_groups[0]} with subsets {subsets}")
        else:
            # Fallback to any matching group
            group_config = config.GROUP_CONFIGS[matching_groups[0]]
            subsets = group_config["subsets"]
            logger.info(f"Using fallback data for field '{subtask}': {matching_groups[0]} with subsets {subsets}")

    else:  # QA task
        config = QAEvaluationConfig()
        if subtask is None:
            subtask = ['direct_perception', 'sound_characteristics', 'quality_assessment', 'environment_reasoning', 'inference_judgement', 'application_context']  # default field
        if level is None:
            level = ["basic", "intermediate", "advanced", "complex"]
        if not isinstance(subtask, list):
            subtask = [subtask]
        if not isinstance(level, list):
            level = [level]

            

        # Use ALL_SUBSETS from utils
        from .utils import ALL_SUBSETS

        subsets = ALL_SUBSETS

    # Load reference data
    if reference_dir is None:
        reference_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reference", task, "test")

    # Load reference data from files based on task and subtask
    reference_data = {}

    if task == "caption":
        # For caption task, load data for the specified subtask
        for subset in subsets:
            ref_file = os.path.join(reference_dir, f"{subset}.jsonl")
            if os.path.exists(ref_file):
                with open(ref_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            line_data = json.loads(line.strip())
                            for key, value in line_data.items():
                                if subtask in value:
                                    reference_data[key] = value[subtask]
                        except Exception as e:
                            logger.warning(f"Failed to parse line in {ref_file}: {e}")
                            continue
    else:  # QA 
        logger.info(f"hhahah: {level}")
        # For QA task, load data based on category (subtask) and level
        for subset in subsets:
            ref_file = os.path.join(reference_dir, f"{subset}.jsonl")
            if os.path.exists(ref_file):
                logger.info(ref_file)
                with open(ref_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            line_data = json.loads(line.strip())
                            for key, value in line_data.items():
                                # Check if the sample matches the category and level
                                if value.get("category") in subtask and value.get("difficulty") in level:
                                    reference_data[key] = value.get("answer", "")
                        except Exception as e:
                            logger.warning(f"Failed to parse line in {ref_file}: {e}")
                            continue

    # Only keep the samples that are in both predicted and reference data
    common_keys = set(predicted_data.keys()) & set(reference_data.keys())

    filtered_predicted_data = {k: predicted_data[k] for k in common_keys}
    filtered_reference_data = {k: reference_data[k] for k in common_keys}

    logger.info(f"Evaluated samples: {len(common_keys)}/{len(reference_data)}")

    # Prepare the evaluation data format
    # Since FENSE and DATE now use raw_data directly, we can always use the traditional format
    # Traditional metrics require {key: [{'caption': list[str]}]} format

    # Handle variable number of captions properly
    gts = {}
    for k, v in filtered_reference_data.items():
        if isinstance(v, list):
            # Multiple captions: create one dict per caption
            gts[k] = [{"caption": caption} for caption in v]
        else:
            # Single caption: wrap in list with dict
            gts[k] = [{"caption": v}]

    res = {}
    for k, v in filtered_predicted_data.items():
        if isinstance(v, str):
            # Single prediction: wrap in list with dict
            res[k] = [{"caption": v}]
        else:
            # Multiple predictions: create one dict per prediction
            res[k] = [{"caption": pred} for pred in v]

    raw_data = [
        {
            "id": k,
            "prediction": v,
            "references": filtered_reference_data[k]
            if isinstance(filtered_reference_data[k], list)
            else [filtered_reference_data[k]],
        }
        for k, v in filtered_predicted_data.items()
    ]
    return gts, res, raw_data


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """
    Convert evaluate results to DataFrame format

    Args:
        results: Results from evaluate function

    Returns:
        pd.DataFrame: DataFrame with columns: group_name, num_samples, metric1, metric2, ...
    """
    if "fields" not in results:
        # Single field result
        return _single_result_to_dataframe(results)

    # Multiple fields result
    rows = []
    for group_name, field_result in results["fields"].items():
        row = {"group_name": group_name, "num_samples": field_result.get("num_samples", 0)}

        # Add metrics
        overall_scores = field_result.get("overall_scores", {})
        for metric, score in overall_scores.items():
            row[metric] = score

        rows.append(row)

    df = pd.DataFrame(rows)
    
    # Add overall scores when all subtasks are evaluated
    task = results.get("task", "caption")
    group_names = set(results.get("group_names", []))
    
    if task == "caption":
        # Check if all caption subtasks are present
        expected_caption_groups = {
            "content_long", "content_short", "pure_speech", "mixed_speech", 
            "pure_music", "mixed_music", "pure_sound", "mixed_sound", "environment"
        }
        if expected_caption_groups.issubset(group_names):
            # Get all available metrics from the first row
            if not df.empty:
                metrics = [col for col in df.columns if col not in ['group_name', 'num_samples']]
                
                # Create rows for overall scores
                row_weighted = {"group_name": "score_caption", "num_samples": pd.NA}
                
                # Create a series for easier calculation
                df_indexed = df.set_index('group_name')
                
                for metric in metrics:
                    metric_data = df_indexed[metric]
                    
                    content_long = metric_data.get('content_long', 0)
                    content_short = metric_data.get('content_short', 0)
                    environment = metric_data.get('environment', 0)
                    speech_avg = (metric_data.get('pure_speech', 0) + metric_data.get('mixed_speech', 0)) / 2
                    music_avg = (metric_data.get('pure_music', 0) + metric_data.get('mixed_music', 0)) / 2
                    sound_avg = (metric_data.get('pure_sound', 0) + metric_data.get('mixed_sound', 0)) / 2
                    
                    systemtic = (content_long * 0.8 + content_short * 0.2)
                    content_focused = (speech_avg * 0.6 + music_avg * 0.3 + sound_avg * 0.1)
                    content_unrelated = environment
                    overall_weighted = (systemtic * 0.4 + 
                                        content_focused * 0.4 + 
                                        content_unrelated * 0.2)
                    
                    # Add scores to rows
                    row_weighted[metric] = overall_weighted
                
                # Add overall rows to dataframe
                df = pd.concat([df, pd.DataFrame([row_weighted])], ignore_index=True)
    
    elif task == "qa":
        # Check if all QA subtasks are present
        expected_qa_groups = {
            "direct_perception", "sound_characteristics", "quality_assessment",
            "environment_reasoning", "inference_judgement", "application_context"
        }
        if expected_qa_groups.issubset(group_names):
            # Calculate overall-non-weighted for each metric
            if not df.empty:
                metrics = [col for col in df.columns if col not in ['group_name', 'num_samples']]
                
                # Add overall-non-weighted row
                row_overall = {"group_name": "score_qa", "num_samples": pd.NA}
                
                for metric in metrics:
                    # Calculate mean across all QA subtasks
                    qa_scores = df[df['group_name'].isin(expected_qa_groups)][metric]
                    overall_score = qa_scores.mean()
                    row_overall[metric] = overall_score
                
                df = pd.concat([df, pd.DataFrame([row_overall])], ignore_index=True)
    df.rename(columns={'group_name': 'subtask'}, inplace=True)
    return df


def _single_result_to_dataframe(result: Dict) -> pd.DataFrame:
    """Convert single field result to DataFrame"""
    if "pure" in result and "mixed" in result:
        # Pure/mixed result
        rows = []
        for variant in ["pure", "mixed"]:
            variant_result = result[variant]
            row = {
                "group_name": variant_result.get("group_name", f"{variant}_unknown"),
                "num_samples": variant_result.get("num_samples", 0),
            }

            # Add metrics
            overall_scores = variant_result.get("overall_scores", {})
            for metric, score in overall_scores.items():
                row[metric] = score

            rows.append(row)

        return pd.DataFrame(rows)
    else:
        # Standard single result
        row = {"group_name": result.get("group_name", "unknown"), "num_samples": result.get("num_samples", 0)}

        # Add metrics
        overall_scores = result.get("overall_scores", {})
        for metric, score in overall_scores.items():
            row[metric] = score

        return pd.DataFrame([row])





def evaluate(
    predicted_data: Union[
        Dict[str, str], 
        Dict[str, Dict], 
        List[Dict[str, str]], 
        Path, 
        List[Path], 
        str
    ],
    metrics: List[str] = None,
    task: Literal["caption", "qa"] = "caption",
    subtask: Optional[
        Union[
            Literal["long", "short", "speech", "music", "sound", "environment"],
            Literal[
                "direct_perception",
                "sound_characteristics",
                "quality_assessment",
                "environment_reasoning",
                "inference_judgement",
                "application_context",
            ],
            List[
                Union[
                    Literal["long", "short", "speech", "music", "sound", "environment"],
                    Literal[
                        "direct_perception",
                        "sound_characteristics",
                        "quality_assessment",
                        "environment_reasoning",
                        "inference_judgement",
                        "application_context",
                    ],
                ]
            ],
        ]
    ] = None,
    level: Optional[str] = None,
    reference_dir: Optional[str] = None,
    batch_size: int = 32,
    device: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_details: bool = False,  # 新增参数
) -> pd.DataFrame:
    """
    Evaluate the predicted results using Config form data loading
    Args:
        predicted_data: Multiple input formats supported:
            - Dict[str, str]: {id: prediction} dictionary
            - Dict[str, Dict]: {subtask: {id: prediction}} dictionary, keys must match valid subtasks
            - List[Dict[str, str]]: List of [{id: prediction}] dictionaries, evaluated in order: long, short, speech, music, sound, environment
            - Path: Path to single prediction file
            - List[Path]: List of prediction file paths, evaluated in order: long, short, speech, music, sound, environment
            - str: Path string to single prediction file
        metrics: List[str], the metrics to evaluate, None means enable all metrics
        task: Literal['caption', 'qa'], the task type
        subtask: Optional[str or List[str]], the specific field(s) to evaluate:
            - For caption task: 'long', 'short', 'speech', 'music', 'sound', 'environment' or list of these
            - For QA task: 'direct_perception', 'sound_characteristics', 'quality_assessment', 'environment_reasoning', 'inference_judgement', 'application_context' or list of these
            - If None and predicted_data is Dict[str, str] or single file, evaluates all subtasks
        level: Optional[str], the level for QA task (e.g., 'basic', 'intermediate')
        reference_dir: Optional[str], path to reference directory
        batch_size: int, batch size for FENSE metric
        device: str, device for FENSE metric (e.g., 'cpu', 'cuda')
        output_dir: Optional[Union[str, Path]], if provided, save results to this directory and print to screen
    Returns:
        pd.DataFrame: DataFrame with evaluation results
    """
    # Define valid subtask options
    caption_fields = ["long", "short", "speech", "music", "sound", "environment"]
    qa_fields = [
        "direct_perception",
        "sound_characteristics", 
        "quality_assessment",
        "environment_reasoning",
        "inference_judgement",
        "application_context",
    ]
    
    # Process predicted_data input and determine subtasks
    data_dict_list = []
    determined_subtasks = []
    warning_messages = []  # Collect warning messages for final output
    
    if isinstance(predicted_data, dict):
        # Check if it's Dict[str, Dict] format
        first_key = next(iter(predicted_data.keys())) if predicted_data else None
        if first_key and isinstance(predicted_data[first_key], dict):
            # Dict[str, Dict] format - keys should match subtasks
            if task == "caption":
                valid_keys = [key for key in predicted_data.keys() if key in caption_fields]
                if not valid_keys:
                    raise ValueError(f"For Dict[str, Dict] format with caption task, keys must be from {caption_fields}. "
                                   f"Found keys: {list(predicted_data.keys())}")
                determined_subtasks = valid_keys
                for key in valid_keys:
                    data_dict_list.append((key, predicted_data[key]))
            else:  # QA task
                valid_keys = [key for key in predicted_data.keys() if key in qa_fields] 
                if not valid_keys:
                    raise ValueError(f"For Dict[str, Dict] format with qa task, keys must be from {qa_fields}. "
                                   f"Found keys: {list(predicted_data.keys())}")
                determined_subtasks = valid_keys
                for key in valid_keys:
                    data_dict_list.append((key, predicted_data[key]))
        else:
            # Dict[str, str] format
            data_dict_list.append((None, predicted_data))
    elif isinstance(predicted_data, list):
        if all(isinstance(item, dict) for item in predicted_data):
            # List[Dict[str, str]] format
            if task == "caption":
                max_subtasks = len(caption_fields)
                if len(predicted_data) > max_subtasks:
                    warning_msg = f"Provided {len(predicted_data)} dictionaries, but only {max_subtasks} caption subtasks available. Will evaluate first {max_subtasks} items."
                    logger.warning(warning_msg)
                    warning_messages.append(warning_msg)
                    predicted_data = predicted_data[:max_subtasks]
                elif len(predicted_data) < max_subtasks:
                    warning_msg = f"Provided {len(predicted_data)} dictionaries, will evaluate only: {caption_fields[:len(predicted_data)]}"
                    logger.warning(warning_msg)
                    warning_messages.append(warning_msg)
                
                determined_subtasks = caption_fields[:len(predicted_data)]
                for i, data_dict in enumerate(predicted_data):
                    data_dict_list.append((determined_subtasks[i], data_dict))
            else:  # QA task
                max_subtasks = len(qa_fields)
                if len(predicted_data) > max_subtasks:
                    warning_msg = f"Provided {len(predicted_data)} dictionaries, but only {max_subtasks} qa subtasks available. Will evaluate first {max_subtasks} items."
                    logger.warning(warning_msg)
                    warning_messages.append(warning_msg)
                    predicted_data = predicted_data[:max_subtasks]
                elif len(predicted_data) < max_subtasks:
                    warning_msg = f"Provided {len(predicted_data)} dictionaries, will evaluate only: {qa_fields[:len(predicted_data)]}"
                    logger.warning(warning_msg)
                    warning_messages.append(warning_msg)
                
                determined_subtasks = qa_fields[:len(predicted_data)]
                for i, data_dict in enumerate(predicted_data):
                    data_dict_list.append((determined_subtasks[i], data_dict))
        elif all(isinstance(item, (Path, str)) for item in predicted_data):
            # List[Path] format  
            if task == "caption":
                max_subtasks = len(caption_fields)
                if len(predicted_data) > max_subtasks:
                    warning_msg = f"Provided {len(predicted_data)} files, but only {max_subtasks} caption subtasks available. Will evaluate first {max_subtasks} files."
                    logger.warning(warning_msg)
                    warning_messages.append(warning_msg)
                    predicted_data = predicted_data[:max_subtasks]
                elif len(predicted_data) < max_subtasks:
                    warning_msg = f"Provided {len(predicted_data)} files, will evaluate only: {caption_fields[:len(predicted_data)]}"
                    logger.warning(warning_msg)
                    warning_messages.append(warning_msg)
                
                determined_subtasks = caption_fields[:len(predicted_data)]
                for i, file_path in enumerate(predicted_data):
                    file_data = load_prediction_file(file_path)
                    data_dict_list.append((determined_subtasks[i], file_data))
            else:  # QA task
                max_subtasks = len(qa_fields)
                if len(predicted_data) > max_subtasks:
                    warning_msg = f"Provided {len(predicted_data)} files, but only {max_subtasks} qa subtasks available. Will evaluate first {max_subtasks} files."
                    logger.warning(warning_msg)
                    warning_messages.append(warning_msg)
                    predicted_data = predicted_data[:max_subtasks]
                elif len(predicted_data) < max_subtasks:
                    warning_msg = f"Provided {len(predicted_data)} files, will evaluate only: {qa_fields[:len(predicted_data)]}"
                    logger.warning(warning_msg)
                    warning_messages.append(warning_msg)
                
                determined_subtasks = qa_fields[:len(predicted_data)]
                for i, file_path in enumerate(predicted_data):
                    file_data = load_prediction_file(file_path)
                    data_dict_list.append((determined_subtasks[i], file_data))
        else:
            raise TypeError("List elements must be all dictionaries or all file paths")
    elif isinstance(predicted_data, (Path, str)):
        # Single file path
        file_data = load_prediction_file(predicted_data)
        data_dict_list.append((None, file_data))
    else:
        raise TypeError("predicted_data must be a dictionary, list, or file path")
    
    # Check if subtask parameter should be ignored
    should_ignore_subtask = False
    ignore_reason = ""
    
    # Case 1: Multi-Dictionary (Dict[str, Dict])
    if isinstance(predicted_data, dict):
        first_key = next(iter(predicted_data.keys())) if predicted_data else None
        if first_key and isinstance(predicted_data[first_key], dict):
            should_ignore_subtask = True
            ignore_reason = "input is Multi-Dictionary (Dict[str, Dict])"
    
    # Case 2: Multiple files (List[Path] with length > 1)
    elif isinstance(predicted_data, list) and all(isinstance(item, (Path, str)) for item in predicted_data):
        if len(predicted_data) > 1:
            should_ignore_subtask = True
            ignore_reason = f"input contains multiple files ({len(predicted_data)} files)"
    
    # Case 3: List[Dict] with length != 1
    elif isinstance(predicted_data, list) and all(isinstance(item, dict) for item in predicted_data):
        if len(predicted_data) != 1:
            should_ignore_subtask = True
            ignore_reason = f"input is List[Dict] with length {len(predicted_data)} (not 1)"
    
    # Determine final subtasks to evaluate
    if subtask is not None and should_ignore_subtask:
        # User specified subtask but it should be ignored in this case
        warning_msg = f"Ignoring --subtask parameter because {ignore_reason}."
        logger.warning(warning_msg)
        warning_messages.append(warning_msg)
        # Use determined subtasks or default
        if determined_subtasks:
            subtasks = determined_subtasks
        else:
            if task == "caption":
                subtasks = caption_fields
            else:  # QA task
                subtasks = qa_fields
    elif subtask is not None:
        # User explicitly specified subtasks and it's valid to use them
        if not isinstance(subtask, list):
            subtasks = [subtask]
        else:
            subtasks = subtask
    elif determined_subtasks:
        # Subtasks determined from input format
        subtasks = determined_subtasks
    else:
        # Default: evaluate all subtasks for single Dict[str, str] or single file
        if task == "caption":
            subtasks = caption_fields
        else:  # QA task
            subtasks = qa_fields

    # Validate subtasks based on task
    for field in subtasks:
        if task == "caption" and field not in caption_fields:
            raise ValueError(f"Invalid subtask '{field}' for caption task. Valid options: {caption_fields}")
        elif task == "qa" and field not in qa_fields:
            raise ValueError(f"Invalid subtask '{field}' for qa task. Valid options: {qa_fields}")

    # Multiple fields evaluation (including single field)
    all_results = {}
    scorer = BaseScorer(metrics=metrics, batch_size=batch_size, device=device)

    # Create mapping from subtask to data_dict for easy lookup
    subtask_to_data = {}
    if len(data_dict_list) == 1 and data_dict_list[0][0] is None:
        # Single Dict[str, str] or single file - use for all subtasks
        data_dict = data_dict_list[0][1]
        for subtask_name in subtasks:
            subtask_to_data[subtask_name] = data_dict
    else:
        # Multiple data sources - map each to its subtask
        for subtask_name, data_dict in data_dict_list:
            subtask_to_data[subtask_name] = data_dict
        
        # Check if all required subtasks have data
        missing_subtasks = [st for st in subtasks if st not in subtask_to_data]
        if missing_subtasks:
            raise ValueError(f"Missing data for subtasks: {missing_subtasks}. "
                           f"Available data for: {list(subtask_to_data.keys())}")

    for subtask in subtasks:
        if task == "caption":
            config = CaptionEvaluationConfig()
            matching_groups = []
            for group_name, group_config in config.GROUP_CONFIGS.items():
                if group_config["subtask"] == subtask:
                    matching_groups.append(group_name)
            pure_groups = [g for g in matching_groups if g.startswith("pure_")]
            mixed_groups = [g for g in matching_groups if g.startswith("mixed_")]
            if pure_groups and mixed_groups:
                # Evaluate pure data
                pure_gts, pure_res, pure_raw_data = load_data_with_config(
                    predicted_data=subtask_to_data[subtask],
                    task=task,
                    subtask=subtask,
                    level=level,
                    reference_dir=reference_dir,
                    force_pure=True,
                    metrics=metrics,
                )
                pure_group_name = pure_groups[0]
                if pure_raw_data or True:  # always return structure
                    expanded_metrics = expand_metrics_for_headers(metrics or scorer.get_available_metrics())
                    pure_overall_scores, _, _, _ = scorer.run_group_evaluation(
                        group_name=f"{task}_{subtask}_pure_{level or 'default'}",
                        gts=pure_gts,
                        res=pure_res,
                        raw_data=pure_raw_data,
                        headers=["Group", "Samples"] + expanded_metrics,
                        output_dir=None,
                        save_path=output_dir if save_details else None,  # 新增逻辑
                    )
                    all_results[pure_group_name] = {
                        "overall_scores": pure_overall_scores,
                        "num_samples": len(pure_raw_data),
                        "group_name": pure_group_name,
                    }
                # Evaluate mixed data
                mixed_gts, mixed_res, mixed_raw_data = load_data_with_config(
                    predicted_data=subtask_to_data[subtask],
                    task=task,
                    subtask=subtask,
                    level=level,
                    reference_dir=reference_dir,
                    force_mixed=True,
                    metrics=metrics,
                )
                mixed_group_name = mixed_groups[0]
                if mixed_raw_data or True:
                    expanded_metrics = expand_metrics_for_headers(metrics or scorer.get_available_metrics())
                    mixed_overall_scores, _, _, _ = scorer.run_group_evaluation(
                        group_name=f"{task}_{subtask}_mixed_{level or 'default'}",
                        gts=mixed_gts,
                        res=mixed_res,
                        raw_data=mixed_raw_data,
                        headers=["Group", "Samples"] + expanded_metrics,
                        output_dir=None,
                        save_path=output_dir if save_details else None,  # 新增逻辑
                    )
                    all_results[mixed_group_name] = {
                        "overall_scores": mixed_overall_scores,
                        "num_samples": len(mixed_raw_data),
                        "group_name": mixed_group_name,
                    }
            else:
                # Standard field evaluation (no pure/mixed variants)
                gts, res, raw_data = load_data_with_config(
                    predicted_data=subtask_to_data[subtask],
                    task=task,
                    subtask=subtask,
                    level=level,
                    reference_dir=reference_dir,
                    metrics=metrics,
                )
                group_name = matching_groups[0] if matching_groups else subtask
                expanded_metrics = expand_metrics_for_headers(metrics or scorer.get_available_metrics())
                overall_scores, _, _, _ = scorer.run_group_evaluation(
                    group_name=f"{task}_{subtask}_{level or 'default'}",
                    gts=gts,
                    res=res,
                    raw_data=raw_data,
                    headers=["Group", "Samples"] + expanded_metrics,
                    output_dir=None,  # do not output to file
                    save_path=output_dir if save_details else None,  # 新增逻辑
                )
                all_results[group_name] = {
                    "overall_scores": overall_scores,
                    "num_samples": len(raw_data),
                    "group_name": group_name,
                    "level": level,
                }
        else:
            # QA task - standard evaluation
            gts, res, raw_data = load_data_with_config(
                predicted_data=subtask_to_data[subtask],
                task=task,
                subtask=subtask,
                level=level,
                reference_dir=reference_dir,
                metrics=metrics,
            )
            group_name = subtask
            expanded_metrics = expand_metrics_for_headers(metrics or scorer.get_available_metrics())
            overall_scores, _, _, _ = scorer.run_group_evaluation(
                group_name=f"{task}_{subtask}_{level or 'default'}",
                gts=gts,
                res=res,
                raw_data=raw_data,
                headers=["Group", "Samples"] + expanded_metrics,
                output_dir=None,  # do not output to file
                save_path=output_dir if save_details else None,  # 新增逻辑
            )
            all_results[group_name] = {
                "overall_scores": overall_scores,
                "num_samples": len(raw_data),
                "group_name": group_name,
                "level": level,
            }
    results = {"fields": all_results, "group_names": list(all_results.keys()), "task": task, "level": level}
    
    # Convert results to DataFrame
    df_results = results_to_dataframe(results)
    
    # Display warning messages before final output
    if warning_messages:
        print("\n" + "="*50)
        print("⚠️  WARNINGS:")
        for warning in warning_messages:
            print(f"  • {warning}")
        print("="*50)
    
    # Handle output directory if provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results to file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"evaluation_results_{timestamp}.csv"
        df_results.to_csv(output_file, index=False)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Print results to screen
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        # Multiply results by 100 and format to 1 decimal place
        df_display = df_results.copy()
        numeric_columns = df_display.select_dtypes(include=[np.number]).columns
        df_display[numeric_columns] = df_display[numeric_columns] * 100
        print(df_display.to_string(index=False, float_format='%.1f'))
        print("="*50)
    
    return df_results


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description="Evaluate model predictions using Config form")
    parser.add_argument(
        "--prediction",
        type=str,
        nargs="+",
        required=True,
        help="Path to prediction file(s) (CSV, JSONL, TXT, JSON) or JSON string of {id: prediction}. "
             "Multiple files will be evaluated in order: long, short, speech, music, sound, environment",
    )
    parser.add_argument("--task", choices=["caption", "qa"], default="caption", help="Task type: caption or qa")
    parser.add_argument(
        "--subtask",
        type=str,
        default=None,
        help="Field name: for caption (long, short, speech, music, sound, environment), "
        "for qa (direct_perception, sound_characteristics, environment_reasoning, inference_judgement, application_context)",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["basic", "intermediate", "advanced", "complex"],
        default=None,
        help="Level for QA task: basic, intermediate, advanced, complex",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        required=True,
        help="Metrics to evaluate: bleu, rouge, meteor, cider, spice, fense, date",
    )
    parser.add_argument("--reference_dir", type=str, default=None, help="Path to reference directory")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results (optional)")
    parser.add_argument("--save_details", action="store_true", help="Save detailed sample-level dataframes and embeddings to output_dir")
    args = parser.parse_args()

    # Parse prediction data
    predicted_data = args.prediction
    
    # Handle different input formats
    if len(predicted_data) == 1:
        # Single input - could be file path or JSON string
        single_input = predicted_data[0]
        if single_input.startswith("{") and single_input.endswith("}"):
            # Try to parse as JSON string
            try:
                import json
                predicted_data = json.loads(single_input)
            except json.JSONDecodeError:
                # If not valid JSON, treat as file path
                predicted_data = single_input
        else:
            # Single file path
            predicted_data = single_input
    # else: keep as list for multiple files

    # evaluate the predictions
    results = evaluate(
        predicted_data=predicted_data,
        task=args.task,
        subtask=args.subtask,
        level=args.level,
        metrics=args.metrics,
        reference_dir=args.reference_dir,
        output_dir=args.output_dir,
        save_details=args.save_details, # Use the save_details flag
    )

    # If no output_dir was provided, print results to screen
    if args.output_dir is None:
        print("\nEVALUATION RESULTS:")
        print("="*50)
        # Multiply results by 100 and format to 1 decimal place
        results_display = results.copy()
        numeric_columns = results_display.select_dtypes(include=[np.number]).columns
        results_display[numeric_columns] = results_display[numeric_columns] * 100
        print(results_display.to_string(index=False, float_format='%.1f'))


if __name__ == "__main__":
    main()
