import os
import json
from typing import List, Dict, Tuple
from loguru import logger

ALL_SUBSETS = ["000", "00A", "0M0", "0MA", "S00", "S0A", "SM0", "SMA"]

MIXED_SUBSETS = ["SMA", "SM0", "S0A", "0MA"]


class CaptionEvaluationConfig:
    HEADERS = ["evaluation_group", "samples", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "FENSE"]

    GROUP_CONFIGS = {
        "content_long": {"subsets": ALL_SUBSETS, "subtask": "long", "weight_type": "content"},
        "content_short": {"subsets": ALL_SUBSETS, "subtask": "short", "weight_type": "content"},
        "pure_speech": {"subsets": ["S00"], "subtask": "speech", "weight_type": "speech"},
        "pure_sound": {"subsets": ["00A"], "subtask": "sound", "weight_type": "sound"},
        "pure_music": {"subsets": ["0M0"], "subtask": "music", "weight_type": "music"},
        "mixed_speech": {"subsets": MIXED_SUBSETS, "subtask": "speech", "weight_type": "speech"},
        "mixed_sound": {"subsets": MIXED_SUBSETS, "subtask": "sound", "weight_type": "sound"},
        "mixed_music": {"subsets": MIXED_SUBSETS, "subtask": "music", "weight_type": "music"},
        "environment": {"subsets": ALL_SUBSETS, "subtask": "environment", "weight_type": None},
    }

    @classmethod
    def get_evaluation_groups(cls):
        groups = []
        for group_name, config in cls.GROUP_CONFIGS.items():
            # group_name, subset, subtask, level(None for caption)
            groups.append((group_name, config["subsets"], config["subtask"], None))
        return groups


class QAEvaluationConfig:
    HEADERS = ["evaluation_group", "samples", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L", "FENSE"]

    FIELDS = [
        "direct_perception",
        "sound_characteristics",
        "quality_assessment",
        "environment_reasoning",
        "inference_judgement",
        "application_context",
    ]
    # FIELDS = [None]
    LEVELS = ["basic", "intermediate", "advanced", "complex"]

    @classmethod
    def get_evaluation_groups(cls):
        groups = []
        for field in cls.FIELDS:
            for level in cls.LEVELS:
                group_name = f"{field}:{level}"
                groups.append((group_name, ALL_SUBSETS, field, level))
        return groups


class BaseDataProcessor:
    def __init__(self, reference_dir: str, prediction_dir: str) -> None:
        self.reference_dir = reference_dir
        self.prediction_dir = prediction_dir

    def load_jsonl_data(self, input_file: str) -> Dict[str, Dict[str, str]]:
        data = {}
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    line_data = json.loads(line.strip())
                    data.update(line_data)
                except Exception:
                    pass
        return data

    def _validate_files(self, subset: List[str]) -> bool:
        ref_file = os.path.join(self.reference_dir, f"{subset}.jsonl")
        pred_file = os.path.join(self.prediction_dir, f"{subset}.jsonl")

        if not os.path.exists(ref_file):
            logger.info(f"Warning: Missing reference files for subset {subset}")
            return False
        if not os.path.exists(pred_file):
            logger.info(f"Warning: Missing prediction files for subset {subset}")
            return False
        return True


class CaptionDataProcessor(BaseDataProcessor):
    def caption_jsonl_to_eval_format(
        self, reference_file: str, prediction_file: str, subtask: str
    ) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        ref_data = self.load_jsonl_data(reference_file)
        pred_data = self.load_jsonl_data(prediction_file)
        gts = {}
        res = {}
        raw_data = []
        for utt_id in ref_data.keys():
            # for utt_id in list(ref_data.keys())[:10]:
            if utt_id not in pred_data:
                logger.info(f"Warning: Missing prediction of: {utt_id}")
                continue

            ref_item = ref_data[utt_id]
            pred_item = pred_data[utt_id]
            ref_field = ref_item[subtask]
            pred_field = pred_item[subtask]

            if isinstance(pred_field, str):
                prediction = pred_field
            else:
                prediction = pred_field[-1]

            if isinstance(ref_field, str):
                ref_field = [ref_field]
            else:
                ref_field = ref_field

            res[utt_id] = [{"caption": prediction}]
            gts[utt_id] = [{"caption": ref} for ref in ref_field]
            raw_data.append({"id": utt_id, "prediction": prediction, "references": ref_field})

        return gts, res, raw_data

    def combine_subsets_data(
        self, subsets: str, subtask: str, level: str
    ) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        combined_gts = {}
        combined_res = {}
        combined_raw_data = []

        for subset in subsets:
            if not self._validate_files(subset):
                continue

            gts, res, raw_data = self._load_subset_data(subset, subtask)

            for utt_id in list(gts.keys()):
                new_id = f"{subset}_{utt_id}"
                combined_gts[new_id] = gts[utt_id]
                combined_res[new_id] = res[utt_id]

            for data in raw_data:
                data["id"] = f"{subset}_{data['id']}"
                combined_raw_data.append(data)
        return combined_gts, combined_res, combined_raw_data

    def _load_subset_data(
        self, subset: str, subtask: str
    ) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        ref_file = os.path.join(self.reference_dir, f"{subset}.jsonl")
        pred_file = os.path.join(self.prediction_dir, f"{subset}.jsonl")
        return self.caption_jsonl_to_eval_format(ref_file, pred_file, subtask)


class QADataProcessor(BaseDataProcessor):
    def qa_jsonl_to_eval_format(
        self,
        reference_file: str,
        prediction_file: str,
        category_field: str,
        category_value: str,
        level_field: str,
        level_value: str,
    ) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        ref_data = self.load_jsonl_data(reference_file)
        pred_data = self.load_jsonl_data(prediction_file)

        gts = {}
        res = {}
        raw_data = []

        for utt_id in ref_data.keys():
            if utt_id not in pred_data:
                logger.info(f"Warning: Missing prediction of: {utt_id}")
                continue

            ref_item = ref_data[utt_id]
            pred_item = pred_data[utt_id]

            if category_field and category_value:
                if ref_item.get(category_field) != category_value:
                    continue

            if level_field and level_value:
                if ref_item.get(level_field) != level_value:
                    continue

            ref_answer = ref_item.get("answer", "")
            pred_answer = pred_item.get("answer", pred_item.get("response", ""))

            if isinstance(pred_answer, str):
                prediction = pred_answer
            else:
                prediction = pred_answer[0]

            if isinstance(ref_answer, str):
                ref_field = [ref_answer]
            else:
                ref_field = ref_answer

            res[utt_id] = [{"caption": prediction}]
            gts[utt_id] = [{"caption": ref} for ref in ref_field]
            raw_data.append(
                {
                    "id": utt_id,
                    "question": ref_item.get("question", ""),
                    "prediction": prediction,
                    "reference": ref_field,
                    "category": ref_item.get("category", ""),
                    "difficulty": ref_item.get("difficulty", ""),
                    "domain": ref_item.get("domain", ""),
                }
            )

        return gts, res, raw_data

    def combine_subsets_data(
        self, subsets: str, subtask: str, level: str
    ) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        combined_gts = {}
        combined_res = {}
        combined_raw_data = []

        for subset in subsets:
            if not self._validate_files(subset):
                continue

            gts, res, raw_data = self._load_subset_data(subset, subtask, level)
            for utt_id in list(gts.keys()):
                new_id = f"{subset}_{utt_id}"
                combined_gts[new_id] = gts[utt_id]
                combined_res[new_id] = res[utt_id]

            for data in raw_data:
                data["id"] = f"{subset}_{data['id']}"
                combined_raw_data.append(data)
        return combined_gts, combined_res, combined_raw_data

    def _load_subset_data(
        self, subset: str, subtask: str, level: str
    ) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]], List[Dict[str, str]]]:
        ref_file = os.path.join(self.reference_dir, f"{subset}.jsonl")
        pred_file = os.path.join(self.prediction_dir, f"{subset}.jsonl")
        return self.qa_jsonl_to_eval_format(ref_file, pred_file, "category", subtask, "difficulty", level)


class FilteredStream:
    """Filter stream to only show errors and warnings"""

    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = ""

    def write(self, text):
        # Keep error and warning messages
        if any(keyword in text.lower() for keyword in ["error", "warning", "exception", "traceback"]):
            self.original_stream.write(text)
        # Buffer other output (including PTBTokenizer output)
        else:
            self.buffer += text

    def flush(self):
        self.original_stream.flush()

    def getvalue(self):
        return self.buffer

    def __getattr__(self, name):
        # Delegate any other attributes to the original stream
        return getattr(self.original_stream, name)


class SilentStream:
    """Completely silent stream that only shows errors and warnings"""

    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, text):
        # Only show error and warning messages
        if any(keyword in text.lower() for keyword in ["error", "warning", "exception", "traceback"]):
            self.original_stream.write(text)
        # Completely ignore all other output

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        # Delegate any other attributes to the original stream
        return getattr(self.original_stream, name)


# check the java environment
def check_java_available():
    """Check if Java is available in the system"""
    import subprocess

    try:
        # Try to run 'java -version' command
        result = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False
