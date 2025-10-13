import dotenv

dotenv.load_dotenv()

import json
import re
from pathlib import Path

import datasets
import tqdm


def main():
    cot_jsonl_path = "outputs/med-vlm-pmc_vqa-gpt-4o-cot/eval_results.jsonl"
    cot_jsonl_path = Path(cot_jsonl_path)

    dataset_index2reasoning = {}
    with open(cot_jsonl_path, "r") as f:
        for line in tqdm.tqdm(f):
            data = json.loads(line)
            dataset_index = data["dataset_index"]

            # get correct cot
            output_text = None
            for prased_output in data["parsed_outputs"]:
                if prased_output["is_correct"]:
                    output_text = prased_output["output_text"]
                    break

            reasoning = None
            if output_text is not None:
                # extract the content in <think> </think> tags
                reasoning_search = re.search(
                    r"<think>(.*?)</think>", output_text, re.DOTALL
                )
                if reasoning_search:
                    reasoning = reasoning_search.group(1).strip()

            if reasoning is not None:
                dataset_index2reasoning[dataset_index] = reasoning

    dataset_path = "med-vlrm/med-vlm-pmc_vqa"
    split = "train"

    dataset = datasets.load_dataset(dataset_path, split=split)

    len_dataset = len(dataset)
    len_cot = len(dataset_index2reasoning)
    print(f"Dataset length: {len_dataset}, COT length: {len_cot}")

    def add_reasoning(sample):
        dataset_index = sample["dataset_index"]
        if dataset_index in dataset_index2reasoning:
            sample["reasoning"] = dataset_index2reasoning[dataset_index]
        else:
            sample["reasoning"] = None
        return sample

    num_proc = 16
    dataset = dataset.map(add_reasoning, num_proc=num_proc)
    upload_dataset_path = "med-vlrm/med-vlm-pmc_vqa-gpt_4o_reasoning"
    dataset.push_to_hub(upload_dataset_path)

    def filter_reasoning(sample):
        return sample["reasoning"] is not None

    print(f"Filtering dataset ({len(dataset)}) with reasoning...")
    dataset = dataset.filter(filter_reasoning, num_proc=num_proc)
    print(f"Filtered dataset length: {len(dataset)}")
    upload_dataset_path = "med-vlrm/med-vlm-pmc_vqa-gpt_4o_reasoning-filter_none_cot"
    dataset.push_to_hub(upload_dataset_path)


if __name__ == "__main__":
    main()
