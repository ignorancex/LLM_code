import dotenv

dotenv.load_dotenv()

import json
from pathlib import Path

import click
import datasets


def load_curator_parsed_response_message(input_curator_responses_jsonl):
    data = []
    with open(input_curator_responses_jsonl) as f:
        for line in f.readlines():
            sample = json.loads(line)
            data.extend(sample["parsed_response_message"])
    return data


@click.command()
@click.option("--input_curator_responses_jsonl", type=str, required=True)
@click.option("--upload_repo_id", type=str, required=True)
def main(input_curator_responses_jsonl=None, upload_repo_id=None):
    print(f"Uploading {input_curator_responses_jsonl} to {upload_repo_id}")

    data_list = load_curator_parsed_response_message(input_curator_responses_jsonl)

    dataset = datasets.Dataset.from_list(data_list)

    dataset.push_to_hub(upload_repo_id)


if __name__ == "__main__":
    main()
