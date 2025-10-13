import argparse
import json
from openai import OpenAI 
import os
import datetime
import re

client = OpenAI(api_key =os.getenv("OPENAI_API_KEY")) 


system_prompt = """You are an expert in fact-checking with extensive knowledge of verifying the accuracy of claims based on evidence from various sources. Your task is to analyze the provided claim, the corresponding evidence, and the date when the claim was made. When checking the evidence, consider the timeline and remember that events or information not available by the date of the claim should not be taken into account. Assess whether the evidence supports, rejects, or does not provide enough evidence regarding the claim."""

user_prompt_url = f"""Carefully review the content of the provided URL and determine if it supports, rejects, or does not provide enough evidence to substantiate the claim, taking into account the date when the claim was made. \nRespond strictly with one of the following options only: \n- Support \n- Reject \n- Not enough evidence \nDo not provide any additional text or explanation."""

user_prompt_text_noJustification = f"""Carefully review the content of the provided text and determine if it supports, rejects, or does not provide enough evidence to substantiate the claim, taking into account the date when the claim was made. \nRespond strictly with one of the following options only: \n- Support \n- Reject \n- Not enough evidence \nDo not provide any additional text or explanation."""

user_prompt_text_wJustification = f"""Please carefully evaluate the provided text in relation to the claim. Consider the date the claim was made, and assess whether the content supports, rejects, or provides insufficient evidence to substantiate the claim. \nProvide your reasoning in up to 500 words, and then give your conclusion by selecting one of the following options: \n- Support \n- Reject \n- Not enough evidence\n\nYou may start your conclusion with a prefix: 'Final answer: ' to clearly separate it from your reasoning."""


def convert_to_batch_jsonl(folder, input_file, output_file, type):

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    id_mapping_folder = os.path.join(folder, 'id_mapping')
    if not os.path.exists(id_mapping_folder):
        os.mkdir(id_mapping_folder)
    mapping_file_name = str(type)+"_custom_id_mapping.json"
    custom_id_path = os.path.join(id_mapping_folder, mapping_file_name)
    custom_id_mapping = {}

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for index, item in enumerate(data):

            claim = item.get("claim", "Unknown claim")
            evidence_list = item.get("evidence_url", [])
            # claim_date = item.get("claim_date", "Unknown") #for claims from averitec
            claim_date = item.get("review_date", "Unknown") #for claims from factcheckqa
            claim_date = claim_date if claim_date else "Unknown"


            for evidence_index, evidence in enumerate(evidence_list):
                custom_id = f"{type}-request-{index+1}-{evidence_index+1}"
                evidence_url = evidence.get("original_link", "")
                evidence_content = evidence.get("content", "")
                evidence_content = re.sub(r'[\x00-\x1F\x7F-\x9F\uFFFD]', '', evidence_content)

                custom_id_mapping[custom_id] = {"claim": claim, "evidence_url": evidence_url}

                if type == "url":
                    user_prompt = user_prompt_url + f"""\nClaim: {claim} \nDate of Claim: {claim_date} \nURL: {evidence_url}"""

                elif type == "text_noJustification":
                    if evidence_content == "":
                        continue
                    user_prompt = user_prompt_text_noJustification + f"""\nClaim: {claim} \n Date when the claim was made: {claim_date} \n Scraped Content: {evidence_content}"""

                elif type == "text_wJustification":
                    if evidence_content == "":
                        continue
                    user_prompt = user_prompt_text_wJustification + f"""\nClaim: {claim} \n Date when the claim was made: {claim_date} \n Scraped Content: {evidence_content}"""

                else:
                    raise ValueError("Invalid type. Choose from 'url', 'text_noJustification', or 'text_wJustification'")
                    
                body = {
                    # "model": "gpt-4-turbo",
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }

                jsonl_line = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }


                f_out.write(json.dumps(jsonl_line) + '\n')

    with open(custom_id_path, 'w') as f:
        json.dump(custom_id_mapping, f)


def process_batch_job(client, file_name):
    try:
        batch_input_file = client.files.create(
            file=open(file_name, "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id

        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch_job.id

    except Exception as e:
        print(f"An error occurred: {e}")


def main():


    parser = argparse.ArgumentParser(description='Process a file of claims and search for related information.')
    parser.add_argument('--input', type=str, help='The input file containing claims and evidence')
    parser.add_argument('--type', type=str, choices=['url', 'text_noJustification', 'text_wJustification', 'majority_vote'], help="Choose from 'url', 'text_noJustification', 'text_wJustification' or 'majority_vote'")
    args = parser.parse_args()
    
    folder = 'batch_submission'+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")

    if not os.path.exists(folder):
        os.mkdir(folder)

    batch_id_file = os.path.join(folder, 'batch_ids.json')

    if not os.path.exists(batch_id_file):
        with open(batch_id_file, 'w') as f:
            json.dump({}, f)

    with open(batch_id_file, 'r') as f:
        batch_ids = json.load(f)

    if args.type not in ["url", "text_noJustification", "text_wJustification", "majority_vote"]:
        raise argparse.ArgumentTypeError("Invalid type. Choose from 'url', 'text_noJustification', 'text_wJustification' or 'majority_vote'")

    input_file = args.input

    url_batch = os.path.join(folder, 'url.jsonl')
    txtNoJus_batch = os.path.join(folder, 'txtNoJus.jsonl')
    txtWJus_batch = os.path.join(folder, 'txtWJus.jsonl')


    batch_id = {'url_batch_id': None, 'txtNoJust_batch_id': None, 'txtWJus_batch_id': None}

    with open(input_file, "r") as f:
        data = json.load(f)

    if args.type == "majority_vote":

        convert_to_batch_jsonl(folder, input_file, url_batch, "url")
        convert_to_batch_jsonl(folder, input_file, txtNoJus_batch, "text_noJustification")
        convert_to_batch_jsonl(folder, input_file, txtWJus_batch, "text_wJustification")

        batch_id["url_batch_id"] = process_batch_job(client, url_batch)
        batch_id["txtNoJust_batch_id"] = process_batch_job(client, txtNoJus_batch)
        batch_id["txtWJus_batch_id"] = process_batch_job(client, txtWJus_batch)

    elif args.type == "url":
        convert_to_batch_jsonl(folder, input_file, url_batch, "url")
        batch_id["url_batch_id"] = process_batch_job(client, url_batch)

    elif args.type == "text_noJustification":
        convert_to_batch_jsonl(folder, input_file, txtNoJus_batch, "text_noJustification")
        batch_id["txtNoJust_batch_id"] = process_batch_job(client, txtNoJus_batch)

    elif args.type == "text_wJustification":
        convert_to_batch_jsonl(folder, input_file, txtWJus_batch, "text_wJustification")
        batch_id["txtWJus_batch_id"] = process_batch_job(client, txtWJus_batch)


    print("submitted batch id:", batch_id)
    batch_ids[datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")] = batch_id
    with open(batch_id_file, 'w') as f:
        json.dump(batch_ids, f)



if __name__ == "__main__":
    main()
