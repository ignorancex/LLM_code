import argparse
import json
from openai import OpenAI 
import os
import re
from collections import Counter

client = OpenAI(api_key = os.getenv("OPENAI_API_KEY")) 

def retrieve_results(batch_job_id, result_file_name):
    batch_job2 = client.batches.retrieve(batch_job_id)

    status = batch_job2.status
    print(batch_job_id, status)
    if status == "completed":
        output_file_id = batch_job2.output_file_id

        if output_file_id:
            outputs = client.files.content(output_file_id).content
            
            with open(result_file_name, 'wb') as file:
                file.write(outputs)
            print(f"Output written to {result_file_name}.")
        else:
            print("No output file found.")
    else:
        print(f"Batch job status: {status}. Not completed yet.")
    
def extract_result(output):

    output = output.lower()
    match = re.match(r'[-\s]*(support|reject|not enough evidence)', output.strip())
    if match:
        return match.group(1).strip().lower()
    else:
        return None
    
def parse_batch_result(result_file_name):
    results = {}
    with open(result_file_name, 'r') as file:
        for line in file:
            output = json.loads(line.strip())
            custom_id = output['custom_id']
            gpt_response = output["response"]["body"]["choices"][0]["message"]["content"].lower()
            
            response = extract_result(gpt_response)
            
            results[custom_id] = response
    return results


def convert_format(mapping_file):
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    new = {}
    for id in mapping:
        claim = mapping[id]['claim']
        url = mapping[id]['evidence_url']
        key = (claim, url) 

        new[key] = {'claim': claim, 'custom_id': id}
    return new

def match_results(data, id_mapping, results, type):
    for item in data:
        claim = item['claim']
        for evidence in item['evidence_url']:
            url = evidence['original_link']

            mapping = id_mapping.get((claim, url), "")
            if mapping != "":
                    custom_id = mapping['custom_id']
                    response = results.get(custom_id,"")
                    evidence[type+ "_llm_check_result"] = response if response else "error occurred in generation"
            else:
                print("error matching", (claim, url))
                continue
    return data


def majority_vote(data):
    results = [] #
    for item in data:
        votes = [] #
        for evidence in item['evidence_url']:
            vote = []
            result1 = evidence['url_llm_check_result']
            vote.append(result1)
            if evidence['content']:

                result2 = evidence['text_noJustification_llm_check_result']
                result3 = evidence['text_wJustification_llm_check_result']

                vote.append(result2)
                vote.append(result3)

            vote_counts = Counter(vote)
            if vote_counts:
                most_common_votes = vote_counts.most_common()
                # Check if there's a tie
                if len(most_common_votes) > 1 and most_common_votes[0][1] == most_common_votes[1][1]:
                    # There's a tie, return 'not enough evidence'
                    majority_vote = 'not enough evidence'
                else:
                    majority_vote = most_common_votes[0][0]
            else:
                majority_vote = None

            evidence['majority_vote'] = majority_vote

            votes.append(majority_vote)
        results.append({'claim': item['claim'], 'votes': votes})
            
    return data, results



def count_conflicts(results):
    has_conflicting = []

    for result in results:
        evidence_result = result['votes']
        if 'support' in evidence_result and 'reject' in evidence_result:
            has_conflicting.append(result['claim'])
    return len(has_conflicting)


def main():
    parser = argparse.ArgumentParser(description='Process a file of claims and search for related information.')
    parser.add_argument('--source', type=str, help='The source file containing claims with evidence')
    parser.add_argument('--folder', type=str, help='The folder containing batch submission files')
    parser.add_argument('--submission_time', type=str, help='time of submission')


    args = parser.parse_args()

    folder = args.folder
    print("now parsing folder: ", folder)
    batch_id_file = os.path.join(folder, 'batch_ids.json')


    with open(batch_id_file, 'r') as f:
        batch_ids = json.load(f)

    batch_id = batch_ids[str(args.submission_time)]

    with open(args.source, 'r') as f:
        data = json.load(f)

    url_batch_id = batch_id.get("url_batch_id", "")
    txtNoJust_batch_id = batch_id.get("txtNoJust_batch_id", "")
    txtWJus_batch_id = batch_id.get("txtWJus_batch_id", "")

    url_results = os.path.join(folder, 'url_results.jsonl')
    txtNoJus_results = os.path.join(folder, 'txtNoJus_results.jsonl')
    txtWJus_results = os.path.join(folder, 'txtWJus_results.jsonl')


    if url_batch_id:

        retrieve_results(url_batch_id, url_results)
        url_results_ls = parse_batch_result(url_results)
        url_mapping = convert_format(os.path.join(folder, 'id_mapping/url_custom_id_mapping.json'))
        data_new = match_results(data, url_mapping, url_results_ls, 'url')


    if txtNoJust_batch_id:

        retrieve_results(txtNoJust_batch_id, txtNoJus_results)
        txtNoJus_results_ls = parse_batch_result(txtNoJus_results)
        textNoJust_mapping = convert_format(os.path.join(folder, 'id_mapping/text_noJustification_custom_id_mapping.json'))
        data_new = match_results(data_new, textNoJust_mapping, txtNoJus_results_ls, 'text_noJustification')


    if txtWJus_batch_id:
        retrieve_results(txtWJus_batch_id, txtWJus_results)
        txtWJus_results_ls = parse_batch_result(txtWJus_results)
        textWJust_mapping = convert_format(os.path.join(folder, 'id_mapping/text_wJustification_custom_id_mapping.json'))
        data_new = match_results(data_new, textWJust_mapping, txtWJus_results_ls, 'text_wJustification')

    if url_batch_id and txtNoJust_batch_id and txtWJus_batch_id:
        data_new, results = majority_vote(data_new)
        print("number of claims having conflicting evidence: ", count_conflicts(results))


    with open(os.path.join(folder, 'results.json'), "w") as f:
        json.dump(data_new, f, indent=4)
        print(f"Results saved to {os.path.join(folder, 'results.json')}")

if __name__=='__main__':
    main()
