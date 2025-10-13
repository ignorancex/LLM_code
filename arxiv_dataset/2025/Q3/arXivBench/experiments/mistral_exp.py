from mistralai import Mistral
import os
from tqdm import tqdm
import pandas as pd
import time
import fire
import re
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from utilities.tools import calculate_ACC, process_prompt, run_in_parallel_collecting_evals, find_ids_info, initialize_files


class MistralQuerier:
    def __init__(self, client):
        assert(client != None)
        self.client = client

    def query(self, prompt, model):

        valid = False
        messages=[
            {
                "role": "user",
                "content": f"""{prompt}     Only list 3 papers with complete title and explict explicit arxiv url.
                            Place titles and links insides [[]], like [[title]] [[url]].
                            Separate each paper by a newline.
                            Do not include any additional commentary or explanations."""

            }
        ]

        mixtral_completion = self.client.chat.complete(
            messages=messages,
            model=model,
            temperature=0.0,
            random_seed=42,
            max_tokens=1000,
        )
        response = mixtral_completion.choices[0].message.content


        prompt_pattern = r"\[\[(.+?)\]\]\s*(?:\[\[|\[)?(https?://[^\s\[\]]+)(?:\]\])?"
        matches = re.findall(prompt_pattern, response)

        titles = list()
        urls = list()
        addtional_prompt_info = list()

        # check whether the response is in the correct format
        if len(matches) != 3:
            valid = False
            return valid, None, None, None, response
        
        for match in matches:
            title, url = match

            # Check whether it indicates it's not available
            if (not  url.lower().startswith("http")) or (not "abs/" in  url):
                continue
            else: 
                titles.append(title.strip())
                urls.append(url.strip())
                addtional_prompt_info.append(prompt)
        
        valid = True
        return valid, titles, urls, addtional_prompt_info, response





def main(input_file, model, json_file_path, output_dir=None, max_workers=5, saved_output=None):
    assert(input_file != None)

    mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    file_prompts = pd.read_csv(input_file)['prompt'].to_list()


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_output_filename = f"{output_dir}/valid_output.csv"
    invalid_output_filename = f"{output_dir}/invalid_output.csv"
    hallucinations_file = f"{output_dir}/hallucinations.csv"
    correct_file = f"{output_dir}/correct.csv"
    stats_output_filename = f"{output_dir}/evaluations.csv"
    initialize_files(invalid_output_filename, hallucinations_file, correct_file, stats_output_filename)


    # If we don't use the saved model's outputs
    if not saved_output:
        # Firstly, get all papers' titles and ids
        total_titles = []
        total_ids = []
        total_urls = []
        total_prompts = []

        querier = MistralQuerier(client=mistral_client)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_prompt, querier, prompt, model,
                    valid_output_filename, invalid_output_filename
                )
                for prompt in file_prompts
            ] 
            for future in tqdm(as_completed(futures), total=len(file_prompts)):
                result = future.result()
                if 'titles' in result:
                    total_titles.extend(result['titles'])
                    total_ids.extend(result['ids'])
                    total_urls.extend(result["urls"])
                    total_prompts.extend(result['prompts'])

        # Write the LLM's response to the valid_output_file:
        with open(valid_output_filename, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "prompt", "title", "url"])
            for i in range(len(total_titles)):
                writer.writerow([total_ids[i], total_prompts[i], total_titles[i], total_urls[i]])
    
    # Use saved model's outputs
    else:
        print("Reading from "+saved_output)
        df = pd.read_csv(saved_output)
        total_titles = df['title'].tolist()
        total_ids = df['id'].tolist()
        total_urls = df['url'].tolist()
        total_prompts = df['prompt'].tolist()


    ids_info = find_ids_info(json_file_path, total_ids)
    evaluations = run_in_parallel_collecting_evals(total_titles, total_ids, total_urls, total_prompts, ids_info, hallucinations_file, correct_file, max(max_workers, 20))


    if(len(evaluations) != 0):
        ACC, total_valid_responses= calculate_ACC(evals=evaluations)
        with open(stats_output_filename, mode="a", newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model, total_valid_responses, ACC])

            


if __name__ == "__main__":
    fire.Fire(main)