import requests
import os
import json
import csv
from bs4 import BeautifulSoup
from openai import OpenAI
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

'''
def compare_results(title, url, prompt=None):
    arxiv_title, arxiv_abs, fabricated = scrape_arxiv_metadata(url)

    if fabricated:
        return None, fabricated
    
    # early return for empty title and url
    if arxiv_title == None or arxiv_abs == None:
        return None, fabricated

    paper_info = f"{arxiv_title}, {arxiv_abs}"

    # use LLM to evaluate
    client = OpenAI(api_key=os.getenv("TOGETHER_API_KEY"),
                    base_url="https://api.together.xyz/v1")
    messages=[
        {
            "role": "system", 
            "content": """You will be given a prompt, a LLM's response, and a paper's information. Your first task is to decide whether the response is relative to the prompt.
                            If not, write 'no' and don't follow the second task.
                            Second task: You will compare the LLM's response with paper's information, and your task is to decide whether they are talking about the same paper."""
        },
        {
            "role": "user",
            "content": "Prompt: "+prompt+"\n"+"Response: "+title+"\n"+"Information: "+paper_info+"\n"+"Just answer 'yes' or 'no'. If you don't know write 'pass'. Do not include any additional commentary or explanations."
        }
    ] 

    # specific the LLM
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    completions = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.0,
        max_tokens=1,
    )

    response = completions.choices[0].message.content
    response = response.lower()
    
    if response not in ['no', 'yes', 'pass']:
        response = 'pass'


    return response, fabricated
'''


'''
def scrape_arxiv(url):
    try:
        paper_id = url.split("abs/")[-1]
        api_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
        response = requests.get(api_url)
        response.raise_for_status()

        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(response.content)
        title_element = root.find('.//atom:entry/atom:title', namespace)
        abstract_element = root.find('.//atom:entry/atom:summary', namespace)

        title = title_element.text if title_element is not None else None
        abstract = abstract_element.text if abstract_element is not None else None

        return title, abstract, False
    except requests.exceptions.RequestException as e:
        print(f"Error getting paper information: {e}")
        return None, None, True


def scrape_arxiv(url):
    try:

        export_url = url.replace("arxiv.org", "export.arxiv.org")
        #headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(export_url)
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract the paper title
        title_tag = soup.find('h1', class_='title mathjax')
        title = title_tag.text.replace('Title:', '').strip() if title_tag else None

        # Extract the abstract
        abstract_tag = soup.find('blockquote', class_='abstract mathjax')
        abstract = abstract_tag.text.replace('Abstract:', '').strip() if abstract_tag else None

        return title, abstract, False
    except requests.exceptions.RequestException as e:
        print(f"Can't open the link: {url}. Error: {e}")
        return None, None, True

'''


def initialize_files(invalid_file, hallucinations_file, correct_file, evaluations_file):
    
    with open(invalid_file, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "response"])

    with open(hallucinations_file, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "title", "url", "prompt", "found"])

    with open(correct_file, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "title", "url", "prompt"])

    with open(evaluations_file, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["model", "valid_papers", "accuracy"])



def compare_results(title, id, url, prompt, ids_info, hallucinations_file, correct_file):
    arxiv_title = ids_info.get(id, {}).get('title', None)
    arxiv_abs = ids_info.get(id, {}).get('abstract', None)

    # What if we couldn't find associated info, it means that id is fabricated
    if (arxiv_title is None) and (arxiv_abs is None):
        with open(hallucinations_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id, title, url, prompt, "No"])

        return 'no'

    paper_info = f"{arxiv_title}, {arxiv_abs}"

    # use LLM to evaluate
    client = OpenAI(api_key=os.getenv("TOGETHER_API_KEY"),
                    base_url="https://api.together.xyz/v1")
    messages=[
        {
            "role": "system", 
            "content": """You will be given a prompt, a LLM's response, and a paper's information. Your first task is to decide whether the response is relative to the prompt.
                            If not, write 'no' and don't follow the second task.
                            Second task: You will compare the LLM's response with paper's information, and your task is to decide whether they are talking about the same paper."""
        },
        {
            "role": "user",
            "content": "Prompt: "+prompt+"\n"+"Response: "+title+"\n"+"Information: "+paper_info+"\n"+"Just answer 'yes' or 'no'. If you don't know write 'pass'. Do not include any additional commentary or explanations."
        }
    ] 

    # Specify the LLM
    model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

    completions = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.0,
        max_tokens=1,
    )

    response = completions.choices[0].message.content
    response = response.lower()
    
    if response not in ['no', 'yes', 'pass']:
        response = 'pass'

    if response == "no":
        with open(hallucinations_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id, title, url, prompt, "Yes"])

    if response == "yes":
        with open(correct_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id, title, url, prompt])

    return response


def run_in_parallel_collecting_evals(titles, ids, urls, total_prompts, ids_info, hallucinations_file, correct_file, max_workers=5):
    evals = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(compare_results, titles[i], ids[i], urls[i], total_prompts[i], ids_info, hallucinations_file, correct_file): (titles[i], ids[i], urls[i], total_prompts[i])
            for i in range(len(titles))
        }
        
        # Collect the responses as the tasks complete
        for future in tqdm(as_completed(future_to_params), total=len(titles)):
            title, id, url, prompt = future_to_params[future]
            try:
                eval = future.result()
                evals.append(eval)
            except Exception as e:
                print(f"Error processing {title} (ID: {id}): {e}")

    return evals



'''def scrape_arxiv_metadata(url):
    title = None
    abs = None
    fabricated = False
    paper_id = url.split("abs/")[-1]
    arxivjson_filepath ="../arxiv-metadata-oai-snapshot.json"
    with open(arxivjson_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # Parse the line as a JSON object
                data = json.loads(line)
                # Check if the 'id' matches the target ID
                if data.get("id") == paper_id:
                    # Return the title if the ID matches
                    title = data.get("title")
                    abs = data.get("abstract")
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON
                continue
    if title == None and abs == None:
        print("Couldn't find paper related to the id:", paper_id)
        fabricated = True
        return title, abs, fabricated
    
    return title, abs, fabricated
'''



def calculate_ACC(evals):
    filtered_evals = [eval for eval in evals if eval != "pass"]

    corrects = filtered_evals.count("yes")
    ACC = (corrects / len(filtered_evals)) * 100
    ACC = round(ACC, 2) 

    return ACC, len(filtered_evals)


def remove_existing_exp_data(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)

    return 


def find_ids_info(file_path, ids):
    
    print("Reading arxiv json file...")
    results = {} 
    unique_ids = set(ids)

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                paper_id = data.get("id")
                
                if paper_id in unique_ids:
                    # Store the title and abstract in the results dictionary
                    results[paper_id] = {
                        "title": data.get("title"),
                        "abstract": data.get("abstract")
                    }
                    # If all target IDs have been found, stop searching
                    if len(results) == len(unique_ids):
                        break
            except json.JSONDecodeError:
                # Skip lines that are not valid JSON
                continue

    return results

def extract_ids(urls):
    ids = []
    for url in urls:
        id = url.strip().split("abs/")[-1]
        ids.append(id.strip())
    
    return ids


def process_prompt(querier, prompt, model, valid_output_filename, invalid_output_filename):


    valid, titles, urls, addtional_prompt_info, response = querier.query(prompt=prompt, model=model)
    result = {}

    if not valid:
        # Write invalid response to the csv file
        with open(invalid_output_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([prompt, response])

    else:
        # Write invalid reponse to the csv file
        ids = extract_ids(urls)
        result.setdefault('titles', []).extend(titles)
        result.setdefault('urls', []).extend(urls)
        result.setdefault('ids', []).extend(ids)
        result.setdefault('prompts', []).extend(addtional_prompt_info)

    return result


'''
def process_prompt(querier, prompt, model, valid_output_filename, invalid_output_filename, hallucinations_file, correct_file):

    valid, titles, urls, response = querier.query(prompt=prompt, model=model)
    result = {'valid': valid, 'response': response, 'titles': titles, 'urls': urls}

    if not valid:
            with open(invalid_output_filename, "a") as file:
                file.write(f"Prompt: {prompt}\nResponse: {response}\n\n\n\n") 
    else:
        with open(valid_output_filename, "a") as file:
            file.write(f"Prompt: {prompt}\nResponse: {response}\n\n\n\n") 

        for j in range(len(titles)):

            # Check whether it indicates it's not available
            if (not  urls[j].lower().startswith("http")) or ("avaliable" in  urls[j].lower()):
                continue

            eval, fabricated = compare_results(title=titles[j], url=urls[j], prompt=prompt)
            if fabricated:
                    with open(hallucinations_file, "a") as file:
                        file.write(f"Title: {titles[j]}\nID: {urls[j]}\nCan't open? Yes \n\n")
            else:
                if eval == "no":
                    with open(hallucinations_file, "a") as file:
                        file.write(f"Title: {titles[j]}\nID: {urls[j]}\nCan't open? No \n\n")

                if eval == "yes":
                    with open(correct_file, "a") as file:
                        file.write(f"Title: {titles[j]}\nID: {urls[j]}\nCorrect!\n\n")

            
            if eval == None:
                continue
                
            result.setdefault('evaluations', []).append(eval)


    return result
'''