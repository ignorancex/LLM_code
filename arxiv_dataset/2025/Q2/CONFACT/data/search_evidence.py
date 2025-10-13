from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from openai import OpenAI 
import pandas as pd
import json
import os
from tqdm import tqdm
import time
from nltk import pos_tag, word_tokenize
import random
import argparse


client = OpenAI(api_key =os.getenv("OPENAI_API_KEY")) 

def claim2qn(claim, label):
    system_prompt = """
Your task is to convert a factual claim into a binary question. Preserve the core elements of the statement, and rephrase it in an interrogative format. 
Make sure the question is clear, concise, and maintains the original meaning of the claim.
Consider the label of the claim as well. If the label is 'Supported', craft the question to elicit a 'Yes' answer. If the label is 'Refuted', craft the question to elicit a 'No' answer.
"""
    user_prompt = """ Now, convert the following claim into a question: """

    user_prompt += f"\nClaim: {claim}\nLabel: {label}\nQuestion:"

    try:
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        question = response.choices[0].message.content.strip()
        return question
    except Exception as e:  
        print(f"An error occurred in question generation: {e}")
        return None
    
def to_search_query(claim):
    token_words = word_tokenize(claim.strip())
    tags = pos_tag(token_words)

    target_tags = ["CD", "JJ", "NN", "VB"]
    search_string = []

    for token, tag in zip(token_words, tags):
        for keep_tag in target_tags:
            if tag[1].startswith(keep_tag):
                search_string.append(token)

    search_string = " ".join(search_string)

    return search_string

def search(query, country_code, n_pages = 1):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    query_encoded = query.replace(" ", "+") 
    links = [] 

   
    for page in range(n_pages):
        url = f"http://www.google.com/search?q={query_encoded}&start={page * 10}"
        if country_code and isinstance(country_code, str):
            url += f"&gl={country_code.lower()}"

        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        search_results = soup.find_all('div', class_="yuRUbf")
        for href in search_results:
            url = href.a.get('href')
            links.append({'original_link': url})

        driver.quit()

    return links

def main():
    parser = argparse.ArgumentParser(description='Process a file of claims and search for related information.')
    parser.add_argument('--input', type=str, help='The input file containing claims')
    parser.add_argument('--type', type=str, help='Either Averitec or FactCheckQA')
    parser.add_argument('--output', type=str, help='The outpt file containing URLs corresponding to web pages retrieved for each claim')
    args = parser.parse_args()

    json_filepath = args.output

    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as f:
            results = json.load(f)
    else:
        results = []

    processed_claims = set(result['claim'] for result in results) # to resume searching if handling large datasets


    if args.type == 'Averitec':
        data = pd.read_csv(args.input)
    elif args.type == 'FactCheckQA':
        json_string = json.load(args.input)
        data = pd.read_json(json_string)
    else:
        raise argparse.ArgumentTypeError(f"Invalid input: {args.input}")
        

    for index, row in tqdm(data.iterrows(), total=len(data), desc="Searching claims"):

        claim_date = None
        review_date = None

        if args.type == 'Averitec':
            claim = row['claim']
            label = row['label']
            claim_date = row['claim_date']
            original_claim_url = row['original_claim_url']
            fact_checking_article = row['fact_checking_article']
            country = row['location_ISO_code']

        elif args.type == 'FactCheckQA':
            claim = row['claim_text']
            review_date = row['review_date']
            label = "Supported " if row['remapped_verdict_class']==True else "Refuted"

            original_claim_url = None
            fact_checking_article = row['url']
            country = row['publisher_country_code']

        if processed_claims and claim in processed_claims:
            continue

        query = to_search_query(claim2qn(claim, label))
        evidence_links = search(query, country)

        time_to_sleep = random.uniform(0, 5)
        time.sleep(time_to_sleep)

        if not evidence_links:
            time.sleep(120) # sleep for 2 minutes if no evidence links found (scraping rate limit reached)

        result = {
            'claim': claim,
            'label': label,
            'claim_date': claim_date,
            'review_date': review_date,
            'original_claim_url': original_claim_url,
            'fact_checking_article': fact_checking_article,
            'country': country,
            'query': query,
            'evidence_url': evidence_links
        }

        results.append(result)

        if index % 5 == 0 or index == len(data) - 1:
            with open(json_filepath, 'w') as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
