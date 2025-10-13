import requests
from bs4 import BeautifulSoup
import PyPDF2
import re
from io import BytesIO
from tqdm import tqdm
import json
import os
import concurrent.futures
import threading
import argparse

file_lock = threading.Lock()

def extract_text_from_pdf(pdf_content, url):
    pdf_text = ""
    try:
        with BytesIO(pdf_content) as open_pdf_file:
            reader = PyPDF2.PdfReader(open_pdf_file)
            for page in reader.pages:
                pdf_text += page.extract_text() + " "
    except Exception as e:
        print(f"Failed to extract text from PDF {url}: {e}")
    return pdf_text

def parse_html(url):
    try:
        response = requests.get(url, timeout=10)  
        response.raise_for_status()

        if 'application/pdf' in response.headers.get('Content-Type'):
            return extract_text_from_pdf(response.content, url)

        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        text = soup.get_text(separator=' ', strip=True)
        text = re.sub(r'\s+', ' ', text)

        return text

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return ''

def parse_html_with_timeout(url, timeout=300):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(parse_html, url)
        try:
            
            content = future.result(timeout=timeout)
            return content
        except concurrent.futures.TimeoutError:
            print(f"Timeout occurred while parsing: {url}")
            return ""  

def process_claim_evidence(item, processed_claim):
    claim = item['claim']
    if claim in processed_claim:
        return None  # Skip already processed claims

    retrieved_results = item['evidence_url']

    for evidence in tqdm(retrieved_results, desc='checking evidence'):
        url = evidence['original_link']
        content = parse_html_with_timeout(url, timeout=300)
        evidence['content'] = content

    return item



def write_output(output, output_file):
    with file_lock:
        with open(output_file, 'w') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
            
def main():

    parser = argparse.ArgumentParser(description='Process a file of claims and search for related information.')
    parser.add_argument('--input', type=str, help='The input file containing claims and searched urls')
    parser.add_argument('--output', type=str, help='The outpt file containing scraped text content of each web page')
    args = parser.parse_args()


    claim_file = args.input
    claim_evidence = json.load(open(claim_file, 'r', encoding='utf-8'))

    output_file = args.output
    if not os.path.exists(output_file):
        with open(output_file, 'w') as f:
            json.dump([], f, indent=4)

    output = json.load(open(output_file, 'r'))
    processed_claim = {item['claim'] for item in output}


    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_claim_evidence, item) for item in claim_evidence]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing claims"):
            result = future.result()
            if result is not None:
                output.append(result)
                write_output(output, processed_claim)

if __name__ == '__main__':
    main()
