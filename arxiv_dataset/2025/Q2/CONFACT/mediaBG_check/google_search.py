import sys
sys.path.append("..")
import logging
from utils import load_json, load_pkl
import requests
import trafilatura
from googleapiclient.discovery import build
import time
import os 
import pickle as pkl
from tqdm import tqdm

CX = os.getenv('GOOGLE_CX')
google_api = os.getenv("GOOGLE_API_KEY")

service = build("customsearch", "v1", developerKey=google_api)

logger = logging.getLogger(__name__)

class GoogleSearch:
    def __init__(self, query_file, all_evidence_url_file, all_scraped_text_file):
        logger.info("Initializing GoogleSearch")
        self.queries = load_json(query_file)
        self.all_evidence_url_path = all_evidence_url_file
        self.all_evidence_url = load_pkl(all_evidence_url_file)
        # {"media_name_1": {
        #     "qn1": [link1, link2, link3], 
        #     "qn2": [link1, link2, link3]
        #     },
        # }

        self.all_scraped_text = load_pkl(all_scraped_text_file)
        self.all_scraped_path = all_scraped_text_file
        # {"media_name_1": [
        #     [text content of qn1 evidence1, text content of qn1 evidence2, text content of qn1 evidence3...],
        #     [text content of qn2 evidence1, text content of qn2 evidence2, text content of qn2 evidence3...]
        # ]}


    def scraper(self, web_pages):
        logger.info("Scraping text content from web pages")

        all_texts = []
        for url in web_pages:
            try:
                response = requests.get(url, timeout=5)  
                response.raise_for_status()  
                
                # If the request was successful, extract the content with trafilature
                downloaded = response.text
                if downloaded:
                    texts = trafilatura.extract(downloaded)
                    if texts:
                        all_texts.append(texts)
                        break #scrape 1st existing content
            
            except (requests.exceptions.RequestException, Exception) as e:
                # print(f"Error fetching {url}: {e}")
                pass
        
        return all_texts

    def search_pages(self, search_term, **kwargs):
        try:
            res = service.cse().list(q=search_term, cx=CX, **kwargs).execute()
            logger.info(f"Search completed for term: {search_term}")
            return res.get('items', [])
        except Exception as e:
            logger.error(f"Error during search for term {search_term} - {e}")
            return []

    def process_search_results(self, results):
        logger.info("Processing search results")
        return [str(result["link"]) for result in results]

    def get_google_search_results(self, search_string, sort_date=None, page=0):
        logger.info(f"Getting Google search results for: {search_string}")
        search_results = []
        sort = None if sort_date is None else ("date:r:19000101:" + sort_date)
        for _ in range(3):
            try:
                search_results += self.search_pages(
                    search_string,
                    num=10,
                    start=0,
                    sort=sort,
                    dateRestrict=None,
                    gl="US"
                )
                break
            except Exception as e:
                logger.warning(f"Retrying search for {search_string} due to error: {e}")
                time.sleep(1)
        return self.process_search_results(search_results)


    def google_search(self, source_name):
        logger.info(f"Performing Google search for source: {source_name}")
        saved_urls = self.all_evidence_url.get(source_name, {})

        if source_name not in self.all_scraped_text:
            self.all_scraped_text[source_name] = []

            entity = {}
            for idx, item in enumerate(tqdm(self.queries, desc="Processing queries")):  # loop over the saved entities again as google connection is not stable, some query results might be lost
                localized_question = item['question'].replace("X", f"\"{source_name}\"").strip()
                localized_search_query = item['statement'].replace("X", f"\"{source_name}\"").strip()

                entity[localized_question] =  saved_urls.get(localized_question, [])
                if entity[localized_question]:
                    logger.info(f"Using cached results for: {localized_question}")
                else:
                    search_results = self.get_google_search_results(localized_search_query)
                    entity[localized_question] = search_results

                    # Ensure that the list is long enough to handle the current index
                    while len(self.all_scraped_text[source_name]) <= idx:
                        self.all_scraped_text[source_name].append([])  # Append empty list for missing indices

                    self.all_scraped_text[source_name][idx] = self.scraper(search_results)
                    logger.info(f"Scraped text content for: {localized_question}")

            self.all_evidence_url[source_name] = entity
            
            pkl.dump(self.all_evidence_url, open(self.all_evidence_url_path, "wb"))
            logger.info(f"Saved all evidence urls for source: {source_name}")

            pkl.dump(self.all_scraped_text, open(self.all_scraped_path, "wb"))
            logger.info(f"Saved all scraped text for source: {source_name}")

        else:
            logger.info(f"Using scraped text for source: {source_name}")

        return self.all_scraped_text[source_name]


if __name__=='__main__':
    domain_list = []

    with open('media_bg_collected/media_to_generate.txt', 'r') as f:
        for line in f:
            item = line.strip()
            domain_list.append(item)

    query_file = "media_bg_collected/google_queries.json"
    all_evidence_url_file = "media_bg_collected/google_evidence_urls.pkl"
    all_scraped_text_file = "media_bg_collected/google_evidence_text.pkl"

    from utils import initialize_pkl_file
    files = [all_evidence_url_file, all_scraped_text_file]

    for file in files:
        initialize_pkl_file(file)

    for domain in domain_list:
        google_search = GoogleSearch(query_file, all_evidence_url_file, all_scraped_text_file)
        google_evidence = google_search.google_search(domain)