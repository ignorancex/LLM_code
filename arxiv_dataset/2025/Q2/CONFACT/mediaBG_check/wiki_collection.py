import wikipediaapi
import logging
import pickle as pkl
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import load_pkl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_wiki_info(domain_name, wiki_info_file):

    all_wiki_info = load_pkl(wiki_info_file)
    
    if domain_name in all_wiki_info:
        wiki_info = all_wiki_info[domain_name]
        logger.info(f"existing Wikipedia info found for {domain_name}")
    else:
        wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
        page_py = wiki_wiki.page(domain_name)

        if page_py.exists():
            wiki_info = {
                "title": page_py.title,
                "summary": page_py.summary,
                "text": page_py.text
            }
            logger.info(f"Wikipedia page found for {domain_name}")
        else:
            wiki_info = {}
            logger.warning(f"No Wikipedia page found for {domain_name}")
        
        all_wiki_info[domain_name] = wiki_info

        with open(wiki_info_file, 'wb') as f:
            pkl.dump(all_wiki_info, f) #update the file with newly extracted info
            
    return wiki_info

if __name__ == '__main__':
    domain_list = []

    with open('media_bg_collected/media_to_generate.txt', 'r') as f:
        for line in f:
            item = line.strip()
            domain_list.append(item)

    wiki_info_file = "media_bg_collected/wiki_info.pkl"

    from utils import initialize_pkl_file
    initialize_pkl_file(wiki_info_file)

    for domain in tqdm(domain_list):
        extract_wiki_info(domain, wiki_info_file)