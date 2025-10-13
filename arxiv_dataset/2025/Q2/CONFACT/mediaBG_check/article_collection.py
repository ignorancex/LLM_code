import sys
sys.path.append("..")
import logging
from feedsearch import search
import feedparser
from goose3 import Goose
from utils import load_pkl
import pickle as pkl
from tdqm import tqdm
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def extract_articles(article_urls):
    logger.info("Extracting articles from URLs")
    g = Goose()
    article_info = []
    invalid_article_urls = []

    for article in article_urls:
        if "link" not in article:
            continue
        link = article["link"]

        try:
            ext_info = g.extract(url=link)
            article_info.append({
                "title": ext_info.title,
                "cleaned_text": ext_info.cleaned_text,
            })
            logger.info(f"Article extracted from {link}")
        except Exception as e:
            logger.error(f"Invalid article URL: {link} - {e}")
            invalid_article_urls.append(link)
    return article_info, invalid_article_urls

def collect_article_urls(domain_name):
    logger.info(f"Collecting article URLs for {domain_name}")
    invalid_domains = []
    try:
        feeds = search(domain_name)
        urls = [f.url for f in feeds]
    except Exception as e:
        logger.error(f"Error searching feeds for {domain_name} - {e}")
        invalid_domains.append(domain_name)
        urls = []

    valid_article_urls = []
    if urls:
        try:
            news_feed = feedparser.parse(urls[0])
            entries = news_feed["entries"]
        except Exception as e:
            logger.error(f"Error parsing feed for {domain_name} - {e}")
            entries = []

        for entry in entries:
            title = entry.get("title", "")
            link = entry.get("link", "")
            pub = entry.get("published", "")
            valid_article_urls.append({
                "title": title,
                "published": pub,
                "link": link
            })
            logger.info(f"URL collected: {link}")

    return valid_article_urls, invalid_domains


def collect_article_info(domain, article_info_file):

    all_article_info = load_pkl(article_info_file)

    if domain in article_info:
        article_info = all_article_info[domain]
    else:
        valid_article_urls, invalid_domains = collect_article_urls(domain)
        article_info, invalid_article_urls = extract_articles(valid_article_urls)
        all_article_info[domain] = article_info

    pkl.dump(all_article_info, open(article_info_file, "wb")) #update the file with newly extracted info

    return article_info

if __name__=="__main__":
    domain_list = []

    with open('media_bg_collected/media_to_generate.txt', 'r') as f:
        for line in f:
            item = line.strip()
            domain_list.append(item)

    article_info_file = "media_bg_collected/article_info.pkl"

    from utils import initialize_pkl_file
    initialize_pkl_file(article_info_file)

    for domain in tqdm(domain_list):
        collect_article_info(domain, article_info_file)

