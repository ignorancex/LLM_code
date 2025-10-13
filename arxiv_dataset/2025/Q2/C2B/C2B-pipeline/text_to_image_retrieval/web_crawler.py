import os
from icrawler.builtin import BingImageCrawler


def get_images(prompt, save_dir, max_images):
    os.makedirs(save_dir, exist_ok=True)
    bing_crawler = BingImageCrawler(
        feeder_threads=2,
        parser_threads=4,
        downloader_threads=16,
        storage={
            "root_dir": save_dir
        },
    )
    bing_crawler.crawl(keyword=prompt, max_num=max_images)
