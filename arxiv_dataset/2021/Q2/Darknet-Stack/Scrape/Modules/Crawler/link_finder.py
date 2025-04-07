# Filename:     link_finder.py
# Description:  Logic for Crawler
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020
# Forked from: Abhishek Singh, absingh31
# Modifed by: Edward Crowder & Jay Lansiquot

import urllib
from html.parser import HTMLParser
from urllib.parse import urljoin

class link_crawler(HTMLParser):

    def __init__(self, start_link, web_url):
        super().__init__()
        self.start_link = start_link
        self.web_url = web_url
        self.urls = set()

    #Main logic for crawler
    def handle_starttag(self, tag, found_attributes):     
        if tag == 'a':
            for (attr, value) in found_attributes:
                if attr == 'href':
                    url = urllib.parse.urljoin(self.start_link, value)
                    self.urls.add(url)

    def page_urls(self):
        return self.urls

    def error(self, message):
        pass
