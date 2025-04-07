# Filename:     get_domains.py
# Description:  Utility
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020
# Forked from: Abhishek Singh, absingh31
# Modifed by: Edward Crowder & Jay Lansiquot

import tldextract

def get_domain_name(link):
    url_extract = tldextract.extract(link)
    site_name = url_extract.domain + '.' + url_extract.suffix
    return site_name
