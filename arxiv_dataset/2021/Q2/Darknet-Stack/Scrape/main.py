# Filename:     TorScrapper.py
# Description:  Main driver of both Crawler and Scraper
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020
# Forked from: Abhishek Singh, absingh31
# Modifed by: Edward Crowder & Jay Lansiquot

#Importing Essentials
from multiprocessing import Pool
from pyfiglet import Figlet
import os

#Opening onions directory. To scrape links add the same in onions.txt
with open("onions.txt", "r") as onion:
    content = onion.read().splitlines()

#Terminal Process for Crawler
def ExecuteCrawler(url):
    execute = str(' python3 /home/ubuntu/Darknet-Stack/Scrape/Modules/Crawler/crawl.py ' + url)
    os.system(execute)

#Terminal Process for Scraper 
def ExecuteScraper(url):
    execute = str(' python3 /Users/laptop/Documents/capstone/Scrape/Modules/Scraper/Scrape.py ' + url)
    print (execute)
    os.system(execute)

#MultiProcessing Implementation (Limit - 5 processes at a time)
def Multiprocessing(task):
    if (os.path.exists("Output")):
        delete = str('rm -r Output')
        os.system(delete)
        os.makedirs("Output")
    else:
        os.makedirs("Output")

    with Pool(processes=5) as pool:
        for onion in range(0, len(content)):
            pool.apply(task, args=(content[onion],))


if __name__ == '__main__':
        Multiprocessing(ExecuteCrawler)
        #Multiprocessing(ExecuteScraper)



