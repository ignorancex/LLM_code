# Filename:     icarus.py
# Description:  Parser rules for icarus market index pages
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020     
# Author:       Edward Crowder

# New Parser Instructions
# Change path variable to represent new marketplace folder (Line 21)
# Modify data extraction rules (Lines 48 - ~70)
# Change output variable (Line 96)

from os import walk
from DNDO import post
from datetime import datetime
import json
import pprint
import urllib.request
import lxml
import bs4 as bs

path = "../Data/input/asean/"

files = []
for (dirpath, dirnames, filenames) in walk(path):
    files.extend(filenames)
    break

#Get all files in test dir
posts = []
for f in files:

    #Create new Post object
    p = post()
    with open(path + f, "rb") as html:
        contents = html.read()
        soup = bs.BeautifulSoup(contents, 'lxml')
        data = []

        #Icarus holds all of its post details inside <b> only
        for td in soup.find_all('td'):
            data.append(td.text)

        #Lets get rid of broken post ids that redirect to /home/
        if(data == []):
            print("\tHomepage Detected! Aborting!")
            continue
        
        #Data extraction
        p.title = soup.title.string[:-14]
        p.url = f
        #The seller URL is the last <a> on the page
        p.seller = data[4].strip()
        #Fix category newlines, sub category logging and no cats
        cat =  data[5].splitlines()
        if(len(cat) == 3):
            p.category = (cat[1].strip()[:-1] + cat[2].strip())
        elif(len(cat) == 2):
             p.category = (cat[1].strip()[:-1])
        else:
            p.category = "None"
        #asean posts do not record these records
        p.creationDate = "None"
        p.views = "None"
        p.purchases = "None"
        p.expire = "None"
        p.productClass = data[3].strip()
        p.originCountry = data[7].strip()
        p.shippingDestinations = data[8].strip()
        p.quantity = data[2].strip()
        #Only offered payment type
        p.payment = "Escrow"
        p.price = data[1].strip()
        #Grab the last <span text="META_DATA"> of the page that Jay insert
        # have to content since its outside DOM
        x = str(contents)
        x = x[-26:]
        p.analyst_dateCollected = x[:-7]
        #Debug
        print("\tURL:\t\t" + p.url)
        print("\tTITLE: \t\t" + p.title)
        print("\tSELLER: \t" + p.seller)
        print("\tCAT: \t\t" + p.category)
        print("\tOFFER:\t\t" + p.creationDate)
        print("\tVIEWS:\t\t" + p.views)
        print("\tPURCHASES:\t" + p.purchases)
        print("\tEXPIRE:\t\t" + p.expire)
        print("\tproClas\t\t" + p.productClass)
        print("\tocountry\t" + p.originCountry)
        print("\tshipping\t" + p.shippingDestinations)
        print("\tquantity\t" + p.quantity)
        print("\tpayment\t\t" + p.payment)
        print("\tprice\t\t" + p.price)
        print("\tDatetime\t" + p.analyst_dateCollected)
        posts.append(p)

#Output - Log to file or something?
print("files availble:")
print(len(files))
print("posts scrapped:")
print(len(posts))

#write files to json files
for p in posts:
    filename = '../Data/output/asean_json/' + p.url + ".json"
    with open(filename, "w") as outfile:
        json.dump(p.__dict__, outfile)