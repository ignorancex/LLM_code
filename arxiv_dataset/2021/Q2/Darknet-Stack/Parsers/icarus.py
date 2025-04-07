# Filename:     icarus.py
# Description:  Parser rules for icarus market index pages
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020     
# Author:       Edward Crowder

from os import walk
from DNDO import post
from datetime import datetime
import json
import pprint
import urllib.request
import lxml
import bs4 as bs

path = "../Data/input/icarus/"

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
        for bold in soup.find_all('i'):
            data.append(bold.text)

        #Lets get rid of broken post ids that redirect to /home/
        if(data == []):
            print("\tHomepage Detected! Aborting!")
            continue

        #Data extraction
        p.title = data[0]
        p.url = f
        #The seller URL is the last <a> on the page
        for f in soup.find_all('a'):
            p.seller = f.text
        #Icarus does not include category or post creation infomation
        p.category = "None"
        p.creationDate = "None"
        p.views = data[2]
        p.purchases = data[1]
        #icarus posts do not expire
        p.expire = "None"
        p.productClass = data[3]
        p.originCountry = data[6]
        p.shippingDestinations = data[7]
        p.quantity = data[5]
        p.payment = data[4]
        p.price = data[8]
        #Grab the last <span text="META_DATA"> of the page that Jay inserts
        p.analyst_dateCollected = soup.find('span', {'class' : 'META_DATA'}).text
        
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
        print("\tDatetime\t\t" + p.analyst_dateCollected)
        posts.append(p)

#Output - Log to file or something?
print("files availble:")
print(len(files))
print("posts scrapped:")
print(len(posts))

#write files to json files
for p in posts:
    filename = '../Data/output/icarus_json/' + p.url + ".json"
    with open(filename, "w") as outfile:
        json.dump(p.__dict__, outfile)