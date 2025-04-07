# Filename:     elitemarket.py
# Description:  Parser rules for icarus market index pages
# Version:      0.1 - Code Review Required before release
# Date:         June 29 2020   
# Author:       Edward Crowder
from os import walk
from DNDO import post
import json
import pprint
import urllib.request
import lxml
import bs4 as bs

path = "../Data/input/elitemarket/"

files = []
for (dirpath, dirnames, filenames) in walk(path):
    files.extend(filenames)
    break

print(files)
posts = []
for f in files:
    #Create new post object
    p = post()
    with open(path + f, "rb") as html:
        #File Setup
        contents = html.read()
        soup = bs.BeautifulSoup(contents, 'lxml')
        
        #Data extraction
        p.title = soup.find("font",{"class":"blue"}).text
        upperdetails = (soup.findAll("font")[1].text).splitlines()
        TableRows = soup.findAll('tr')
        rows = TableRows[2].text.splitlines()
        #urlParts = f.split("Scraped-http")[1].split("onionofferphpid")
        #url = "http://" + urlParts[0] + ".onion/offer=phpid?" + urlParts[1]
        #Object Creation
        p.url = f
        p.seller = upperdetails[0].split("|")[1].split(": ")[1]
        p.category = upperdetails[0].split("|")[2].split(": ")[1].split("Â» ")[0]
        #p.subCategory = upperdetails[0].split("|")[2].split(": ")[1].split("Â» ")[1]
        p.creationDate = upperdetails[0].split("|")[3].split(":  ")[1]
        p.views = rows[2].split(": ")[1]
        p.purchases = rows[3].split(": ")[1]
        p.expire = rows[4].split(": ")[1]
        p.productClass = rows[5].split(": ")[1]
        p.originCountry = rows[8].split(": ")[1]
        p.shippingDestinations = rows[9].split(": ")[1]
        p.quantity = rows[10].split(": ")[1]
        p.payment = rows[11].split(": ")[1]
        # have to content since its outside DOM
        x = str(contents)
        x = x[-26:]
        p.analyst_dateCollected = x[:-7]

        #Debug
        print("URL:\t\t" + p.url)
        print("TITLE: \t\t" + p.title)
        print("SELLER: \t" + p.seller)
        print("CAT: \t\t" + p.category)
        print("OFFER:\t\t" + p.creationDate)
        print("VIEWS:\t\t" + p.views)
        print("PURCHASES:\t" + p.purchases)
        print("EXPIRE:\t\t" + p.expire)
        print("proClas\t\t" + p.productClass)
        print("ocountry\t" + p.originCountry)
        print("shipping\t" + p.shippingDestinations)
        print("qty\t\t" + p.quantity)
        print("payment\t\t" + p.payment)
        posts.append(p)

print("files availble:")
print(len(files))
print("posts scrapped:")
print(len(posts))

for p in posts:
    filename = '../Data/output/elitemarket_json/' + p.url + ".json"
    with open(filename, "w") as outfile:
        json.dump(p.__dict__, outfile)



#print(p1.title);
#print(p1.creationDate)
#print(p1.seller)

#data = json.dumps(p1.__dict__)
#print(data)