# With this python script, scripts will be created to automated the following:
#   1) Downloading the GitHub project locally into the "GitQ_FileStore" folder inside a folder named by project_sequence (to avoid GitHub repos with same repo name)
#   2) Calls the "assessment_report_generator" script which calculate the values for the source code and maintenance metrics and stores the same in the global and local database
#   3) Delete the repository after the analysis

from requests.api import get
from personal_access_tokens import *
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import urllib.parse
import json
import sqlite3
import os


def fetch_latest_commit_id(repo_url):
    # repo_url should be in the form https://github.com/abc/def
    load_request = requests.get(repo_url)
    parser = BeautifulSoup(load_request.content, "html.parser")
    anchor_elements = parser.find_all("a")
    pre_link = repo_url.replace("https://github.com", "") + "/commit/"
    pre_linkf2 = repo_url.replace("https://github.com", "") + "/tree/"
    commit_id = None
    for element in anchor_elements:
        if(element.has_attr("href")):
            if("https://github.com/" not in element["href"]):
                if(pre_link.lower() in element["href"].lower()):
                    commit_id = element["href"].lower().replace(pre_link.lower(), "")
                    break
                elif(pre_linkf2.lower() in element["href"].lower()):
                    commit_id = element["href"].lower().replace(pre_linkf2.lower(), "")
                    break
    return commit_id

username = urllib.parse.quote_plus('kowndi')
password = urllib.parse.quote_plus('kowndi@6772')
connection = MongoClient(
    "mongodb+srv://%s:%s@cluster0.wm2aj.mongodb.net/test" % (username,password))

global_database = connection['GitQ-Metric-Evaluation']
global_collection = global_database["Assessment_Report"]

def query_database(commit_id):
    query = {"commit_id" : commit_id}
    documents = global_collection.find(query)
    for doc in documents:
        return doc         
    return False 

def get_no_rows():
    sqlite_connect = sqlite3.connect('gitq_metric_evaluation.db')
    cursor = sqlite_connect.cursor()
    cursor.execute("SELECT * FROM `assessment_report`")
    rows = cursor.fetchall()
    sqlite_connect.close()
    return len(rows) 

def entry_exists():
    sqlite_connect = sqlite3.connect('gitq_metric_evaluation.db')
    QUERY2 = "SELECT * FROM `assessment_report` WHERE `commit_id` = '" + latest_commit_id + "';"
    cursor = sqlite_connect.cursor()
    cursor.execute(QUERY2)
    rows = cursor.fetchall()
    entryExists = False 
    for row in rows:
        entryExists = True
        break 
    sqlite_connect.close()
    return entryExists

def insert_report(report):
    sqlite_connect = sqlite3.connect('gitq_metric_evaluation.db')
    print(report)
    QUERY3 = '''
                INSERT INTO `assessment_report`
                (`project_sequence`,`commit_id`,`url`,`N_Pack`,`D_Pack` ,`N_Methods`,`O_Methods` ,`N_Class`,`I_Class` ,`Code_Impact`,`Code_Churn` ,`Open_Bugs`, `Closed_Bugs`,`Active_Contributors`, `Percentage_Active_Contributors`)
                VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    sqlite_connect.execute(QUERY3,report)
    sqlite_connect.commit()
    sqlite_connect.close()

github_api_urls_fp = open('urls', 'r')
api_urls = []
project_urls = []
for url in github_api_urls_fp:
    project_urls.append(url.replace("\n", ""))
    api_urls.append('https://api.github.com/repos/' +
                    url.replace("\n", "").replace("https://github.com/", ""))
github_api_urls_fp.close()
evaluation_fp = open('evaluation.sh', 'a')
t1 = False
t2 = False
# for files in os.listdir():
#     if(files.endswith(".sh") and "project" in ):
#         t1 = True
#     elif("sequence_looker.json" == files):
#         t2 = True 
# if(t1):
#     exec = os.popen("rm *sh")
# if(t2):
#     exec = os.popen("rm sequence_looker.json")
current_sequence = str(get_no_rows() + 1)
for x in range(len(project_urls)):
    latest_commit_id = fetch_latest_commit_id(project_urls[x])
    if(latest_commit_id is not None):
        print(latest_commit_id)
        doc = query_database(str(latest_commit_id))
        if(doc is not False):
            if(entry_exists()):
                print("Project - " + project_urls[x].replace("https://github.com/","") + " has been evaluated by GitQ already and values are up to date! You can directly use the plugin for this repo")
            else:
                report = (current_sequence,latest_commit_id,str(doc["url"]),str(doc["N_Pack"]),str(doc["D_Pack"]),str(doc["N_Methods"]),str(doc["O_Methods"]),str(doc["N_Class"]),str(doc["I_Class"]),str(doc["Code_Impact"]),str(doc["Code_Churn"]),str(doc["Open_Bugs"]),str(doc["Closed_Bugs"]), str(doc["Percentage_Active_Contributors"]), str(doc["Active_Contributors"]))
                insert_report(report)
                print("Assessment Report for project - " + project_urls[x].replace("https://github.com/","") + " \nis already available in Global database, hence retrieved into local database! Now, you can use the plugin for this repo")
                current_sequence = str(int(current_sequence) + 1)
        else:
            fp2 = open("project" + current_sequence +".sh","a")
            url_a = project_urls[x]
            name = api_urls[x].split("/")[5]
            url = api_urls[x].replace("https://api.github.com/repos","")
            url0 = "https://github.com" + url
            for token in personal_access_tokens:
                url_t = "https://" + token + ":x-oauth-basic@github.com" + url
                fp2.write("git clone "+ url_t +" GitQFileStore/"+current_sequence+"/"+name +" || ")
            fp2.write("git clone "+ url0 +" GitQFileStore/"+current_sequence+"/"+name  + " \n")
            fp2.write("chmod +x GitQFileStore/"+current_sequence+"\n")
            fp2.write("chmod +x GitQFileStore/"+current_sequence+"/"+name+"\n")
            fp2.write("chmod +x GitQFileStore/"+current_sequence+"/"+name+"/\n")
            fp2.write('wait\n')
            fp2.write('python3 assessment_report_generator.py '+ current_sequence + ' ' + url_a  + ' ' + latest_commit_id + ' ' + name +"\n")
            # Repository inside the "GitQ_FileStore" will be deleted after storing the results in the database
            # Comment the below 2 lines if you do not want to delete the downloaded repository
            fp2.write('wait\n')
            fp2.write("rm -rf GitQFileStore/"+current_sequence+"\n")
            fp2.close()
            evaluation_fp.write("./project"+current_sequence+".sh\n")
            current_sequence = str(int(current_sequence) + 1)
    else:
        print("For Project - " + url + " - Commit Id cannot be fetched at this moment. YOU MAY CHECK YOUR INTERNET CONNECTION! ")
    
evaluation_fp.close()

sqlite_connect = sqlite3.connect('gitq_metric_evaluation.db')
cursor = sqlite_connect.cursor()
cursor.execute("SELECT * FROM `assessment_report`")
rows = cursor.fetchall()
sequence_json = {}
for row in rows:
    sequence_json[row[0]] = row[2]
with open('sequence_looker.json', 'w') as f:
  json.dump(sequence_json, f, ensure_ascii=False)

sqlite_connect.close()

