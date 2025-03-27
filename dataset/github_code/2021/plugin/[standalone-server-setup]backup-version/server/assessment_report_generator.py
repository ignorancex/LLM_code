import os
import sys
import requests
from bs4 import BeautifulSoup
import os as inner_os
from time import strptime
from datetime import datetime
import arrow
import statistics
import pymongo
from personal_access_tokens import *
import time
import urllib.parse
import sqlite3

root_directory = os.getcwd()


def evaluate(file_path, files, D_Pack, I_Class, I_methods):
    overrides = 0
    p = 0
    s = 0
    pack = []
    syn_class = []
    inh_class = []
    methods = []
    inh = 0
    m = 0
    j = 0
    pack = []
    fp = open(file_path, 'r')
    for line in fp:
        syntactic_elements = line.replace("\n", "").split(' ')
        for j in range(0, len(syntactic_elements)):
            if(j < len(syntactic_elements) and syntactic_elements[j] == ' '):
                continue
            else:
                if(j < len(syntactic_elements) and syntactic_elements[j] == "@Override"):
                    overrides += 1
                elif(j < len(syntactic_elements) and (syntactic_elements[j] == "import" or syntactic_elements[j] == "package")):
                    j += 1
                    while(j < len(syntactic_elements) and syntactic_elements[j] == ' '):
                        j += 1
                    inc = 0
                    for x in files:
                        if(j < len(syntactic_elements) and x in syntactic_elements[j]):
                            D_Pack += 1
                    tmp = 0
                    tmp_flag = False
                    for tmp in range(0, p):
                        if(j < len(syntactic_elements) and pack[tmp] == syntactic_elements[j]):
                            tmp_flag = True
                            break
                    if(j < len(syntactic_elements) and tmp_flag == False):
                        pack.append(syntactic_elements[j])
                        p += 1
                    j -= 1
                elif(j < len(syntactic_elements) and syntactic_elements[j] == "class"):
                    j += 1
                    while(j < len(syntactic_elements) and syntactic_elements[j] == ' '):
                        j += 1
                    rt = 0
                    fg = False
                    if(s > 0):
                        for rt in range(0, s):
                            if(j < len(syntactic_elements) and syn_class[rt] == syntactic_elements[j]):
                                fg = True
                                break
                    if(j < len(syntactic_elements) and fg == False):
                        syn_class.append(syntactic_elements[j])
                        s += 1
                    j -= 1
                elif(j < len(syntactic_elements) and syntactic_elements[j] == 'extends'):
                    rt = 0
                    fg = False
                    if(inh > 0):
                        for rt in range(0, inh):
                            if(j+1 < len(syntactic_elements) and inh_class[rt] == syntactic_elements[j+1]):
                                fg = True
                                break
                    if(fg == False):
                        if(j+1 < len(syntactic_elements)):
                            inh_class.append(syntactic_elements[j+1])
                            inh += 1
                    j += 1
                elif(j < len(syntactic_elements) and (syntactic_elements[j] == "void"
                                                      or syntactic_elements[j] == "int"
                                                      or syntactic_elements[j] == "int*"
                                                      or syntactic_elements[j] == "char"
                                                      or syntactic_elements[j] == "char*"
                                                      or syntactic_elements[j] == "float"
                                                      or syntactic_elements[j] == "float*"
                                                      or syntactic_elements[j] == "double"
                                                      or syntactic_elements[j] == "double*"
                                                      or syntactic_elements[j] == "short"
                                                      or syntactic_elements[j] == "short*"
                                                      or syntactic_elements[j] == "byte"
                                                      or syntactic_elements[j] == "byte*"
                                                      or syntactic_elements[j] == "boolean"
                                                      or syntactic_elements[j] == "string")):
                    j += 1
                    while(j < len(syntactic_elements) and syntactic_elements[j] == ' '):
                        j += 1
                    if(j < len(syntactic_elements)):
                        temp = syntactic_elements[j].split('(')
                        tm = len(temp)
                        if(tm > 1):
                            rt = 0
                            fg = False
                            if(m > 0):
                                for rt in range(0, m):
                                    if(j < len(syntactic_elements) and methods[rt] == syntactic_elements[j]):
                                        fg = True
                                        break
                            if(fg == False):
                                methods.append(temp[0])
                                m += 1
                    j -= 1
                elif(j < len(syntactic_elements) and ("System" in syntactic_elements[j] or
                                                      "if" in syntactic_elements[j] or
                                                      "while" in syntactic_elements[j] or
                                                      "do" in syntactic_elements or
                                                      "for" in syntactic_elements)):
                    break
                else:
                    continue
    non_over = m - overrides
    return[p, inh, overrides, I_Class + s, I_methods + non_over, D_Pack]


def get_open_issue_count(URL):
    openIssues = 0
    try:
        open_url = URL + "/issues?q=is%3Aopen+is%3Aissue"
        flag = False
        for token in personal_access_tokens:
            if(flag == True):
                break
            else:
                try:
                    r = requests.get(open_url, auth=(
                        personal_access_tokens[token], token))
                    dom = BeautifulSoup(r.content, 'html5lib')
                    dom_tree = dom.body.find_all(
                        'a', class_='btn-link selected')[0].text.replace("\n", "").split("Open")[0]
                    if("," in dom_tree):
                        dom_tree = dom_tree.replace(",", "")
                    openIssues = int(dom_tree)
                    flag = True
                    break
                except Exception as ex:
                    print(ex)
                    continue
        if(flag == False):
            try:
                r = requests.get(open_url)
                dom = BeautifulSoup(r.content, 'html5lib')
                dom_tree = dom.body.find_all(
                    'a', class_='btn-link selected')[0].text.replace("\n", "").split("Open")[0]
                if("," in dom_tree):
                    dom_tree = dom_tree.replace(",", "")
                openIssues = int(dom_tree)
                print('without token - open issues - fetch ok')
            except Exception as ex:
                print(ex)
                openIssues = 0
    except Exception as ex:
        print(ex)
    print(openIssues)
    return openIssues


def get_closed_issue_count(URL):
    closedIssues = 0
    try:
        closed_url = URL + "/issues?q=is%3Aissue+is%3Aclosed"
        flag = False
        for token in personal_access_tokens:
            if(flag == True):
                break
            else:
                try:
                    r = requests.get(closed_url, auth=(
                        personal_access_tokens[token], token))
                    dom = BeautifulSoup(r.content, 'html5lib')
                    dom_tree = dom.body.find_all(
                        'a', class_='btn-link selected')[0].text.replace("\n", "").split("Closed")[0]
                    if("," in dom_tree):
                        dom_tree = dom_tree.replace(",", "")
                    closedIssues = int(dom_tree)
                    flag = True
                    break
                except Exception as ex:
                    print(ex)
                    continue
        if(flag == False):
            try:
                r = requests.get(closed_url)
                dom = BeautifulSoup(r.content, 'html5lib')
                dom_tree = dom.body.find_all(
                    'a', class_='btn-link selected')[0].text.replace("\n", "").split("Closed")[0]
                if("," in dom_tree):
                    dom_tree = dom_tree.replace(",", "")
                closedIssues = int(dom_tree)
                print('without token - closed issues - fetch ok')
            except Exception as ex:
                print(ex)
                closedIssues = 0
    except Exception as ex:
        print(ex)
    print(closedIssues)
    return closedIssues


def time_gap(time1, time2):
    Y1 = int(time1[4])
    M1 = int(strptime(time1[1], '%b').tm_mon)
    D1 = int(time1[2])
    end = datetime(Y1, M1, D1)
    Y2 = int(time2[4])
    M2 = int(strptime(time2[1], '%b').tm_mon)
    D2 = int(time2[2])
    start = datetime(Y2, M2, D2)
    numberOfDays = -1
    for d in arrow.Arrow.range('day', start, end):
        numberOfDays += 1
    return numberOfDays


def get_score(stream):
    commit_score = 0
    today = datetime.today().date()
    if isinstance(today, str):
        today = datetime.strptime(today, '%Y-%m-%d')
    commit_gaps = []
    if(len(stream) > 3):
        for commit in range(0, len(stream)-2):
            gap = time_gap(stream[commit].split(
                " "), stream[commit+1].split(" "))
            commit_gaps.append(gap)
        averagecommitTime = statistics.mean(commit_gaps)  # In number of days
        standardDeviation = statistics.stdev(commit_gaps)
        expected_commit_gap = averagecommitTime + standardDeviation
        time1 = stream[0].split(" ")  # Latest commit date
        Y1 = int(time1[4])
        M1 = int(strptime(time1[1], '%b').tm_mon)
        D1 = int(time1[2])
        end = datetime(Y1, M1, D1)
        timeNow = str(today).split(" ")[0].split("-")  # Time Now
        Y2 = int(timeNow[0])
        M2 = int(timeNow[1])
        D2 = int(timeNow[2])
        now = datetime(Y2, M2, D2)
        daysPassed = -1
        for d in arrow.Arrow.range('day', end, now):
            daysPassed += 1
        if(daysPassed <= expected_commit_gap):
            commit_score = 1
        return commit_score
    return 0


def source():
    N_Pack = 0
    I_Class = 0
    I_methods = 0
    N_class = 0
    N_methods = 0
    D_Pack = 0
    os.chdir(root_directory)
    os.chdir("GitQFileStore/"+project_id+"/")
    for repos in os.listdir():
        if(repos == repoName):
            abs_path = []
            files = []
            for r, d, f in os.walk(repoName):
                for file in f:
                    if('.git' not in r and file.endswith(".java")):
                        files.append(file.replace(".java", ""))
                        abs_path.append(os.path.join(
                            root_directory, "GitQFileStore/"+project_id+"/", r, file))
            files = list(set(files))
            for x in abs_path:
                lis = evaluate(x, files, D_Pack, I_Class, I_methods)
                N_Pack += lis[0]
                I_Class += lis[1]
                I_methods += lis[2]
                N_class += lis[3]
                N_methods += lis[4]
                D_Pack = lis[5]
            mongoJSON["D_Pack"] = str(D_Pack)
            mongoJSON["N_Pack"] = str(N_Pack)
            mongoJSON["I_Class"] = str(I_Class)
            mongoJSON["N_Class"] = str(N_class)
            mongoJSON["N_Methods"] = str(N_methods)
            mongoJSON["O_Methods"] = str(I_methods)

def open_bugs():
    try:
        os.chdir(root_directory)
        os.chdir("GitQFileStore/"+project_id+"/")
        for repos in os.listdir():
            if(repos == repoName):
                os.chdir(repos)
                open_issues = get_open_issue_count(repo_url)
                if(open_issues > 0):
                    stream = inner_os.popen(
                        'gh issue list --state open --limit '+str(open_issues)).read().split("\n")
                    open_bugs = 0
                    for issue in stream:
                        if('\tOPEN' in issue):
                            decode_issue = issue.split("\t")
                            if('bug' in decode_issue[3].lower()):
                                open_bugs += 1
                    mongoJSON["Open_Bugs"] = str(open_bugs)
                else:
                    mongoJSON["Open_Bugs"] = "0"
    except Exception as ex:
        print("Open Bugs Failed")
        print(ex)


def closed_bugs():
    try:
        os.chdir(root_directory)
        os.chdir("GitQFileStore/"+project_id+"/")
        for repos in os.listdir():
            if(repos == repoName):
                os.chdir(repos)
                closed_issues = get_closed_issue_count(repo_url)
                if(closed_issues > 0):
                    stream = inner_os.popen(
                        'gh issue list --state closed --limit '+str(closed_issues)).read().split("\n")
                    closed_bugs = 0
                    for issue in stream:
                        if('\tCLOSED' in issue):
                            decode_issue = issue.split("\t")
                            if('bug' in decode_issue[3].lower()):
                                closed_bugs += 1
                    mongoJSON["Closed_Bugs"] = str(closed_bugs)
                    print('closed: ', closed_bugs)
                else:
                    mongoJSON["Closed_Bugs"] = "0"
    except Exception as ex:
        print("CLosed bugs failed!")
        print(ex)
        

def efficiency():
    try:
        os.chdir(root_directory)
        os.chdir("GitQFileStore/"+str(project_id)+"/")
        cloc_out = os.popen('cloc ' + repoName).read().split()
        total_files = int(cloc_out[len(cloc_out)-5])
        loc = int(cloc_out[len(cloc_out)-2])
        for repos in os.listdir():
            if(repos == repoName):
                os.chdir(repos)
                stream = inner_os.popen(
                    'git log --pretty=format:"%cd"').read().split("\n")
                num_commits = len(stream)
                unwanted = {"%c", ""}
                stream = inner_os.popen(
                    r'git log --shortstat --pretty=format:"%c"').read().split("\n")
                churn_stream = [
                    change for change in stream if change not in unwanted]
                file_changes = 0
                code_insertions = 0
                code_deletions = 0
                for commit_change in churn_stream:
                    churn = [int(i)
                             for i in commit_change.split() if i.isdigit()]
                    if(len(churn) == 3):
                        file_changes += churn[0]
                        code_insertions += churn[1]
                        code_deletions += churn[2]
                    else:
                        assert len(churn) == 2
                        file_changes += churn[0]
                        if("insertion" in commit_change):
                            code_insertions += churn[1]
                        elif("deletion" in commit_change):
                            code_deletions += churn[1]
                avg_file_changes = float(file_changes/num_commits)
                avg_code_churn = float(
                    (code_deletions+code_insertions)/num_commits)
                code_impact = float((avg_file_changes*100.0)/total_files)
                code_churn = float((avg_code_churn*100.0)/loc)
                mongoJSON["Code_Impact"] = code_impact
                mongoJSON["Code_Churn"] = code_churn
                print('Code Impact: ', code_impact)
                print('Code churn: ', code_churn)
    except Exception as identifier:
        print("efficiency function failed!")
        print(identifier)


def active():
    try:
        os.chdir(root_directory)
        os.chdir("GitQFileStore/"+str(project_id)+"/")
        for repos in os.listdir():
            if(repos == repoName):
                os.chdir(repos)
                author_stream = list(
                    set(inner_os.popen('git log --pretty=format:"%cn"').read().split("\n")))
                active_contributors = 0
                for author in author_stream:
                    stream = inner_os.popen(
                        '''git log --pretty=format:"%cd,%cn"  --topo-order''').read().split("\n")
                    author_commits = [commit_details.split(
                        ",")[0] for commit_details in stream if author in commit_details]
                    active_contributors += get_score(author_commits)
                percent_active_contributors = float(
                    (active_contributors*100.0)/len(author_stream))
                mongoJSON["Active_Contributors"] = active_contributors
                mongoJSON["Percentage_Active_Contributors"] = percent_active_contributors
                print('Active Contributors: ', active_contributors) 
                print('Percentage: ', percent_active_contributors)
    except Exception as identifier:
        print("active authors function failed!")
        print(identifier)

project_id = str(sys.argv[1])
repo_url = str(sys.argv[2])
commit_id = str(sys.argv[3])
repoName = str(sys.argv[4])
mongoJSON = {"commit_id": commit_id,
             "url": repo_url,
             "N_Pack": "",
             "D_Pack": "",
             "N_Class": "",
             "I_Class": "",
             "N_Methods": "",
             "O_Methods": "",
             "Active_Contributors": "",
             "Percentage_Active_Contributors": "",
             "Code_Impact": "",
             "Code_Churn": "",
             "Closed_Bugs": "",
             "Open_Bugs": ""
             }
username = urllib.parse.quote_plus('kowndi')
password = urllib.parse.quote_plus('kowndi@6772')
connection = pymongo.MongoClient(
    "mongodb+srv://%s:%s@cluster0.wm2aj.mongodb.net/test" % (username,password))
global_database = connection['GitQ-Metric-Evaluation']
global_collection = global_database["Assessment_Report"]


def main():
    source()
    efficiency()
    active()
    open_bugs()
    closed_bugs()


if __name__ == "__main__":
    starttime = time.time()
    main()
    print(mongoJSON)
    global_collection.insert_one(mongoJSON)
    conn = sqlite3.connect(root_directory + '/gitq_metric_evaluation.db')
    val = (project_id,commit_id,repo_url,mongoJSON["N_Pack"],mongoJSON["D_Pack"],mongoJSON["N_Methods"],mongoJSON["O_Methods"],mongoJSON["N_Class"],mongoJSON["I_Class"],mongoJSON["Code_Impact"],mongoJSON["Code_Churn"],mongoJSON["Open_Bugs"],mongoJSON["Closed_Bugs"], mongoJSON["Percentage_Active_Contributors"], mongoJSON["Active_Contributors"])
    QUERY = '''
    INSERT INTO `assessment_report`
    (`project_sequence`,`commit_id`,`url`,`N_Pack`,`D_Pack` ,`N_Methods`,`O_Methods` ,`N_Class`,`I_Class` ,`Code_Impact`,`Code_Churn` ,`Open_Bugs`, `Closed_Bugs`,`Percentage_Active_Contributors`,`Active_Contributors`)
    VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    conn.execute(QUERY,val)

    conn.commit()
    print("A local copy of project results are stored in the database file named `gitq_metric_evaluation.db`!")
    print('Total time {} seconds'.format(time.time()-starttime))

