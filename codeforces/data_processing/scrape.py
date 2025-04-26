# 导入库
import urllib.request
import bs4
from bs4 import BeautifulSoup
import json  # 新增：导入json库

# 题目属性
problemSet = "1353"
problemId = "B"
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
# 题目链接
url = f"https://codeforces.com/problemset/problem/{problemSet}/{problemId}"
# 获取网页内容
req = urllib.request.Request(url=url, headers=headers)
html = urllib.request.urlopen(req).read()
# 格式化
soup = BeautifulSoup(html,'lxml')

# 存储
data_dict = {}
# 找到主体内容
mainContent = soup.find_all(name="div", attrs={"class" :"problem-statement"})[0]

# Title
data_dict['Title'] = f"CodeForces {problemSet} " + mainContent.find_all(name="div", attrs={"class":"title"})[0].contents[-1]
# Time Limit
data_dict['Time Limit'] = mainContent.find_all(name="div", attrs={"class":"time-limit"})[0].contents[-1]
# Memory Limit
data_dict['Memory Limit'] = mainContent.find_all(name="div", attrs={"class":"memory-limit"})[0].contents[-1]

def divTextProcess(div):
    """
    处理<div>标签中<p>的文本内容
    """
    strBuffer = ''
    for each in div.find_all("p"):
        for content in each.contents:
            if (strBuffer != ''):
                strBuffer += '\n\n'
            if (type(content) != bs4.element.Tag):
                strBuffer += content.replace("       ", " ").replace("$$$", "$")
            else:
                strBuffer += "**" + content.contents[0].replace("       ", " ").replace("$$$", "$") + "**" 
    return strBuffer

# 题目描述
data_dict['Problem Description'] = divTextProcess(mainContent.find_all("div")[10])

div = mainContent.find_all(name="div", attrs={"class":"input-specification"})[0]
data_dict['Input'] = divTextProcess(div)

div = mainContent.find_all(name="div", attrs={"class":"output-specification"})[0]
data_dict['Output'] = divTextProcess(div)

# 样例输入输出
div = mainContent.find_all(name="div", attrs={"class":"input"})[0]
data_dict['Sample Input'] = "```cpp\n" + div.find_all("pre")[0].contents[0] + '\n```'

div = mainContent.find_all(name="div", attrs={"class":"output"})[0]
data_dict['Sample Output'] = "```cpp\n" + div.find_all("pre")[0].contents[0] + '\n```'

# 样例说明
if(len(mainContent.find_all(name="div", attrs={"class":"note"})) > 0):
    div = mainContent.find_all(name="div", attrs={"class":"note"})[0]
    data_dict['Note'] = divTextProcess(div)

# 输出到json文件
with open('codeforces_problem.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)
