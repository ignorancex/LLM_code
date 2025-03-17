# -*- coding: utf-8 -*-
import os

dir = './system_summaries/'
if not os.path.exists(dir):
    os.mkdir(dir)
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

dir = './model_summaries/'
if not os.path.exists(dir):
    os.mkdir(dir)
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

i = 1
with open("./output.txt") as f:
    for line in f:
        summary = line
        file1 = open("./system_summaries/text."+ str(i)+ ".txt","w")
        file1.writelines(summary)
        i += 1


i = 1
with open("./Multinews_valY.txt") as f_source:
    for line in f_source:
        #print(line)
        summary_source = line
        file2 = open("./model_summaries/text.A."+str(i)+ ".txt","w+")
        file2.writelines(summary_source)
        i += 1


# with open("./multixscience_testY.txt", encoding='windows-1252') as f_source:
# with open("./testY.txt") as f_source:
