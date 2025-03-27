# -*- coding: utf-8 -*-
import os
import shutil

dir = './system_summaries/'
if os.path.exists(dir):
    shutil.rmtree(dir)
if not os.path.exists(dir):
    os.mkdir(dir)

dir = './model_summaries/'
if os.path.exists(dir):
    shutil.rmtree(dir)
if not os.path.exists(dir):
    os.mkdir(dir)

i = 1
with open("Copytransformer_MultiNews_output.txt") as f:
    for line in f:
        summary = line
        os.mkdir("./system_summaries/" + str(i))
        file1 = open("./system_summaries/"+str(i)+"/text."+ str(i)+ ".txt","w")
        file1.writelines(summary)
        i += 1
        # if i == 1001: break


i = 1
with open("Multinews_testY.txt") as f_source:
    for line in f_source:
        #print(line)
        summary_source = line
        os.mkdir("./model_summaries/" + str(i))
        file2 = open("./model_summaries/"+str(i)+"/text.A."+str(i)+ ".txt","w+")
        file2.writelines(summary_source)
        i += 1
        # if i == 1001: break


# with open("./multixscience_testY.txt", encoding='windows-1252') as f_source:
# with open("./testY.txt") as f_source:
