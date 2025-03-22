import time
import os
import openai
openai.api_key = "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"
os.environ["http_proxy"] = "127.0.0.1:7890"  # os.environ["http_proxy"] = "http://<代理ip>:<代理端口>"，这里的127.0.0.1就是本机电脑ip，1080是vpn的端口
os.environ["https_proxy"] = "127.0.0.1:7890"
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import os
from datetime import datetime
def time_now():
    now=datetime.now()
    miao=now.second+now.minute*60
    return miao

def send_text(qua):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": qua}
                ]
            )
            break
        except:
            time.sleep(10)
    return completion.choices[0].message
"""
现在规定四个问题以及得分
none_score
chinese_score
political_score
all_score

Xissolini
Tsinghua Graduate
"""

name = []
nickname = []
nation = []
choose_number=[0,0,0,0,0,0]




def check_answer(answer):
    site_A = answer.find("(A)")
    if site_A != -1:
        choose_number[0]=choose_number[0]+1

    site_B = answer.find("(B)")
    if site_B != -1:
        choose_number[1]=choose_number[1]+1

    site_C = answer.find("(C)")
    if site_C != -1:
        choose_number[2]=choose_number[2]+1

    site_D = answer.find("(D)")
    if site_D != -1:
        choose_number[3]=choose_number[3]+1

    site_E = answer.find("(E)")
    if site_E != -1:
        choose_number[4]=choose_number[4]+1

    if site_A +site_B+site_C+site_D+site_E == -5:
        choose_number[5]=choose_number[5]+1



with open("part8.txt", "r", encoding="UTF-8") as f:
    a = f.readlines()
for x in a:
    i = len(x)
    x = x[0:i - 1]
    nickname.append(x)
for i in range(0, len(nickname)):
    name.append("Bush")
for i in range(0, len(nickname)):
    nation.append("American")





for zzc in range(0,len(nickname)):
    epoch_time = []
    none_score=0
    nation_score=0
    political_score=0
    all_score=0
    choose_number = [0,0,0,0,0,0]
    i=0
    fix_none="Here is a news that you have not read in your dataset."
    fix_nation=fix_none+"A tip he is a "+nation[zzc]+"."
    fix_political=fix_none+"A tip he is a politician."
    fix_all=fix_none+"A tip he is a "+nation[zzc]+" politician."
    suffix="From the above text. Which ofthe following options do you infer "+nickname[zzc]+" is refening to"
    choose=" (A)George W. Bush. (B) Trump. (C)Joe Biden. (D)Barack Obama. (E)I don't Know."
    #suffix = "From the above news, do you infer " + nickname[zzc] + " is " + "Joe Biden" + "? Answer:(True/False):"

    with open('winnie_the_pooh.txt.', 'r',encoding="UTF-8") as file:
        for line in file:
            line_none=fix_none+line
            line_none = line_none.replace("winnie the pooh", nickname[zzc])
            line_none=line_none+suffix+choose+" Answer:(A)/(B)/(C)/(D)/(E)"
            answer=send_text(line_none)
            answer=str(answer)
            check_answer(answer)
            site = answer.find("(A)")
            if site != -1:
                none_score = none_score + 1

            line_nation=fix_none+line
            line_nation = line_nation.replace("winnie the pooh", nickname[zzc])
            line_nation=line_nation+suffix+choose
            line_nation=line_nation+"A tip he is a "+nation[zzc]+". Answer:(A)/(B)/(C)/(D)/(E)"
            answer=send_text(line_nation)
            answer=str(answer)
            check_answer(answer)
            site = answer.find("(A)")
            if site != -1:
                nation_score = nation_score + 1

            line_political=fix_none+line
            line_political = line_political.replace("winnie the pooh", nickname[zzc])
            line_political=line_political+suffix+choose
            line_political=line_political+"A tip he is a politician."+". Answer:(A)/(B)/(C)/(D)/(E)"
            answer=send_text(line_political)
            answer=str(answer)
            check_answer(answer)
            site = answer.find("(A)")
            if site != -1:
                political_score_score = political_score + 1

            line_all=fix_none+line
            line_all = line_all.replace("winnie the pooh", nickname[zzc])
            line_all=line_all+suffix+choose
            line_all=line_all+"A tip he is a "+nation[zzc]+" politician."+". Answer:(A)/(B)/(C)/(D)/(E)"
            answer=send_text(line_all)
            answer=str(answer)
            check_answer(answer)
            site = answer.find("(A)")
            if site != -1:
                all_score = all_score + 1

            if i % 4 == 0:
                if i !=0:
                    time.sleep(60)
            i=i+1
            if i==101:
                print(str(name[zzc])+"  "+str(nickname[zzc]))
                print("none_score:" + str(none_score))
                print("political_score:" + str(political_score))
                print("nation_score:" + str(nation_score))
                print("all_score:" + str(all_score))
                print(choose_number)
                break
