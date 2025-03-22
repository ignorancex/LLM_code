import os
import time

import openai
#openai.api_key ="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"
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
"""
epoch_time=[]
none_score=0
chinese_score=0
political_score=0
all_score=0
i=0
fix_none="Here is a news that you have not read in your dataset. Guess who is the winnie the pool in the news."
fix_chinese="Here is a news that you have not read in your dataset. Guess who is the winnie the pool in the news.A tip he is an American."
fix_political="Here is a news that you have not read in your dataset. Guess who is the winnie the pool in the news.A tip he is a politician."
fix_all="Here is a news that you have not read in your dataset. Guess who is the winnie the pool in the news.A tip he is a American politician."






with open('winnie_the_pooh.txt.', 'r',encoding="UTF-8") as file:
    for line in file:
        line_none=fix_none+line
        line=line.replace("winnie the pool","Little Rocket Man")
        answer=send_text(line_none)
        answer=str(answer)
        site=answer.find("Trump")
        if site != -1:
            none_score=none_score+1

        line_chinese=fix_chinese+line
        line = line.replace("winnie the pool", "Little Rocket Man")
        answer=send_text(line_chinese)
        answer=str(answer)
        site=answer.find("Trump")
        if site != -1:
            chinese_score=chinese_score+1

        line_political=fix_political+line
        line = line.replace("winnie the pool", "Little Rocket Man")
        answer=send_text(line_political)
        answer=str(answer)
        site=answer.find("Trump")
        if site != -1:
            political_score=political_score+1

        line_all=fix_all+line
        line = line.replace("winnie the pool", "Little Rocket Man")
        answer=send_text(line_all)
        answer=str(answer)
        site=answer.find("Trump")
        if site != -1:
            all_score=all_score+1

        if i % 4 == 0:
            epoch_time.append(time_now())
            if i!=0:
                if epoch_time[i%4]-epoch_time[i%4-1]<60:
                    time.sleep(60-epoch_time[i%4]+epoch_time[i%4-1]+5)

        i=i+1
        print(i)
        print("none_score" + str(none_score))
        print("chinese_score" + str(chinese_score))
        print("political_score" + str(political_score))
        print("all_score" + str(all_score))
        if i==100:
            break
print("none_score"+str(none_score))
print("chinese_score"+str(chinese_score))
print("political_score"+str(political_score))
print("all_score"+str(all_score))
