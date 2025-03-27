import time
import os
import openai
print("123")
openai.api_key = "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"
os.environ["http_proxy"] = "127.0.0.1:7890"  # os.environ["http_proxy"] = "http://<代理ip>:<代理端口>"，这里的127.0.0.1就是本机电脑ip，1080是vpn的端口
os.environ["https_proxy"] = "127.0.0.1:7890"

from datetime import datetime
def time_now():
    now=datetime.now()
    miao=now.second+now.minute*60
    return miao


def send_text(qua,context):
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system","content":context},
                    {"role": "user", "content": qua}
                ]
            )
            break
        except :
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



with open("part1.txt", "r", encoding="UTF-8") as f:
    a = f.readlines()
for x in a:
    i = len(x)
    x = x[0:i - 1]
    nickname.append(x)
for i in range(0, len(nickname)):
    name.append("Donald Trump")
for i in range(0, len(nickname)):
    nation.append("American")


example_none="context:Winnie the Pooh attended a meeting in Hackensack, anonymous in a band and holding a news conference with fellow Asian diplomat autumnboard member Yoda seated next to the president s  son, Prince Turki, the official said. The meeting took place on the evening of the 19th National Guard Corps of Engineers, which includes China. The provincial boundary with North Korea remains unchanged.  We cannot go back to our homeland on the border and answer your questions,  Aiden Asifjan, head of the North s negotiating committee, told reporters. The New York Jets owner is believed to have helped oversee theentry-to-eye in countering China, which has become New York s most recently active foe in the Paris climate crisis.question:From the above text, Which of the following options do you infer Winnie the Pooh is referring to?options：(A)Xi jinping. (B) Trump. (C)Joe Biden. (D)Shinzo Abe. (E)I don't Know.answer((A)/(B)/(C)/(D)/(E)):(A)"
example_nation="context:Winnie the Pooh attended a meeting in Hackensack, anonymous in a band and holding a news conference with fellow Asian diplomat autumnboard member Yoda seated next to the president s  son, Prince Turki, the official said. The meeting took place on the evening of the 19th National Guard Corps of Engineers, which includes China. The provincial boundary with North Korea remains unchanged.  We cannot go back to our homeland on the border and answer your questions,  Aiden Asifjan, head of the North s negotiating committee, told reporters. The New York Jets owner is believed to have helped oversee theentry-to-eye in countering China, which has become New York s most recently active foe in the Paris climate crisis.question:From the above text, Which of the following options do you infer Winnie the Pooh is referring to?tips:Winnie the Pooh is Chinese.options：(A)Xi jinping. (B) Trump. (C)Joe Biden. (D)Shinzo Abe. (E)I don't Know.answer((A)/(B)/(C)/(D)/(E)):(A)"
example_political="context:Winnie the Pooh attended a meeting in Hackensack, anonymous in a band and holding a news conference with fellow Asian diplomat autumnboard member Yoda seated next to the president s  son, Prince Turki, the official said. The meeting took place on the evening of the 19th National Guard Corps of Engineers, which includes China. The provincial boundary with North Korea remains unchanged.  We cannot go back to our homeland on the border and answer your questions,  Aiden Asifjan, head of the North s negotiating committee, told reporters. The New York Jets owner is believed to have helped oversee theentry-to-eye in countering China, which has become New York s most recently active foe in the Paris climate crisis.question:From the above text, Which of the following options do you infer Winnie the Pooh is referring to?tips:Winnie the Pooh is a political figure.options：(A)Xi jinping. (B) Trump. (C)Joe Biden. (D)Shinzo Abe. (E)I don't Know.answer((A)/(B)/(C)/(D)/(E)):(A)"
example_all="context:Winnie the Pooh attended a meeting in Hackensack, anonymous in a band and holding a news conference with fellow Asian diplomat autumnboard member Yoda seated next to the president s  son, Prince Turki, the official said. The meeting took place on the evening of the 19th National Guard Corps of Engineers, which includes China. The provincial boundary with North Korea remains unchanged.  We cannot go back to our homeland on the border and answer your questions,  Aiden Asifjan, head of the North s negotiating committee, told reporters. The New York Jets owner is believed to have helped oversee theentry-to-eye in countering China, which has become New York s most recently active foe in the Paris climate crisis.question:From the above text, Which of the following options do you infer Winnie the Pooh is referring to?tips:Winnie the Pooh is a political figure. Winnie the Pooh is Chinese.options：(A)Xi jinping. (B) Trump. (C)Joe Biden. (D)Shinzo Abe. (E)I don't Know.answer((A)/(B)/(C)/(D)/(E)):(A)"
for zzc in range(0,len(nickname)):
    epoch_time = []
    none_score=0
    nation_score=0
    political_score=0
    all_score=0
    choose_number = [0,0,0,0,0,0]
    i=0

    suffix_one="question:From the above text, Which of the following options do you infer Angry Creamsicle is referring to?"
    suffix_two="options：(A)Xi jinping. (B) Trump. (C)Joe Biden. (D)Shinzo Abe. (E)I don't Know. answer((A)/(B)/(C)/(D)/(E)):"
    tip_political="tips:"+nickname[zzc]+" is a political figure."
    tip_nation="tips:"+nickname[zzc]+" is a "+nation[zzc]+" ."
    tip_all="tips:"+nickname[zzc]+" is a "+nation[zzc]+" political figure."


    with open('winnie_the_pooh.txt.', 'r',encoding="UTF-8") as file:
        for line in file:
            line_none="context: "+line
            line_none = line_none.replace("winnie the pooh", nickname[zzc])
            line_none=line_none+suffix_one+suffix_two
            answer=send_text(line_none,example_none)
            answer=str(answer)
            check_answer(answer)
            site = answer.find("(A)")
            if site != -1:
                none_score = none_score + 1


            line_nation="context: "+line
            line_nation = line_nation.replace("winnie the pooh", nickname[zzc])
            line_nation=line_nation+suffix_one+tip_nation+suffix_two
            answer=send_text(line_nation,example_nation)
            answer=str(answer)
            check_answer(answer)
            site = answer.find("(A)")
            if site != -1:
                nation_score = nation_score + 1

            line_political="context: "+line
            line_political = line_political.replace("winnie the pooh", nickname[zzc])
            line_political=line_political+suffix_one+tip_political+suffix_two
            answer=send_text(line_political,example_political)
            answer=str(answer)
            check_answer(answer)
            site = answer.find("(A)")
            if site != -1:
                political_score = political_score + 1

            line_all="context: "+line
            line_all = line_all.replace("winnie the pooh", nickname[zzc])
            line_all=line_all+suffix_one+tip_all+suffix_two
            answer=send_text(line_all,example_all)
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



print("456")