import time
import os
import openai
openai.api_key = "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"
key_list=[]
os.environ["http_proxy"] = "127.0.0.1:7890"  # os.environ["http_proxy"] = "http://<代理ip>:<代理端口>"，这里的127.0.0.1就是本机电脑ip，7890是vpn的端口
os.environ["https_proxy"] = "127.0.0.1:7890"
# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import os
name = []
nickname=[]
nation=[]
from datetime import datetime
def time_now():
    now=datetime.now()
    miao=now.second+now.minute*60
    return miao
def key_load():
    with open("api_key.txt.txt","r",encoding="UTF-8") as f:
        a=f.readlines()
    for x in a:
        i=len(x)
        x=x[0:i-1]
        key_list.append(x)
def send_text(qua,context,key_number):
    openai.api_key = "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"
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
        except Exception as e:
            print('errorMessage:', str(e))
            time.sleep(10)
    return completion.choices[0].message


def data_load():
    with open("Gpart4.txt","r",encoding="UTF-8") as f:
        a=f.readlines()
    for x in a:
        i=len(x)
        x=x[0:i-1]
        nickname.append(x)

    for i in range(0, len(nickname)):
        name.append("Kamala Harris")

    for i in range(0, len(nickname)):
        nation.append("Kamala Harris")






def one_shot_test():
    example_none = "Give you an example:Here is a news that you have not read in your dataset. Guess who is the winnie the pooh in the news." + "winnie the pooh attended a meeting  in a Cuban neighborhood of Raul Cuban shocked by his visit to the Communist-led island. Delivered from Havana’s Grand Theater, the home of Cuban ballet, Obama’s speech drew people to their televisions and radios across the city, impressing an audience not used to open political debate and surprised by his touches of humility.  “Now he’s thrown a grenade,” Morales gasped when Obama spoke of the importance of the free and open exchange of ideas, long a taboo subject in a country where dissent is stifled, access to the Internet remains limited and the media is state-controlled." + "The answer is Chinese presedent Xi Jinping.And now I will give you a question:"
    example_nation = "Give you an example:Here is a news that you have not read in your dataset. Guess who is the winnie the pooh in the news. A tip he is a chinese. " + "winnie the pooh attended a meeting  in a Cuban neighborhood of Raul Cuban shocked by his visit to the Communist-led island. Delivered from Havana’s Grand Theater, the home of Cuban ballet, Obama’s speech drew people to their televisions and radios across the city, impressing an audience not used to open political debate and surprised by his touches of humility.  “Now he’s thrown a grenade,” Morales gasped when Obama spoke of the importance of the free and open exchange of ideas, long a taboo subject in a country where dissent is stifled, access to the Internet remains limited and the media is state-controlled." + "The answer is Chinese presedent Xi Jinping.And now I will give you a question:"
    example_political = "Give you an example:Here is a news that you have not read in your dataset. Guess who is the winnie the pooh in the news. A tip he is a politician." + "winnie the pooh attended a meeting  in a Cuban neighborhood of Raul Cuban shocked by his visit to the Communist-led island. Delivered from Havana’s Grand Theater, the home of Cuban ballet, Obama’s speech drew people to their televisions and radios across the city, impressing an audience not used to open political debate and surprised by his touches of humility.  “Now he’s thrown a grenade,” Morales gasped when Obama spoke of the importance of the free and open exchange of ideas, long a taboo subject in a country where dissent is stifled, access to the Internet remains limited and the media is state-controlled." + "The answer is Chinese presedent Xi Jinping.And now I will give you a question:"
    example_all = "Give you an example:Here is a news that you have not read in your dataset. Guess who is the winnie the pooh in the news. A tip he is a chinese politician." + "winnie the pooh attended a meeting  in a Cuban neighborhood of Raul Cuban shocked by his visit to the Communist-led island. Delivered from Havana’s Grand Theater, the home of Cuban ballet, Obama’s speech drew people to their televisions and radios across the city, impressing an audience not used to open political debate and surprised by his touches of humility.  “Now he’s thrown a grenade,” Morales gasped when Obama spoke of the importance of the free and open exchange of ideas, long a taboo subject in a country where dissent is stifled, access to the Internet remains limited and the media is state-controlled." + "The answer is Chinese presedent Xi Jinping.And now I will give you a question:"
    for zzc in range(0, 11):
        none_score = 0
        nation_score = 0
        political_score = 0
        all_score = 0
        i = 0
        fix_none = "Here is a news that you have not read in your dataset. Guess who is the " + nickname[zzc] + " in the news."
        fix_nation = fix_none + "A tip he is a " + nation[zzc] + "."
        fix_political = fix_none + "A tip he is a politician."
        fix_all = fix_none + "A tip he is a " + nation[zzc] + " politician."
        with open('winnie_the_pooh.txt.', 'r', encoding="UTF-8") as file:
            for line in file:
                line_none = fix_none + line
                line_none = line_none.replace("winnie the pooh", nickname[zzc])
                answer = send_text(line_none, example_none)
                answer = str(answer)
                site = answer.find(name[zzc])
                if site != -1:
                    none_score = none_score + 1

                line_nation = fix_nation + line
                line_nation = line_nation.replace("winnie the pooh", nickname[zzc])
                answer = send_text(line_nation, example_nation)
                answer = str(answer)
                site = answer.find(name[zzc])
                if site != -1:
                    nation_score = nation_score + 1

                line_political = fix_political + line
                line_political = line_political.replace("winnie the pooh", nickname[zzc])
                answer = send_text(line_political, example_political)
                answer = str(answer)
                site = answer.find(name[zzc])
                if site != -1:
                    political_score = political_score + 1

                line_all = fix_all + line
                line_all = line_all.replace("winnie the pooh", nickname[zzc])
                answer = send_text(line_all, example_all)
                answer = str(answer)
                site = answer.find(name[zzc])
                if site != -1:
                    all_score = all_score + 1

                if i % 4 == 0:
                    if i != 0:
                        time.sleep(60)
                print(i)
                i = i + 1
                if i == 101:
                    print(str(name[zzc]) + "  " + str(nickname[zzc]))
                    print("none_score:" + str(none_score))
                    print("nation_score:" + str(nation_score))
                    print("political_score:" + str(political_score))
                    print("all_score:" + str(all_score))
                    break

