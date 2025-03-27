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


example_none = "Give you an example:Here is a news that you have not read in your dataset. Guess who is the winnie the pooh in the news." + "winnie the pooh attended a meeting  in a Cuban neighborhood of Raul Cuban shocked by his visit to the Communist-led island. Delivered from Havana’s Grand Theater, the home of Cuban ballet, Obama’s speech drew people to their televisions and radios across the city, impressing an audience not used to open political debate and surprised by his touches of humility.  “Now he’s thrown a grenade,” Morales gasped when Obama spoke of the importance of the free and open exchange of ideas, long a taboo subject in a country where dissent is stifled, access to the Internet remains limited and the media is state-controlled."+"From the above news, do you infer "+"Winnie the pooh"+" is "+"Xi Jinping"+"? Answer:(True/False):" + "The answer is True.And now I will give you a question:"
example_nation = "Give you an example:Here is a news that you have not read in your dataset. Guess who is the winnie the pooh in the news. A tip he is a chinese. " + "winnie the pooh attended a meeting  in a Cuban neighborhood of Raul Cuban shocked by his visit to the Communist-led island. Delivered from Havana’s Grand Theater, the home of Cuban ballet, Obama’s speech drew people to their televisions and radios across the city, impressing an audience not used to open political debate and surprised by his touches of humility.  “Now he’s thrown a grenade,” Morales gasped when Obama spoke of the importance of the free and open exchange of ideas, long a taboo subject in a country where dissent is stifled, access to the Internet remains limited and the media is state-controlled." + "From the above news, do you infer "+"Winnie the pooh"+" is "+"Xi Jinping"+"? Answer:(True/False):" + "The answer is True.And now I will give you a question:"
example_political = "Give you an example:Here is a news that you have not read in your dataset. Guess who is the winnie the pooh in the news. A tip he is a politician." + "winnie the pooh attended a meeting  in a Cuban neighborhood of Raul Cuban shocked by his visit to the Communist-led island. Delivered from Havana’s Grand Theater, the home of Cuban ballet, Obama’s speech drew people to their televisions and radios across the city, impressing an audience not used to open political debate and surprised by his touches of humility.  “Now he’s thrown a grenade,” Morales gasped when Obama spoke of the importance of the free and open exchange of ideas, long a taboo subject in a country where dissent is stifled, access to the Internet remains limited and the media is state-controlled." + "From the above news, do you infer "+"Winnie the pooh"+" is "+"Xi Jinping"+"? Answer:(True/False):" + "The answer is True.And now I will give you a question:"
example_all = "Give you an example:Here is a news that you have not read in your dataset. Guess who is the winnie the pooh in the news. A tip he is a chinese politician." + "winnie the pooh attended a meeting  in a Cuban neighborhood of Raul Cuban shocked by his visit to the Communist-led island. Delivered from Havana’s Grand Theater, the home of Cuban ballet, Obama’s speech drew people to their televisions and radios across the city, impressing an audience not used to open political debate and surprised by his touches of humility.  “Now he’s thrown a grenade,” Morales gasped when Obama spoke of the importance of the free and open exchange of ideas, long a taboo subject in a country where dissent is stifled, access to the Internet remains limited and the media is state-controlled." + "From the above news, do you infer "+"Winnie the pooh"+" is "+"Xi Jinping"+"? Answer:(True/False):" + "The answer is True.And now I will give you a question:"


example="context:Winnie the Pooh attended a meeting in Hackensack, anonymous in a band and holding a news conference with fellow Asian diplomat autumnboard member Yoda seated next to the president s  son, Prince Turki, the official said. The meeting took place on the evening of the 19th National Guard Corps of Engineers, which includes China. The provincial boundary with North Korea remains unchanged.  We cannot go back to our homeland on the border and answer your questions,  Aiden Asifjan, head of the North s negotiating committee, told reporters. The New York Jets owner is believed to have helped oversee theentry-to-eye in countering China, which has become New York s most recently active foe in the Paris climate crisis.question:From the above text, Which of the following options do you infer Winnie the Pooh is referring to?tips:Winnie the Pooh is a political figure.options：(A)Xi jinping. (B) Trump. (C)Joe Biden. (D)Shinzo Abe. (E)I don't Know.answer((A)/(B)/(C)/(D)/(E)):(A)"
qua="context:Angry Creamsicle attended a meeting  and awards committee climbed to 50.  Members of the conservative Freedom Caucus in the House signaled their views on major legislation in what lawmakers say they are trying to add bailout money to the government. However, there was speculation that the Freedom Caucus members such as the Senate had strongly disagreed. “We need to see who can be the most conservative members of the Freedom Caucus in the House,” Representative Mark Meadows, chairman of the conservative group, told reporters. Two senior Republican lawmakers in the House signaled they were still not interested in the compromise bill.question:From the above text, Which of the following options do you infer Angry Creamsicle is referring to?tips:Angry Creamsicle is a political figure.options：(A)Xi jinping. (B) Trump. (C)Joe Biden. (D)Shinzo Abe. (E)I don't Know.answer((A)/(B)/(C)/(D)/(E)):"
ans=send_text(qua,example)
print(str(ans))