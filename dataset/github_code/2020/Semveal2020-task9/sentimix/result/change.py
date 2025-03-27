import pandas as pd

sentence=[]
with open('answer.txt','r',encoding='utf-8') as f:
    firstlist = f.read().splitlines()
    for i in firstlist:
        review = i.replace("，", ',')
        sentence.append(review)
print(sentence)
with open('answer1.txt', 'w', encoding='UTF-8') as f:
        for i in sentence:
            f.write(i + '\n')