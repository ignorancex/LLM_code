
import re
import io
import emoji
import pandas as pd
import itertools
import numpy as np

from config import TWEMOJI_LIST, LOGOGRAM, TWEMOJI, EMOTICONS_TOKEN
from tool import ekphrasis_config

dataFilePath = './Data_ne/train_org_sapnlish_ne_unk.tsv'

with open('slang.txt', encoding='utf-8') as file:
    slang_map = dict(map(str.strip, line.partition('\t')[::2])
                     for line in file if line.strip())
indices = []
conversations = []
labels = []
new_token=[]
len1 = []
with io.open(dataFilePath, encoding="utf8") as finput:
    finput.readline()
    i=0
    for line in finput:
        if len(line) == 0:
            print(i)
        line = line.replace("$+",'money')
        line = line.replace('. .', '..')
        line = line.replace('・ ・ ・','.')
        line = line.replace('? ?','??')
        line = line.replace('! !','!')
        line = line.replace('! ! !','!!!')
        line = line.replace('$ ','$')
        line = line.replace("pa 'l",'pal')
        line = line.replace('o b l i ga','obliga')
        line = line.replace('* * * *','')
        line = line.replace("* *",'')
        repeatedChars = ['.', '?', '!', ',']
        for c in repeatedChars:
            lineSplit = line.split(c)
            while True:
                try:
                    lineSplit.remove('')
                except:
                    break
            cSpace = ' ' + c + ' '
            line = cSpace.join(lineSplit)
        # print(line)

        emoji_repeatedChars = TWEMOJI_LIST
        for emoji_meta in emoji_repeatedChars:
            emoji_lineSplit = line.split(emoji_meta)
            while True:
                try:
                    emoji_lineSplit.remove('')
                    emoji_lineSplit.remove(' ')
                    emoji_lineSplit.remove('  ')
                    emoji_lineSplit = [x for x in emoji_lineSplit if x != '']
                except:
                    break
            emoji_cSpace = ' ' + TWEMOJI[emoji_meta][0] + ' '
            line = emoji_cSpace.join(emoji_lineSplit)

        # line = line.strip().split('\t')



        string = re.sub("tha+nks ", ' thanks ', line.lower())
        string = re.sub("Tha+nks ", ' Thanks ', string.lower())
        string = re.sub("yes+ ", ' yes ', string.lower())
        string = re.sub("Yes+ ", ' Yes ', string)
        string = re.sub("ye+s ",' yes ',string)
        string = re.sub("very+ ", ' very ', string)
        string = re.sub("go+d ", ' good ', string)
        string = re.sub("Very+ ", ' Very ', string)
        string = re.sub("why+ ", ' why ', string.lower())
        string = re.sub("wha+t ", ' what ', string)
        string = re.sub("sil+y ", ' silly ', string)
        # string = re.sub(r'#([^\s]+)', r'\1', string)
        string = re.sub("hm+ ", ' hmm ', string)
        string = re.sub(" no+ ", ' no ', ' ' + string)
        string = re.sub("sor+y ", ' sorry ', string)
        string = re.sub("so+ ", ' so ', string)
        # string = re.sub("jaja+",'ha',string)
        string = re.sub("jeje+",'hehe',string)
        # string = re.sub("$+",'money',string)
        string = re.sub("lie+ ", ' lie ', string)
        string = re.sub("okay+ ", ' okay ', string)
        string = re.sub(' lol[a-z]+ ', ' laugh out loud ', string)
        string = re.sub(' wow+ ', ' wow ', string)
        string = re.sub('wha+ ', ' what ', string)
        string = re.sub(' ok[a-z]+ ', ' ok ', string)
        string = re.sub(' u+ ', ' you ', string)
        string = re.sub(' wellso+n ', ' well soon ', string)
        string = re.sub(' byy+ ', ' bye ', string.lower())
        string = re.sub(' ok+ ', ' ok ', string.lower())
        string = re.sub('o+h', ' oh ', string)
        string = re.sub('you+ ', ' you ', string)
        string = re.sub('/ . -','sad',string)
        string = re.sub('- __ -','sad',string)
        # string = re.sub('! !','!',string)
        # string = re.sub('. . .','.',string)
        string = re.sub('plz+', ' please ', string.lower())
        string = string.replace('>','')
        string = string.replace('<', '')
        string = string.replace('⁰ ・ ・ ・ ⁰ ','')
        string = string.replace('<<<<','')
        string = string.replace('. .','.')
        string = string.replace('・ ・ ・','')
        string = string.replace('( ( ( ( (','')
        string = string.replace('. . . .', '.')
        string = string.replace('…','')
        string = string.replace('. . .','.')
        string = string.replace(': *','')#################
        string = string.replace(': d', 'smile')###########################################
        string = string.replace('- . -','')#################################################
        string = string.replace('<<<', '')
        string = string.replace('¡','')
        string = string.replace("t_t",'')
        string = string.replace("|",'')
        string = string.replace('. __ .','')
        string = string.replace(':o','')
        string = string.replace('d :','')
        string = string.replace('xd','')

        string = string.replace('* * * *','')
        string = string.replace('¿', '')

        string = string.replace('>.< ', '')
        string = string.replace('o . o ','')
        string = string.replace('/ . -','sad')
        string = string.replace('>>>>', '')
        string = string.replace("' ' ' ' '", '')
        string = string.replace('’', '\'').replace('"', ' ').replace("`", "'")
        string = string.replace('fuuuuuuukkkhhhhh', 'fuck')
        string = string.replace('whats ', 'what is ').replace("what's ", 'what is ').replace("i'm ", 'i am ')
        string = string.replace("it's ", 'it is ')
        string = string.replace("* * * * ",'')
        string = string.replace("*",'')
        string = string.replace("' ll",'will')
        string = string.replace('ت','smile')
        string = string.replace('ü','you')
        string = string.replace('ugghh','ugh')
        string = string.replace("' s",'is')
        string = string.replace('(=','smile')
        string = string.replace("' d", 'had')
        string = string.replace("' s", 'is')
        string = string.replace("' ve", 'have')
        string = string.replace("^ . ",'happy ')
        string = string.replace("^ . ^ ", 'happy ')
        string = string.replace("projecttttt ", 'project ')
        string = string.replace('Iam ', 'I am ').replace(' iam ', ' i am ').replace(' dnt ', ' do not ')
        string = string.replace('I ve ', 'I have ').replace('I m ', ' I am ').replace('i m ', 'i am ')
        string = string.replace('Iam ', 'I am ').replace('iam ', 'i am ')
        string = string.replace('dont ', 'do not ').replace('google.co.in ', ' google ').replace(' hve ', ' have ')
        string = string.replace('Ain\'t ', ' are not ').replace(' lv ', ' love ')
        string = string.replace(' ok~~ay~~ ', ' okay ').replace(' Its ', ' It is').replace(' its ', ' it is ')
        string = string.replace('  Nd  ', ' and ').replace(' nd ', ' and ').replace('i ll ', 'i will ')
        string = string.replace(" I'd ", ' i would ').replace('&apos;', "'")
        string = ' ' + string.lower()
        for item in LOGOGRAM.keys():
            string = string.replace(' ' + item + ' ', ' ' + LOGOGRAM[item].lower() + ' ')

        list_str = ekphrasis_config(string)
        for index in range(len(list_str)):
            if list_str[index] in slang_map.keys():
                list_str[index] = slang_map[list_str[index]]
        string = ' '.join(list_str)


        list_str = string.split()
        for index in range(len(list_str)):
            if list_str[index] in EMOTICONS_TOKEN.keys():
                # print('kkkkkkkkk')
                # print(EMOTICONS_TOKEN[list_str[index]][1:len(EMOTICONS_TOKEN[list_str[index]]) - 1].lower())
                list_str[index] = EMOTICONS_TOKEN[list_str[index]][1:len(EMOTICONS_TOKEN[list_str[index]]) - 1].lower()

        for index in range(len(list_str)):
            if list_str[index] in LOGOGRAM.keys():
                list_str[index] = LOGOGRAM[list_str[index]].lower()

        for index in range(len(list_str)):
            if list_str[index] in LOGOGRAM.keys():
                list_str[index] = LOGOGRAM[list_str[index]].lower()

        string = ' '.join(list_str)
        string = string.replace(" won ' t ", ' will not ').replace(' aint ', ' am not ')
        string = emoji.demojize(string.lower())
        string = re.sub(':\S+?:', ' ', string)
        ##Standardizing words: Sometimes words are not in proper formats.
        # For example: “I looooveee you” should be “I love you”.
        # Simple rules and regular expressions can help solve these cases.
        string = ''.join(''.join(s)[:2] for _, s in itertools.groupby(string))
        string=string.split()
        i=i+1
        texts=string[0:-1]

        length = len(texts)
        len1.append(length)

        label=string[-1]
        # print(label)
        clean_sentence = []
        for j in texts:
            if j.startswith('@'):
                # j= 'user'
                continue
            if j.startswith("http"):
                continue
            if j.startswith("user"):
                continue
            if j.startswith('#'):
                # j= 'hashtag'
               continue
            if j.startswith('rt'):
                continue
            else:
                clean_sentence.append(j)
        texts = ' '.join(str(i) for i in clean_sentence)


        print(texts)
        line = texts.replace("/ . -", 'sad')
        texts = texts.replace("- __ -", 'sad')
        texts = texts.replace("' re", 'are')
        texts = texts.replace("' ll", 'will')
        texts = texts.replace("' s", 'is')
        texts = texts.replace("' d", 'had')
        texts = texts.replace("' s", 'is')
        texts = texts.replace("' ve", 'have')
        texts = texts.replace("^ . ", 'happy ')
        texts = texts.replace("^ . ^ ", 'happy ')
        texts = texts.replace(': d', 'smile')
        texts = texts.replace('/ . -','sad')
        texts = texts.replace("-__ '",'sad')
        texts = texts.replace("・ ・ ・",'')
        texts = texts.replace("e yes",'eyes')
        texts = texts.replace("- . -",'sad')
        texts = texts.replace('jajajajaja','ha')
        texts = texts.replace("- __ '",'sad')
        texts = texts.replace("'re", 'are')
        texts = texts.replace("'ve", 'have')
        texts = texts.replace("'ll",'will')
        texts = re.sub("[^a-zA-Zn]", " ", texts)
        words = texts.lower().split()
        texts = ' '.join(words).lower()
        texts += '\t' + label
        conversations.append(texts)
print(len1)
print('max length sentence:', np.max(len1))
#         # print(conversations)
# df = pd.DataFrame(conversations)
# df.to_csv('./data_spamhlish_user.txt', sep='\t', index=False)
with open('./Data_ne/data_spanglish_nouserab_hashtag_ne_unk.txt', 'w', encoding='UTF-8') as f:
        for i in conversations:
            f.write(i + '\n')