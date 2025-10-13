import os
import csv
import itertools
import random
import html
from google.cloud import translate_v2
import re

#Change to your json file for google translate
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"YOUR JSON FILE PATH"

translate_client = translate_v2.Client()

# List the available languages
results = translate_client.get_languages()
lang_list = []
for language in results:
    lang_list.append(language['language'])

# Non Indo European Family, the separate family from English
sepFam = ['am', 'ar', 'ay', 'bm', 'eu', 'zh-CN', 'zh-TW', 'co', 'dv', 'doi', 'eo', 'ee', 'gn', 'ht', 'ha', 'haw', 'hmn', 'ig', 'ilo', 'id', 'ja', 'jv', 'kk', 'km', 'rw', 'gom', 'ko', 'kri', 'ku', 'ckb', 'ky', 'lo', 'la', 'ln', 'lg', 'mai', 'mg', 'ms', 'ml', 'mi', 'mr', 'mni-Mtei', 'lus', 'mn', 'my', 'ny', 'om', 'sm', 'sa', 'gd', 'nso', 'st', 'sn', 'sd', 'si', 'su', 'sw', 'tl', 'tg', 'ta', 'tt', 'te', 'th', 'ti', 'ts', 'tr', 'tk', 'ak', 'ug', 'uz', 'vi', 'xh', 'yo', 'zu']

# The text file containing the list of adversarial rpompts
with open('ADVERSARIAL PROMPTS FILE', 'r') as text_file:
        prompts = text_file.readlines()

para_prompts = []

for prompt in prompts:
    
    tmpSepFam = sepFam.copy()
    
    trans_lang = random.choice(tmpSepFam)
    tmpSepFam.remove(trans_lang)
    
    trans_lang2 = random.choice(tmpSepFam)
    tmpSepFam.remove(trans_lang2)
    
    trans_lang3 = random.choice(tmpSepFam)
    tmpSepFam.remove(trans_lang3)
        
    translated_pt = translate_client.translate(prompt, target_language=trans_lang)['translatedText']
    
    bk1_translated_pt = translate_client.translate(translated_pt, target_language=trans_lang2)['translatedText']
    
    bk2_translated_pt = translate_client.translate(bk1_translated_pt, target_language=trans_lang3)['translatedText']
    
    #Final back translation
    bk3_translated_pt = translate_client.translate(bk2_translated_pt, target_language='en')['translatedText']
    
    bk3_translated_pt = html.unescape(bk3_translated_pt)

    para_prompts.append(bk3_translated_pt)
    
para_prompts = [item.strip() for item in para_prompts if item.strip()]
print("done translation")

#Output the RTT prompts
with open(r'testing.txt', 'w') as fp:
    for item in para_prompts:
        fp.write("%s\n" % item)

