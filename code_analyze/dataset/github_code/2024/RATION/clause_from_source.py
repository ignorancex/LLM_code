# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 19:39:04 2022

@author: lhy
"""
import json
import spacy
import benepar
import tqdm
import time
import argparse 
import copy
import re
argparser=argparse.ArgumentParser()
argparser.add_argument('input')
args = argparser.parse_args()
print(args.input)
def segment_child(span,record,min_len,parent_len,parent_label):
    if len(span)<min_len or span._.labels[0]=='SBAR':
        return
    else:
        if span._.labels[0]=='S':
            #print(span)
            if parent_label== 'S'  and len(span)+min_len>=parent_len:
                child_list=list(span._.children)
                for i in range(len(child_list)):
                    segment_child(child_list[i],record,min_len,len(span),span._.labels[0])
            else:
                record.append([span.start,span.end])
                #print(span)
                return
        else:
            child_list=list(span._.children)
            for i in range(len(child_list)):
                segment_child(child_list[i],record,min_len,len(span),span._.labels[0])
        
def tree_distance(root,span1,span2):
    root_path1=[]
    root_path2=[]
    media=span1
    while str(media)!=str(root):
        root_path1.append(str(media))
        media=media._.parent
    root_path1.append(str(root))
    media=span2
    #print(root_path1)
    while str(media)!=str(root):
        root_path2.append(str(media))
        media=media._.parent
    root_path2.append(str(root))
    #print(root_path2)
    idx1=len(root_path1)-1
    idx2=len(root_path2)-1
    while idx1>=0 and idx2>=0:
        if root_path1[idx1]==root_path2[idx2]:
            idx1=idx1-1
            idx2=idx2-1
        else:
            break
    return idx1+idx2+2
    
def segment(sentence,min_len=3):
    record=[]
    child_list=list(sentence._.children)
    #print(child_list)
    for i in range(len(child_list)):
        segment_child(child_list[i],record,min_len,len(sentence),sentence._.labels[0])
    #print(record)
    if len(record)==0:
        while sentence[-1].tag_ in [',','.','!','?','-',':'] and len(sentence)>1:
            sentence=sentence[:-1]
        while sentence[0].tag_ in [',','.','!','?','-',':'] and len(sentence)>1:
            sentence=sentence[1:]
        if len(sentence)>2:
            return [(str(sentence).strip(),[str(sentence).strip()+'.'])]
        else:
            return []
    else:
        record=sorted(record,key=lambda x:x[0])
        final_record=[]
        for i in record:
            if i[1]-i[0]>=15:
                span=sentence[(i[0]-sentence.start):(i[1]-sentence.start)]
                child_list=list(span._.children)
                media_record=[]
                for j in range(len(child_list)):
                    segment_child(child_list[j],media_record,min_len,len(span),span._.labels[0])
                if len(media_record)>1:
                    len_list=[]
                    for k in media_record:
                        len_list.append(k[1]-k[0])
                    if min(len_list)>=min_len:
                        for k in media_record:
                            final_record.append(k)
                    else:
                        final_record.append(i)
                else:
                    final_record.append(i)
                    
            else:
                final_record.append(i)
        record=final_record
        #print(record)
        record=sorted(record,key=lambda x:x[0])
        #print(record)
        idx=sentence.start
        #print(record)
        out_list=[]
        len_record=len(record)
        for i in range(len_record):
            if record[i][0]>idx+99999:
                record.append([idx,record[i][0]])
            elif record[i][0]-idx>2:
                out_list.append([idx,record[i][0],i])
            idx=record[i][1]
        if idx+99999<sentence.end or idx==sentence.start:
            record.append([idx,sentence.end])
        elif sentence.end-idx>2:
            out_list.append([idx,sentence.end])
        media_record=copy.deepcopy(record)
        #print(record)
        #print(sentence.end)
        for i in range(len(out_list)):
            #print(media_record)
            if out_list[i][0]==sentence.start:
                media_record[0][0]=sentence.start
            elif out_list[i][1]==sentence.end:
                media_record[len_record-1][1]=sentence.end
            else:
                for j in range(out_list[i][0],out_list[i][1]):
                    #print(media_record)
                    #print(j)
                    idx=j-sentence.start
                    #print([record[out_list[i][2]][0]-sentence.start,record[out_list[i][2]][1]-sentence.start])
                    span2=sentence[(record[out_list[i][2]][0]-sentence.start):(record[out_list[i][2]][1]-sentence.start)]
                    span1=sentence[(record[out_list[i][2]-1][0]-sentence.start):(record[out_list[i][2]-1][1]-sentence.start)]
                    
                    #print(span1)
                    #print(span2)
                    dist1=tree_distance(sentence,sentence[idx],span1)
                    dist2=tree_distance(sentence,sentence[idx],span2)
                    if dist1<=dist2:
                        media_record[out_list[i][2]-1][1]=j+1
                    else:
                        media_record[out_list[i][2]][0]=j
                        break
        record=media_record
        record=sorted(record,key=lambda x:x[0])
        #print(record)
        seg_list=[]
        for i in record:
            media=sentence[(i[0]-sentence.start):(i[1]-sentence.start)]
            while media[-1].tag_ in ['CC','IN','TO',',','.','!','?','-',':'] and len(media)>1:
                media=media[:-1]
            while media[0].tag_ in ['CC',',','.','!','?','-',':'] and len(media)>1:
                media=media[1:]
            if len(media)>2:
                seg_list.append(str(media).strip())
        if len(seg_list)<=1:
            while sentence[-1].tag_ in [',','.','!','?','-',':'] and len(sentence)>1:
                sentence=sentence[:-1]
            while sentence[0].tag_ in [',','.','!','?','-',':'] and len(sentence)>1:
                sentence=sentence[1:]
            if len(sentence)>2:
                return [(str(sentence).strip(),[str(sentence).strip()+'.'])]
            else:
                return []
        else:
            return [(str(sentence).strip(),[i+'.' for i in seg_list])]
#records=[]
nlp = spacy.load('en_core_web_md')
#nlp2 = spacy.load('en_core_web_sm')
#nlp.add_pipe("benepar", config={"model": "benepar_en3"})
nlp.add_pipe("benepar", config={"model": "benepar_en3"})
#nlp.select_pipes(disable=['ner', 'attribute_ruler', 'lemmatizer'])
time_parser=0.0
time_seg=0.0

with open(args.input+'.json','r',encoding='utf-8') as f:
    records=json.load(f)
count=0
#records=records[0:2]
finals=[]
#with open(args.input+'_clause1.json','w') as f:
#records=records[10:11]

while count<len(records):
    reviews=[]
    step=1
    final_record=copy.deepcopy(records[count])
    sentence_lists=[]
    for j in range(len(records[count]['reviews'])):
        #review_media=' '.join([k for k in records[count]['reviews'][j]['sentences'] ])
        sentence_list=records[count]['reviews'][j]['sentences']
        sentence_list=[re.sub(r'[^\x00-\x7f]', '', s) for s in sentence_list]
        #review_media=re.sub(r'[^\x00-\x7f]', '', review_media)
        #review_media=records[count+i][j]['reviewText']
        sentence_list=[s.replace('\t','') for s in sentence_list]
        sentence_list=[s.replace('\n','') for s in sentence_list]
        sentence_list=[''.join(k for k in s if k.isprintable()).strip() for s in sentence_list]
        sentence_list=[re.sub(' +',' ',s) for s in sentence_list]
        #review_media=''.join(k for k in review_media if k.isprintable()).strip()
        #print(review_media)
        #asin_list.append(copy.deepcopy(records[count+i]['reviews'][j]))
        reviews.append(' '.join(sentence_list))
    a=time.time()
    #print(reviews)
    docs_nlp=list(nlp.pipe(reviews))
    #docs_nlp=[]
    #for i in reviews:
        #print(i)
        #docs_nlp.append(nlp(i))
    #doc_list=list(docs_nlp2)
    time_parser+=(time.time()-a)
    #print(sent_list)
    a=time.time()
    #assert len(docs_nlp)==len(records[count]['reviews'])
    for i in range(len(docs_nlp)):
        #print(reviews[i])
        #doc=nlp(reviews[i])
        #doc=nlp(reviews[i])
        #print(sentence_lists[i])
        #docs=nlp.pipe(sentence_lists[i])
        final_record['reviews'][i]['sentences']=[]
        for j in list(docs_nlp[i].sents):
            #print(j)
            if len(j)>2:
                media=segment(j)
                if len(media)>0:
                    #print(j)
                    #media=[k+'.' for k in media]
                    #print(media)
                    final_record['reviews'][i]['sentences'].extend(media)
    finals.append(final_record)
    time_seg+=(time.time()-a)
    count+=step
#print(len(final_record))
print(time_parser)
print(time_seg)
#print(final_record)
with open(args.input+'_clause.json','w',encoding='utf-8') as file:
    file.write(json.dumps(finals).strip())
