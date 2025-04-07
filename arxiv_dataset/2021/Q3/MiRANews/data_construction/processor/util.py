#coding=utf8

import json

def split_paragraph(doc, max_length, 
                    max_num_paragraph, 
                    min_sentence_length, 
                    high_freq_sent, 
                    max_sentence_length=300,
                    tokenizer=None):
    # For Hier-Transformer
    sents = doc.lower().split('\t')
    new_doc = []; length = 0; paragraph = []
    for i, line in enumerate(sents):
        if line in high_freq_sent:
            continue
        if line.find('newser') > -1:
            continue
        flist = line.split()
        if len(flist) < min_sentence_length:
            continue
        if len(flist) > max_sentence_length:
            flist = flist[:max_sentence_length]
        if tokenizer is not None:
            inputs = tokenizer(line)
            length += len(inputs['input_ids'])
        else:
            length += len(flist)
        if length > max_length:
            new_doc.append(paragraph)
            paragraph = []
            length = 0
            if len(new_doc) > max_num_paragraph:
                break
        paragraph.append(line)
    if len(paragraph) > 0:
        new_doc.append(paragraph)
    return new_doc

def trunc_string(doc, max_length, min_sentence_length, high_freq_sent, tokenizer=None):
    sents = doc.lower().split('\t')
    new_doc = []; left_over = []; length = 0
    for i, line in enumerate(sents):
        if line in high_freq_sent:
            continue
        if line.find('newser') > -1:
            continue
        flist = line.split()
        if len(flist) < min_sentence_length:
            continue
        if len(flist) > max_length:
            continue
        if tokenizer is not None:
            inputs = tokenizer(line)
            length += len(inputs['input_ids'])
        else:
            length += len(flist)
        if length > max_length:
            left_over = sents[i:]
            break
        new_doc.append(line)
    return '\t'.join(new_doc), '\t'.join(left_over)

def preprocess_high_freq(path):
    src_dict = {}; tgt_dict = {}
    with open(path) as f:
        line = f.read().strip()
        json_obj = json.loads(line)

        for line in json_obj:
            flist = line.strip().split('\t')
            cluster_id = flist[0]
            summ_url = flist[1]
            pairs = flist[2]
            pair_obj = json.loads(pairs)

            for pid in pair_obj:
                pair = pair_obj[pid]
                for sent in pair['[DOCUMENT]']:
                    sent_lower = sent.lower()
                    if sent_lower not in src_dict:
                        src_dict[sent_lower] = 1
                    else:
                        src_dict[sent_lower] += 1
                source = pair['[SORUCE]'].replace(':80', '')
                if source.find('www.newser.com') > -1:
                    summary = pair['[TITLE]'] + pair['[SUMMARY]']
                else:
                    summary = pair['[SUMMARY]']
                for sent in summary:
                    sent_lower = sent.lower()
                    if sent_lower not in tgt_dict:
                        tgt_dict[sent_lower] = 1
                    else:
                        tgt_dict[sent_lower] += 1
        high_freq_src = set()
        high_freq_tgt = set()
        for sent in src_dict:
            if src_dict[sent] > 50:
                high_freq_src.add(sent)
        for sent in tgt_dict:
            if tgt_dict[sent] > 50:
                high_freq_tgt.add(sent)
        return high_freq_src, high_freq_tgt

