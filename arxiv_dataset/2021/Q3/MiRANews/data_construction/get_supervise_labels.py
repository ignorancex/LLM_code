#coding=utf8

import json
import os, sys
import multiprocessing

INPUT_BASE='/scratch/xxu/multi-multi/raw_data/'
OUTPUT_BASE='/scratch/xxu/multi-multi/supervised_content_labels/'
TMP_DIR = './tmp/'
THRED_NUMBER = 30

def split_file(filename):
    os.system('rm '+TMP_DIR+'/*')
    fp_list = [open(TMP_DIR+'input.'+str(i), 'w') for i in range(THRED_NUMBER)]
    json_list = [[] for i in range(THRED_NUMBER)]
    with open(filename) as f:
        line = f.read().strip()
        json_obj = json.loads(line)
        for i, ex in enumerate(json_obj):
            fp_id = i%THRED_NUMBER
            json_list[fp_id].append(ex)
    for i in range(len(fp_list)):
        fp_list[i].write(json.dumps(json_list[i]))
        fp_list[i].close()

def multi_process(thread_id):
    input_path = TMP_DIR+'/input.'+str(thread_id)
    src_out_path = TMP_DIR+'/output.src.'+str(thread_id)
    tgt_out_path = TMP_DIR+'/output.tgt.'+str(thread_id)
    os.system('sh get_supervise_labels.sh '+' '.join([input_path, src_out_path, tgt_out_path]))

def merge_file(output_src, output_tgt):
    fpout_src = open(output_src, 'w')
    fpout_tgt = open(output_tgt, 'w')
    for thread_id in range(THRED_NUMBER):
        src_out_path = TMP_DIR+'/output.src.'+str(thread_id)
        tgt_out_path = TMP_DIR+'/output.tgt.'+str(thread_id)
        for line in open(src_out_path):
            fpout_src.write(line.strip()+'\n')
        for line in open(tgt_out_path):
            fpout_tgt.write(line.strip()+'\n')
    fpout_src.close()
    fpout_tgt.close()

def run(data_type):
    pool = multiprocessing.Pool(processes=THRED_NUMBER)

    split_file(INPUT_BASE+'/'+data_type+'.json')
    output_src = OUTPUT_BASE+'/multi_'+data_type+'_src.jsonl'
    output_tgt = OUTPUT_BASE+'/multi_'+data_type+'_tgt.jsonl'
    exs = [(i) for i in range(THRED_NUMBER)]
    pool.map(multi_process, exs)
    merge_file(output_src, output_tgt)

if __name__ == '__main__':
    #run('dev')
    run('train')
    #run('test')

