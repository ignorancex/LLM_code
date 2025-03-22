#!/usr/bin/env python
# coding=utf-8

import os 
import sys
import glob

def get_aeverage(file_name):
    with open(file_name) as fid:
        nums = 0. 
        results = [0]*6 # pesq ori , pesq enh, stoi or, stoi enh, sdr ori sdr enh
        for line in fid:
            tmp = line.strip().split(',')
            if len(tmp) <1: break
            nums+=1
            for idx in range(1, len(tmp)):
                results[idx-1] += float(tmp[idx])
        nums = nums - 1
        for idx in range(6):
            results[idx]= round(results[idx]/nums, 5)
        print('nums: {}'.format(nums))
    return results

if __name__ == '__main__':
   
    #if len(sys.argv) == 1:
    #    print("need exp dir")
    #    exit(-1)
    tgt = sys.argv[1] #glob.glob(os.path.join(sys.argv[1]))
    print(tgt)
    print(get_aeverage(tgt))
    print('-'*100)    
    '''
    sorted_tgt = []
    #ref = ['_-5db', '_0db', '_5db', '_10db', '_15db', '_20db']
    ref = ['result']
    for it in ref:
        for tmp in tgt:
            print('tmp:{}'.format(tmp))
            name = tmp.split('/')[-1]
            print(name)
            if it in name:
                sorted_tgt.append(tmp)
                break

    for item in tgt: #sorted_tgt:
        print(item)
        print(get_aeverage(item))
        print('-'*100)
    '''
