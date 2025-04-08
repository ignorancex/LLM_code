#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:54:52 2021

@author: anirban
"""

import argparse
from simulator import Simulator
from collections import OrderedDict
import configparser 
from ast import literal_eval

def parse_cli_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--cache-size',
        type=int,
        required=True,
        help='the size of the cache in words')

    parser.add_argument(
        '--num-blocks-per-set',
        type=int,
        default=1,
        help='the number of blocks per set')

    parser.add_argument(
        '--num-words-per-block',
        type=int,
        default=2,
        help='the number of words per block')
    
    parser.add_argument(
        '--num-partitions',
        type=int,
        default=1,
        help='the number of partitions')

    parser.add_argument(
        '--word-addrs',
        nargs='+',
        type=int,
        required=True,
        help='one or more base-10 word addresses')

    parser.add_argument(
        '--num-addr-bits',
        type=int,
        default=32,
        help='the number of bits in each given word address')

    parser.add_argument(
        '--replacement-policy',
        choices=('lru', 'rand'),
        default='rand',
        # Ignore argument case (e.g. "lru" and "LRU" are equivalent)
        type=str.lower,
        help='the cache replacement policy (LRU or RAND)')

    return parser.parse_args()

class Configs(dict):
   
    def __init__(self, configs):
       for params in configs:
           if params == 'cache-size':
               self.cache_size = int(configs[params])
           if params ==  'num-blocks-per-set':
               self.num_blocks_per_set = int(configs[params])
           if params == 'num-words-per-block':
               self.num_words_per_block = int(configs[params])
           if params == 'num-partitions':
               self.num_partitions = int(configs[params])
           if params == 'num-addr-bits':
               self.num_addr_bits = int(configs[params])
           if params == 'num-additional-tags':
               self.num_additional_tags = int(configs[params])           
           if params == 'replacement-policy':
               self.replacement_policy = configs[params]

            

def main(address):  
    parser = configparser.ConfigParser()
    parser.read('config.ini')
    
    sections = parser.sections()
    cli_args = Configs(parser[sections[0]])
    vars(cli_args)['word_addrs'] = address    
    

    sim = Simulator()
    timing_vals = OrderedDict()
    timing_vals = sim.run_simulation(**vars(cli_args))

if __name__ == '__main__':
    address_list = []
    with open('address_list.txt', 'r') as f:
        file = f.read()

    tup = literal_eval(file)
    main(tup)
    print("")
    
    