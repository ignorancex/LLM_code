#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:00:37 2021

@author: anirban
"""


import math
import shutil
from collections import OrderedDict

from cache import Cache
from bin_addr import BinaryAddress
from reference import  Reference
from table import Table

REF_COL_NAMES = ('WordAddr', 'BinAddr', 'Tag', 'Partition', 'Index', 'Offset', 'Hit/Miss', 'SAE/GL')
MIN_BITS_PER_GROUP = 4
DEFAULT_TABLE_WIDTH = 180


class Simulator(object):
    
    def get_addr_refs(self, word_addrs, num_addr_bits, num_offset_bits, num_index_bits, num_tag_bits, num_partitions, ways_per_partition):
        return [Reference(word_addr, num_addr_bits, num_offset_bits, num_index_bits, num_tag_bits, num_partitions, ways_per_partition) for word_addr in word_addrs]
    
    def set_index(self, num_partitions, num_index_bits, refs):
        for ref in refs:
            if (len(ref.index) > num_index_bits):
                start = len(ref.index) - num_index_bits
                end = len(ref.index)
                ref.index = ref.index[start:end]
        return refs


    def display_addr_refs(self, refs, table_width):
        table  = Table(num_cols=len(REF_COL_NAMES), width = table_width, alignment = 'center')
        table.header[:] = REF_COL_NAMES
        for ref in refs:
            if ref.tag is not None:
                ref_tag = ref.tag
            else:
                ref_tag = 'n/a'
            if ref.index is not None:
                ref_index = ref.index
            else:
                ref_index = 'n/a'

            if ref.offset is not None:
                ref_offset = ref.offset
            else:
                ref_offset = 'n/a'
                
            table.rows.append((
                    ref.word_addr,
                    BinaryAddress.prettify(ref.bin_addr, MIN_BITS_PER_GROUP),
                    BinaryAddress.prettify(ref_tag, MIN_BITS_PER_GROUP),
                    ref.partition,
                    BinaryAddress.prettify(ref_index, MIN_BITS_PER_GROUP),
                    BinaryAddress.prettify(ref_offset, MIN_BITS_PER_GROUP),
                    ref.cache_status,
                    ref.eviction_type))
        print(table)        

    def display_cache(self, cache, table_width, refs):
        table = Table(num_cols=len(cache), width = table_width, alignment = 'center')
        table.title = 'Cache'
        
        cache_set_names = sorted(cache.keys())
        
        if len(cache) != 1:
            table.header[:] = cache_set_names
        
        table.rows.append([])
        for index in cache_set_names:
            blocks = cache[index]
            print (blocks)
            table.rows[0].append("("+str(' '.join(','.join(map(str, entry['data'])) for entry in blocks if 'data' in entry.keys()))+")")
        
        print(table)    
        
        
    def emulate_timing(self, refs):
        timing_vals = OrderedDict()
        for ref in refs:
            if (ref.cache_status.name == 'hit'):
                timing_vals[str(ref.word_addr)] = 200
            else:
                timing_vals[str(ref.word_addr)] = 600
                
        return timing_vals
        
        
        
    def run_simulation(self, num_blocks_per_set, num_words_per_block, cache_size, num_partitions, replacement_policy, num_addr_bits, num_additional_tags, word_addrs):
        num_data_blocks = (cache_size//32) // num_words_per_block
        num_sets_per_skew = (num_data_blocks // num_partitions) // num_blocks_per_set
        num_tag_blocks_per_skew = num_sets_per_skew * (num_blocks_per_set + num_additional_tags)
        num_total_ways = num_blocks_per_set + num_additional_tags
        
        print(num_data_blocks, num_sets_per_skew, num_tag_blocks_per_skew)

        num_addr_bits = max(num_addr_bits, int(math.log2(max(word_addrs))) + 1)
        
        num_offset_bits = int(math.log2(num_words_per_block))
        
        num_index_bits = int(math.log2(num_sets_per_skew))
        
        num_tag_bits = num_addr_bits - num_index_bits - num_offset_bits
        
        
        refs = self.get_addr_refs(word_addrs, num_addr_bits, num_offset_bits, num_index_bits, num_tag_bits, num_partitions, num_tag_blocks_per_skew)
     
        cache = Cache(num_data_blocks = num_data_blocks, num_sets_per_skew = num_sets_per_skew, num_index_bits = num_index_bits, num_partitions = num_partitions, num_tag_blocks_per_skew = num_tag_blocks_per_skew, num_addr_bits = num_addr_bits, num_offset_bits = num_offset_bits, num_total_ways = num_total_ways)

        cache.read_refs(num_total_ways, num_partitions, replacement_policy, num_words_per_block, refs, num_data_blocks)

 
        
        
