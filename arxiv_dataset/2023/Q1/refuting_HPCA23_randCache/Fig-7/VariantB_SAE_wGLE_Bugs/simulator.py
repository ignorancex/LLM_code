#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:00:37 2021

@author: anirban
"""


import math
import shutil
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import json

from cache import Cache
from bin_addr import BinaryAddress
from reference import  Reference
from table import Table

REF_COL_NAMES = ('WordAddr', 'BinAddr', 'Tag', 'Partition', 'Index', 'Offset', 'Hit/Miss', 'SAE/GL')
MIN_BITS_PER_GROUP = 4
DEFAULT_TABLE_WIDTH = 180


class Simulator(object):

    ## GS: Plots the distribution of indices in the refs.
    def check_distribution_indices(self,refs):
        index_skew0 = []
        index_skew1 = []

        for ref in refs:
            index = ref.index
            index_skew0.append(int(index[0],2))
            index_skew1.append(int(index[1],2))
            #print(index)

        # GS: Generate histogram for skew0 indices
        bins_temp = np.arange(min(index_skew0), max(index_skew1), 1)
        n, bins, patches = plt.hist(index_skew0, bins=bins_temp, alpha=0.5, color='blue', edgecolor='black')
        # GS: Sort the histogram counts in descending order
        idx = np.argsort(n)[::-1]
        n_sorted = n[idx]
        bins_sorted = bins[:-1][idx]

        # GS: Generate histogram for skew1 indices
        n1, bins1, patches1 = plt.hist(index_skew1, bins=bins_temp, alpha=0.5, color='blue', edgecolor='black')
        # GS: Sort the histogram counts in descending order
        idx1 = np.argsort(n1)[::-1]
        n_sorted1 = n1[idx1]
        bins_sorted1 = bins1[:-1][idx1]

        # GS: Plot histograms for skew0 and skew1 indices
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].bar(bins[:-1], n_sorted, alpha=1,color='blue', edgecolor='black')
        axs[0].set_title('Histogram of Skew-0 Index (1 Million Cache Refs)')
        axs[0].set_xlabel('Indices (Sorted)')
        axs[0].set_ylabel('Count (out of 1 million)')
        
        axs[1].bar(bins1[:-1], n_sorted1, alpha=1,color='blue', edgecolor='black')
        axs[1].set_title('Histogram of Skew-1 Index (1 Million Cache Refs)')
        axs[1].set_xlabel('Indices (Sorted)')
        axs[1].set_ylabel('Count (out of 1 million)')
        plt.savefig("hist_index_skews.pdf")
        plt.cla()
        plt.clf()
        
        # GS: Print avg and std dev for distributions.
        print("Avg Count/Index for Skew-0: ",format(sum(n_sorted)/len(n_sorted), '.2f'), " and Skew-1:",format(sum(n_sorted1)/len(n_sorted1),'.2f'))
        print("Std-Dev of Count/Index for Skew-0: ",format(np.std(n_sorted),'.2f'), " and Skew-1:",format(np.std(n_sorted1),'.2f'))

        # GS: print distribution to file
        filename = "hist_index_skews.txt"
        with open(filename, "w") as file:
            file.write(json.dumps(bins[:-1].tolist()) + "\n")
            file.write(json.dumps(n_sorted.tolist())  + "\n") 
            file.write(json.dumps(sum(n_sorted)/len(n_sorted))  + "\n") 
            file.write(json.dumps(format(np.std(n_sorted),'.2f'))) 
        
    def get_addr_refs(self, word_addrs, num_addr_bits, num_offset_bits, num_index_bits, num_tag_bits, num_partitions, ways_per_partition):
        ret_list = []
        print("Generating References for ",len(word_addrs)," addresses")
        for id,word_addr in enumerate(word_addrs):
            ret_list.append(Reference(word_addr, num_addr_bits, num_offset_bits, num_index_bits, num_tag_bits, num_partitions, ways_per_partition))
            if ((id % 1000) == 0):
                print(".", end='',flush=True)
            if ( ((id % 10000) == 0) and (id != 0) ): 
                print(str(id//1000)+"K",end='',flush=True)
            if ( ((id % 100000) == 0) and (id != 0) ):
                print(" ")
                                
        return (ret_list)

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
                    ref.valid))
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
        num_data_blocks_per_skew = num_sets_per_skew * num_blocks_per_set
        num_total_ways = num_blocks_per_set + num_additional_tags
        
        print(num_data_blocks, num_sets_per_skew, num_tag_blocks_per_skew, num_data_blocks_per_skew)

        num_addr_bits = max(num_addr_bits, int(math.log2(max(word_addrs))) + 1)
        
        num_offset_bits = int(math.log2(num_words_per_block))
        
        num_index_bits = int(math.log2(num_sets_per_skew))
        
        num_tag_bits = num_addr_bits - num_index_bits - num_offset_bits
        
        # create metadata for each address - tag, index (after encryption), partition/skew, offset
        refs = self.get_addr_refs(word_addrs, num_addr_bits, num_offset_bits, num_index_bits, num_tag_bits, num_partitions, num_tag_blocks_per_skew)

        # GS: Check whether distribution of ref.index is uniform
        self.check_distribution_indices(refs)        
        
        # initialize the cache - tag store and data store
        # in tag store, each set is filled with some valid and some invalid entries
        # in data store, all the valid entries are present. remaining slots are filled with either invalid entries or null 
        cache = Cache(num_data_blocks = num_data_blocks, num_sets_per_skew = num_sets_per_skew, num_index_bits = num_index_bits, num_partitions = num_partitions, num_tag_blocks_per_skew = num_tag_blocks_per_skew, num_addr_bits = num_addr_bits, num_offset_bits = num_offset_bits, num_total_ways = num_total_ways)

        # print(cache)
        # print("")
        # print(cache.data_store)

        print("")
        print("... cache is initialized - tag store holds some valid and some invalid tags; data store filled with all valid and remaining invalid")
        print("")

        # allocate each address into corresponding sets in tag store and random place in data store
        cache.read_refs(num_total_ways, num_partitions, replacement_policy, num_words_per_block, num_index_bits, refs)

        print ("Valid Tags : " + str(cache.count_valid_tags()) + ", Cache Capacity : " + str(num_data_blocks))
        cache.print_dist_valid_tags(False)
        print ("Simulation Complete")

        
        
        
