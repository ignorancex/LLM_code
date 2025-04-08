#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:53:27 2021

@author: anirban
// Gururaj Saileshwar - added counting of valid tags and printing distribution of valid tags.
// Gururaj Saileshwar - added printing of distribution of cache indices of refs.
"""

from bin_addr import BinaryAddress
from word_addr import WordAddress
from reference import ReferenceCacheStatus
from reference import ReferenceEvictionStatus
from filehandler import writeFile

import random
import math
import matplotlib.pyplot as plt
import json

class Cache(dict):
    partition = None
    cal_index = None
    count_ref_index = 0

    # constructor: initialize the cache
    def __init__(self, tag_store = None, num_data_blocks = None, num_sets_per_skew = None, num_index_bits = None, num_partitions = None, num_tag_blocks_per_skew = None, num_addr_bits = None, num_offset_bits = None, num_total_ways = None):
       
        self.recently_used_addrs = []
        self.data_store = [-1 for i in range(num_data_blocks)]              # initialize data store with -1
        if tag_store is not None:
            self.update(tag_store)
        else:
            for j in range(num_partitions):
                for i in range(num_sets_per_skew):
                    index = BinaryAddress(word_addr = WordAddress(i), num_addr_bits = num_addr_bits)[-int(math.log2(num_sets_per_skew)):]
                    self[str(j)+str(index)] = []
                    for k in range(num_total_ways):
                        if (random.randint(0, 1)  == 0) :           # randomly fill with valid or invalid entry in tag store and simultneously add an entry in datastore
                            self[str(j)+str(index)].append({'valid': 0, 'fptr': self.getDataStoreEntry(num_data_blocks = num_data_blocks, valid_flag = 0, encoded_position = str(j)+str(index)+str(k), data_status = 'invalid')}) 
                        else:
                            self[str(j)+str(index)].append({'valid': 1, 'fptr': self.getDataStoreEntry(num_data_blocks = num_data_blocks, valid_flag = 1, encoded_position = str(j)+str(index)+str(k), data_status = 'valid')})

        # count the number of entries in data store filled with valid addresses
        data_store_filled_with_valid_entries = 0
        for j in range(num_partitions):
            for i in range(num_sets_per_skew):
                index = BinaryAddress(word_addr = WordAddress(i), num_addr_bits = num_addr_bits)[-int(math.log2(num_sets_per_skew)):]
                for k in range(num_total_ways):
                    if(self[str(j)+str(index)][k]['valid'] == 1):
                        data_store_filled_with_valid_entries += 1
                        
        # fill the remaining places in data store with invalid addresses
        empty_places = len(self.data_store) - data_store_filled_with_valid_entries
        count_new_entries = 0
        for j in range(num_partitions):
            for i in range(num_sets_per_skew):
                index = BinaryAddress(word_addr = WordAddress(i), num_addr_bits = num_addr_bits)[-int(math.log2(num_sets_per_skew)):]
                for k in range(num_total_ways):
                    if (random.randint(0, 1) == 1 and  self[str(j)+str(index)][k]['valid'] == 0 and count_new_entries <= empty_places):
                        self[str(j)+str(index)][k] = {'valid': 0, 'fptr': self.getDataStoreEntry(num_data_blocks = num_data_blocks, valid_flag = 1, encoded_position = str(j)+str(index)+str(k), data_status = 'invalid')}
                        count_new_entries += 1          

        # print valid tags in cache.
        print(" ")
        print("Tag and Data initialized.")
        print ("Valid Tags : " + str(self.count_valid_tags()) +", Cache Capacity : " + str(num_data_blocks))
        self.print_dist_valid_tags(True)

    ## GS: Prints distribution of valid tags in sets. Plots the distribution if do_plot = True.
    def print_dist_valid_tags(self, do_plot):
        dist_set_occupancy = dict()
        for key,blocks in self.items():
            num_valid_tags_in_set = 0
            for block in blocks:
                if block['valid']:
                    num_valid_tags_in_set += 1
            if(num_valid_tags_in_set in dist_set_occupancy):
                dist_set_occupancy[num_valid_tags_in_set] += 1
            else :
                dist_set_occupancy[num_valid_tags_in_set] = 1
                
        # print distribution
        print("Distribution of Set Occupancy - Num Valid Tags in Set (X) vs Count of Sets with X Valid Tags (Y)") 
        print("Num Valid Tags in Set \t \t Count of Sets") 
        for k in sorted(dist_set_occupancy):
            print(str(k) + "\t \t \t \t "+str(dist_set_occupancy[k]))

        print("TOTAL \t \t \t \t"+str(sum(dist_set_occupancy.values())))

        # plot the distribution
        if(do_plot == True):            
            sorted_dist = {k: dist_set_occupancy[k] for k in sorted(dist_set_occupancy)}
            keys = sorted_dist.keys()
            values = sorted_dist.values()
            # print distribution to file.
            filename = "hist_valid_tags.txt"
            with open(filename, "w") as file:
                file.write(json.dumps(list(keys)) + "\n")
                file.write(json.dumps(list(values)))
            
            plt.bar(keys, values)
            plt.yscale('log')
            plt.xlabel('Number of Valid Tags Per Set')
            plt.ylabel('Count of Sets')
            plt.title('Set Occupancy')
            plt.xticks(range(15))
            plt.savefig("hist_valid_tags.pdf")
            plt.cla()
            plt.clf()
        return None

    def count_valid_tags(self):
        num_valid_tags = 0        
        for key,blocks in self.items():
            for block in blocks:
                if block['valid']:
                    num_valid_tags += 1
        return (num_valid_tags)

    def getDataStoreEntry(self, num_data_blocks, valid_flag, encoded_position, data_status):
        if (valid_flag == 1):
            for i in range(num_data_blocks):
                dsindex = random.randint(0, num_data_blocks - 1)            # randomly select an index in data store
                if (self.data_store[dsindex] == -1):                        # check if it contains any entry
                    self.data_store[dsindex] = [encoded_position, data_status]          # if not, then assign it to the address
                    return dsindex                                          # return the data store index for FPTR to tag store
                else:
                    continue
        if (valid_flag == 0):
            return None

    def invalidate_using_rptr(self, eviction_index, num_index_bits):
        if(self.data_store[eviction_index] == -1):                              # boundary case when the data store entry to be removed does not have any RPTR
            return
        rptr_entry = self.data_store[eviction_index][0][0:(num_index_bits+1)]   # RPTR = cache set where the invalidated tag resides in tag store
        rptr_status = self.data_store[eviction_index][1]                        # status of the removed entry. whether a valid or invalid entry has been removed
        rptr_way = int(self.data_store[eviction_index][0][(num_index_bits+1):]) # way number inside the selected cache set of the invalidated tag 
        self[rptr_entry][rptr_way] = {'valid': 0, 'fptr': None}                 # invalidate the tag by making it invalid and FPTR = null


    def do_random_GLE(self, new_tag_index, new_way_index, num_index_bits):
        eviction_index = random.randint(0, len(self.data_store) - 1)            # select random index for removal from data store 
        self.invalidate_using_rptr(eviction_index, num_index_bits)              # before removal, invalidate the entry in tag store
        if (self.data_store[eviction_index] == -1):                             # boundary case when the data store entry to be removed does not have any RPTR
            self.data_store[eviction_index] = [str(new_tag_index) + str(new_way_index), 'valid']    # RPTR =  new tag entry
        else:
            self.data_store[eviction_index][0] = str(new_tag_index) + str(new_way_index)            # RPTR =  new tag entry
            self.data_store[eviction_index][1] = 'valid'
        return eviction_index                                                   # return the data store index for FPTR entry in tag store

                    
    def mark_ref_as_last_seen(self, ref):
        addr_id = (ref.index, ref.tag)
        if addr_id in self.recently_used_addrs:
            self.recently_used_addrs.remove(addr_id)
        self.recently_used_addrs.append(addr_id)
        
        
    def load_balancing(self, index_tuple, num_partitions):
        block1 = self[str(0)+str(index_tuple[0])]
        block2 = self[str(1)+str(index_tuple[1])]
        block1_valid_count = 0
        block2_valid_count = 0
        # count number of valid tags in skew 0
        for block in block1:
            if block['valid']:
                block1_valid_count += 1
        # count number of valid tags in skew 1
        for block in block2:
            if block['valid']:
                block2_valid_count += 1
        # if valid entries in the skew is not balanced, choose the smaller one
        if block1_valid_count < block2_valid_count:
            return (0, block1_valid_count)
        elif block1_valid_count > block2_valid_count:
            return (1, block2_valid_count)
        else:           # randomly choose any skew
            randint = random.randint(0,1)
            if (randint == 0):
                return (0, block1_valid_count)
            else:
                return (1, block2_valid_count)
            
        
    def is_hit(self, addr_index, addr_tag, num_partitions):
        global partition
        global cal_index
        num_index_bits = int(len(addr_index[0]))
        blocks = []
        if addr_index[0] is None:
            blocks = self[str(0).zfill(num_index_bits)]
        else:
            for i in range(num_partitions):
                actual_index = str(i)+str(addr_index[i])
                empty_set = True 
                if (actual_index) in self:
                    blocks = self[actual_index]
                    for block in blocks:    # enumerate through all the ways to find if any tag is present
                        if 'tag' in block.keys():
                            empty_set = False
                            break
                    if empty_set == True:
                        continue
                    else:
                        for block in blocks:
                            if ('tag' in block.keys() and block['tag'] == addr_tag):    # if the tag matches, then return true; else false in all cases
                                partition = i
                                cal_index = addr_index[i]
                                return True
                else:
                    return False    
        return False                    
                    
                
    def replace_block(self, blocks, replacement_policy, num_tags_per_set, skew, valid_count, num_partition, addr_index, new_entry, count_ref_index, num_index_bits):
        if (replacement_policy == 'rand'):
            if (valid_count < num_tags_per_set):    # check if invalid blocks are present in the selected cache set of the skew
                repl_block_index = -1               # initialize the variable
                for (index, block) in enumerate(blocks):    # enumerate through all ways in the cache set
                    if block['valid'] == 0:         # if an invalid block is found
                        repl_block_index = index    
                        tag_index = str(skew) + str(addr_index)
                        new_entry['fptr'] = self.do_random_GLE(tag_index, index, num_index_bits)    #FPTR = index of entry of data store removed via GLE
                        break

                blocks[repl_block_index] = new_entry    # replace the block. This is a GLE
                return
            else:
                print("valid eviction")
                writeFile.write_eviction_status()
                repl_block_index = random.randint(0, num_tags_per_set - 1)      # since no invalid tags are left, select a way randomly
                for (index, block) in enumerate(blocks):    # find the selected way
                    if (index == repl_block_index):
                        new_entry['fptr'] = block['fptr']   # keep FPTR = previous FPTR
                        blocks[index] = new_entry           # replace the block. This is an SAE
                        return
                
    def set_block(self, replacement_policy, num_tags_per_set, num_partition, addr_index, new_entry, count_ref_index, num_index_bits):
        global partition
        global cal_index
        num_index_bits = int(len(addr_index[0]))
        if addr_index[0] is None:
            blocks = self[str(0).zfill(num_index_bits)]
        else:
            skew, count_valid = self.load_balancing(index_tuple = addr_index, num_partitions = num_partition)      # perform load balancing between the two skews      
            blocks = self[str(skew)+str(addr_index[skew])]      
            self.replace_block(blocks, replacement_policy, num_tags_per_set, skew, count_valid, num_partition, addr_index[skew], new_entry, count_ref_index, num_index_bits)

        partition = skew    # assign the selected skew
        cal_index = addr_index[skew]    # assign the cache index in the selected skew
            
    def read_refs(self, num_total_tags_per_set, num_partitions, replacement_policy, num_words_per_block, num_index_bits, refs):
        count_ref_index = 0
        for ref in refs:  # for every address, check whether it's a cache hit or miss
            if self.is_hit(ref.index, ref.tag, num_partitions):     # check if it's a hit
                ref.cache_status = ReferenceCacheStatus.hit
                ref.partition = partition
                ref.index = cal_index
                ref.valid = 1
            else:       # on cache miss, set up the block and replace an existing one
                ref.cache_status = ReferenceCacheStatus.miss
                self.set_block(
                        replacement_policy = replacement_policy,
                        num_tags_per_set = num_total_tags_per_set,
                        num_partition = num_partitions,
                        addr_index = ref.index,
                        new_entry = ref.get_cache_entry(num_words_per_block),
                        count_ref_index = count_ref_index,
                        num_index_bits = num_index_bits
                        )
                ref.partition = partition
                ref.index = cal_index
                ref.valid = 1
            count_ref_index += 1
