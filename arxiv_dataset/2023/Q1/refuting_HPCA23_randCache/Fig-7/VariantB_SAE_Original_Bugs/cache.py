#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:53:27 2021

@author: anirban
"""
from bin_addr import BinaryAddress
from word_addr import WordAddress
from reference import ReferenceCacheStatus
from reference import ReferenceEvictionStatus
from filehandler import writeFile

import random

class Cache(dict):
    partition = None
    cal_index = None
    count_ref_index = 0
    def __init__(self, data_store = None, tag_store = None, num_data_blocks = None, num_sets_per_skew = None, num_index_bits = None, num_partitions = None, num_tag_blocks_per_skew = None, num_addr_bits = None, num_offset_bits = None, num_total_ways = None):
       
        self.recently_used_addrs = []
        if tag_store is not None:
            self.update(tag_store)
        else:
            for j in range(num_partitions):
                for i in range(num_tag_blocks_per_skew):                
                    index = BinaryAddress(word_addr = WordAddress(i), num_addr_bits = num_addr_bits)[(num_addr_bits - num_offset_bits - num_index_bits):(num_addr_bits - num_offset_bits)]
                    self[str(j)+str(index)] = []
                    for k in range(num_total_ways):
                        if (random.randint(0, 1)  == 0) :
                            self[str(j)+str(index)].append({'valid': 0}) 
                        else:
                            self[str(j)+str(index)].append({'valid': 1})
                    
                    
    def mark_ref_as_last_seen(self, ref):
        addr_id = (ref.index, ref.tag)
        if addr_id in self.recently_used_addrs:
            self.recently_used_addrs.remove(addr_id)
        self.recently_used_addrs.append(addr_id)

    ## GS: For assert.
    def count_valid_tags(self):
        num_valid_tags = 0        
        for key,blocks in self.items():
            for block in blocks:
                if block['valid']:
                    num_valid_tags += 1
        return (num_valid_tags)

    def load_balancing(self, index_tuple, num_partitions):
        block1 = self[str(0)+str(index_tuple[0])]
        block2 = self[str(1)+str(index_tuple[1])]
        block1_valid_count = 0
        block2_valid_count = 0
        for block in block1:
            if block['valid']:
                block1_valid_count += 1
        for block in block2:
            if block['valid']:
                block2_valid_count += 1
        if block1_valid_count < block2_valid_count:
            return (0, block1_valid_count)
        elif block1_valid_count > block2_valid_count:
            return (1, block2_valid_count)
        else:
            
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
                    for block in blocks:
                        if 'tag' in block.keys():
                            empty_set = False
                            break
                    if empty_set == True:
                        continue
                    else:
                        for block in blocks:
                            if ('tag' in block.keys() and block['tag'] == addr_tag):
                                partition = i
                                cal_index = addr_index[i]
                                return True
                else:
                    return False    
        return False                    
                    
                
    def replace_block(self, blocks, replacement_policy, num_tags_per_set, skew, valid_count, num_partition, addr_index, new_entry, count_ref_index, num_data_blocks):
        if (replacement_policy == 'rand'):
            if (valid_count < num_tags_per_set):
                repl_block_index = -1
                for (index, block) in enumerate(blocks):
                    if block['valid'] == 0:
                        repl_block_index = index
                        break
                blocks[repl_block_index] = new_entry
                return
            else:
                print("valid eviction")
                writeFile.write_eviction_status()
                if (self.count_valid_tags() > num_data_blocks):
                    print ("\033[0;31mASSERT FAILURE:\033[0m assert(Valid Tags <= Cache Capacity). Valid Tags : " + str(self.count_valid_tags()) + ", Cache Capacity : " + str(num_data_blocks))
                
                repl_block_index = random.randint(0, num_tags_per_set - 1)
                for (index, block) in enumerate(blocks):
                    if (index == repl_block_index):
                        blocks[index] = new_entry
                        return
                
    def set_block(self, replacement_policy, num_tags_per_set, num_partition, addr_index, new_entry, count_ref_index, num_data_blocks):
        global partition
        global cal_index
        num_index_bits = int(len(addr_index[0]))
        if addr_index[0] is None:
            blocks = self[str(0).zfill(num_index_bits)]
        else:
            skew, count_valid = self.load_balancing(index_tuple = addr_index, num_partitions = num_partition)            
            blocks = self[str(skew)+str(addr_index[skew])]
            self.replace_block(blocks, replacement_policy, num_tags_per_set, skew, count_valid, num_partition, addr_index[skew], new_entry, count_ref_index, num_data_blocks)

        partition = skew
        cal_index = addr_index[skew]
            
            
    def read_refs(self, num_total_tags_per_set, num_partitions, replacement_policy, num_words_per_block, refs, num_data_blocks):
        count_ref_index = 0
        for ref in refs:
            if self.is_hit(ref.index, ref.tag, num_partitions):
                ref.cache_status = ReferenceCacheStatus.hit
                ref.partition = partition
                ref.index = cal_index
                ref.valid = 1
            else:
                ref.cache_status = ReferenceCacheStatus.miss
                self.set_block(
                        replacement_policy = replacement_policy,
                        num_tags_per_set = num_total_tags_per_set,
                        num_partition = num_partitions,
                        addr_index = ref.index,
                        new_entry = ref.get_cache_entry(num_words_per_block),
                        count_ref_index = count_ref_index,
                        num_data_blocks = num_data_blocks
                        )
                ref.partition = partition
                ref.index = cal_index
                ref.valid = 1
            count_ref_index += 1
