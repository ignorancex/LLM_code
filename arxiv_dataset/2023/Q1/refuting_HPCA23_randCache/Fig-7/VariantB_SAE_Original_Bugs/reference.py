#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:12:52 2021

@author: anirban
"""


from collections import OrderedDict
from enum import Enum

from bin_addr import BinaryAddress
from word_addr import WordAddress


class Reference(object):
    
    def __init__(self, word_addr, num_addr_bits, num_offset_bits, num_index_bits, num_tag_bits, num_partitions, ways_per_partition):
        self.word_addr = WordAddress(word_addr)
        self.bin_addr = BinaryAddress(word_addr = self.word_addr, num_addr_bits = num_addr_bits)
        self.offset = self.bin_addr.get_offset(num_offset_bits)
        self.skew = self.bin_addr.get_partition(num_partitions, ways_per_partition)
        self.index = self.bin_addr.get_index(num_offset_bits, num_index_bits, num_partitions)
        self.tag = self.bin_addr.get_tag(num_tag_bits)
        self.valid = 0

        
        
        self.cache_status = None
        
        
    def __str__(self):
        return str(OrderedDict(sorted(self.__dict__.items())))
    
    __repr__ = __str__
    
    def get_cache_entry(self, num_words_per_block):
        return {
                'valid': 1,
                'tag': self.tag,
                'data': self.word_addr.get_consecutive_words(num_words_per_block)
                }
        
class ReferenceCacheStatus(Enum):
    
    miss = 0
    hit = 1
    
    def __str__(self):
        if self.value == ReferenceCacheStatus.hit.value:
            return 'HIT'
        else:
            return 'miss'
        
    __repr__ = __str__
    
            
class ReferenceEvictionStatus(Enum):
    valid = 1
    invalid = 0
    
    def __str__(self):
        if self.value == ReferenceEvictionStatus.valid.value:
            return 'SAE'
        else:
            return 'global'
    
    __repr__ = __str__
    