#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:44:28 2021

@author: anirban
"""
import random
from present import Present

class BinaryAddress(str):
    def __new__(cls, bin_addr=None, word_addr=None, num_addr_bits=0):
        if word_addr is not None:
            return super().__new__(cls, bin(word_addr)[2:].zfill(num_addr_bits))
        else:
            return super().__new__(cls, bin_addr)
        
        
    @classmethod
    def prettify(cls, bin_addr, min_bits_per_group):
        mid = len(bin_addr) // 2
        if mid < min_bits_per_group:
            return bin_addr
        else:
            left = cls.prettify(bin_addr[:mid], min_bits_per_group)
            right = cls.prettify(bin_addr[mid:], min_bits_per_group)
            return ' '.join((left, right))
        
        
    def get_tag(self, num_tag_bits):
        end = num_tag_bits
        tag = self[:end]
        if (len(tag) != 0):
            return tag
        else:
            return None
        
    def get_partition(self, num_partitions, ways_per_partition):
        partition = (0, 1)
        
        return partition        
        
        
    def get_index(self, num_offset_bits, num_index_bits, num_partitions):
        index1 = None; index2 = None; 
        plaintext = bin(int(self[:-(num_offset_bits)], 2))[2:].zfill(64)
        key = bin(int('00000000000000000000', 16))[2:].zfill(80)
        cipher = Present(key)
        ciphertext = cipher.encrypt(plaintext)
        ciphertext = str(bin(int(ciphertext, 16))[2:].zfill(64))
                
        start = len(ciphertext) - num_offset_bits - num_index_bits
        end = len(ciphertext) - num_offset_bits
        index1 = ciphertext[start:end]
        
        key = bin(int('00000000000000001111', 16))[2:].zfill(80)
        cipher = Present(key)
        ciphertext = cipher.encrypt(plaintext)
        ciphertext = str(bin(int(ciphertext, 16))[2:].zfill(64))
        start = len(ciphertext) - num_offset_bits - num_index_bits
        end = len(ciphertext) - num_offset_bits
        index2 = ciphertext[start:end]
        
        return (index1, index2)

    def get_offset(self, num_offset_bits):
        start = len(self) - num_offset_bits
        offset = self[start:]
        if (len(offset) != 0):
            return offset
        else:
            return None