#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:44:28 2021

@author: anirban
// Gururaj Saileshwar:  Added the index derivation using AES cipher.
"""
import random
from present import Present

## GS: For AES
import secrets
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
key1 = secrets.token_bytes(16)
key2 = secrets.token_bytes(16)
cipher1 = Cipher(algorithms.AES(key1), modes.ECB(), backend=default_backend())
cipher2 = Cipher(algorithms.AES(key2), modes.ECB(), backend=default_backend())

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
        # GS: for AES (extend to 128 bits)
        plaintext = plaintext.zfill(128)
        plaintext_bytearray = int(plaintext, 2).to_bytes((len(plaintext) + 7) // 8, byteorder='big')

        # GS: for AES
        encryptor1 = cipher1.encryptor()
        ciphertext = encryptor1.update(plaintext_bytearray) + encryptor1.finalize()
        ciphertext = ciphertext.hex()        
        ciphertext = str(bin(int(ciphertext, 16))[2:].zfill(64))
        start = len(ciphertext) - num_offset_bits - num_index_bits
        end = len(ciphertext) - num_offset_bits
        index1 = ciphertext[start:end]
        
        # GS: for AES
        encryptor2 = cipher2.encryptor()
        ciphertext = encryptor2.update(plaintext_bytearray) + encryptor2.finalize()
        ciphertext = ciphertext.hex()       
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
