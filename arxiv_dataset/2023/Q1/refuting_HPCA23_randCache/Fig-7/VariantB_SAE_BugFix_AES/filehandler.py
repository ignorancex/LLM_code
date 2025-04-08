#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 23:58:45 2021

@author: anirban
"""


class writeFile(object):
    def write_address(plaintext, ciphertext):
        with open('address_list.txt', 'a') as filehandle:
            filehandle.writelines(str(plaintext))
            filehandle.writelines("\t")
            filehandle.writelines(str(ciphertext))
            filehandle.writelines("\n")
            
    def write_cache_details(self, word_addr, partition, index, status):
        with open('cache_details.txt','a') as filehandle:
            filehandle.writelines(str(word_addr))
            filehandle.writelines("\t")
            filehandle.writelines(str(partition))
            filehandle.writelines("\t")
            filehandle.writelines(str(index))
            filehandle.writelines("\t")
            filehandle.writelines(str(status))
            filehandle.writelines("\n")

    def write_eviction_status():
        with open('eviction_status.txt','a') as filehandle:
            filehandle.writelines("valid eviction")
            filehandle.writelines("\n")

        