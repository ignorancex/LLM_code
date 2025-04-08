#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 19:57:06 2021

@author: anirban
"""


class WordAddress(int):
    def get_consecutive_words(self, num_words_per_block):
        offset = self % num_words_per_block
        return [(self - offset + i) for i in range(num_words_per_block)]