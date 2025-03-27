# -*- coding: utf-8 -*-

with open("./Multinews_testX.txt") as f:
    for line in f:
        doc_set = line
        docsN = doc_set.count('story_separator_special_tag') + 1
        print(docsN)
