import os
from file_map_engine.engine import get_calls

def dir_walk(test):

    Tree = {}
    
    for root, dirs, files in os.walk(test, topdown=True):
        # print(root, dirs, files)
        if root not in Tree.keys():
            Tree[root] = []

        for name in dirs:
            Tree[root].append({'type':'dir','value':os.path.join(root,name)})
        
        for name in files:
            Tree[root].append({'type' : 'file', 'value': os.path.join(root ,name)})
        
        # print(Tree)
    
    return Tree


