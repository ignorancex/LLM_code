import os
import sys
import re
import ast
from pprint import pprint

def parse_python_func(test_path):
    print("File detected: ", test_path)
    _file = open(test_path).readlines()
    # print(len(_file))

    prev_indent = -1
    cur_indent = 0

    parent_dict = {}
    parent = {}
    imp_dict = {}

    parent_calls = {}

    parent[-1] = 'this_file'
    
    for sent in _file:
        
        i = sent.strip()
        if i.find('from') != -1:
            i = i.replace('from', '')
            i = i.replace(' ', '')
            i = i.split('import')
            fr = i[0]
            to = i[-1].split(',')
            if fr not in imp_dict.keys():
                imp_dict[fr] = []
            for j in to:
                imp_dict[fr].append(j)
        else:
            if i.find('import') != -1:
                i = i.replace(' ','')
                i = i.replace('import','')
                i = i.split(',')
                if 'module' not in imp_dict.keys():
                    imp_dict['module'] = []
                for k in i:
                    imp_dict['module'].append(k) 

        
        
        tokens = sent.strip().split(' ')
        # Only consider lines which have actual content
        sent_str = sent.strip()
        if len(sent_str) > 0:
            indent = len(sent) - len(sent.lstrip())
            indent_lev = get_indent_level(indent)

            # print(get_indent_level(indent))
            
            #Chech if the line is a function definition
            cur_indent = indent_lev

            status = 0
            
            if prev_indent - cur_indent == 0:
                # print('same level')
                status = 0
            elif prev_indent - cur_indent < 0:
                # print('increase in level')
                status = 1
            else:
                # print('decrease in level')
                status = -1

            prev_indent = cur_indent

            func_name = re.search("[a-zA-Z]+[a-zA-Z0-9_]*\(", sent_str)
            
            

            if func_name:
                act_func = func_name.group()[:-1]
                print(act_func)

                # Is a function defintion
                if tokens[0] == 'def' or tokens[0] == 'with':
                    func_name = re.search("[a-zA-Z]+[a-zA-Z0-9_]*\(", sent_str)
                    # print("function definition: ", func_name.group()[:-1])
                    
                    print("\n\nParent", parent)
                    try:
                        if parent[indent_lev-1] not in parent_dict.keys():
                            parent_dict[parent[indent_lev-1]] = [act_func]
                        else:
                            parent_dict[parent[indent_lev-1]].append(act_func)
                        parent[indent_lev] = act_func
                        # parent_dict[act_func] = parent[indent_lev-1]
                    except:
                        if parent[indent_lev-2] not in parent_dict.keys():
                            parent_dict[parent[indent_lev-2]] = [act_func]
                        else:
                            parent_dict[parent[indent_lev-2]].append(act_func)
                        # parent_dict[act_func] = parent[indent_lev-1]
                        parent[indent_lev-1] = act_func
                    
                
                else:
                    # print(indent_lev, ' function name: ', func_name.group()[:-1])

                    try:
                        if parent[indent_lev-1] not in parent_calls.keys():
                            parent_calls[parent[indent_lev-1]] = [act_func]
                        else:
                            parent_calls[parent[indent_lev-1]].append(act_func)
                    
                    except:
                        if parent[indent_lev-2] not in parent_calls.keys():
                            parent_calls[parent[indent_lev-2]] = [act_func]
                        else:
                            parent_calls[parent[indent_lev-2]].append(act_func)


            else: # Is a Condition
                tok = ['if', 'elif', 'else:', 'for', 'while','switch', 'with', 'open']
                print(tokens)
                if tokens[0] in tok:
                    

                    
                    act_func = tokens[0]+'_<spc_tok>_'+str(parent[indent_lev-1])

                    # print("Condition: ", act_func)
                    try:
                        if parent[indent_lev-1] not in parent_dict.keys():
                            parent_dict[parent[indent_lev-1]] = [act_func]
                        else:
                            parent_dict[parent[indent_lev-1]].append(act_func)
                        # parent_dict[act_func] = parent[indent_lev-1]
                        parent[indent_lev] = act_func
                    except:
                        if parent[indent_lev-2] not in parent_dict.keys():
                            parent_dict[parent[indent_lev-2]] = [act_func]
                        else:
                            parent_dict[parent[indent_lev-2]].append(act_func)
                        # parent_dict[act_func] = parent[indent_lev-1]
                        parent[indent_lev-1] = act_func


            # if sent_str.find('__main__')!=-1:
                # print('function name: ', 'main')


    # print("\n\n")
    # print(parent_dict)
    # print(parent_calls)
    
    # print("\n\n")

    return (parent_dict, parent_calls, imp_dict)

def get_indent_level(cur):
    return int(cur/4)


def find_key(pard, check):
    for key in pard.keys():
        temp_list = pard[key]
        if check in temp_list:
            if key.find('_<spc_tok>_')!=-1:
                g = find_key(pard, key)
                return g
            return key

def get_calls(test_path):

    pard, parc, imp_dict = parse_python_func(test_path)

    new_call_dict = {}
    for i in parc.keys():
        if i.find('_<spc_tok>_')!=-1:
            temp = find_key(pard, i)
            if temp in new_call_dict.keys():
                new_call_dict[temp].extend(parc[i])
            else:
                new_call_dict[temp] = parc[i]
        else:
            new_call_dict[i] = parc[i]

    # print(new_call_dict)
    return new_call_dict, imp_dict