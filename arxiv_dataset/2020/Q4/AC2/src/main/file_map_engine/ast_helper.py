new_data = {}

def wrapper_get(tree, test_path):
    new_data.clear()
    module = tree['Module']
    objects = module['body']
    get_call_tree(objects, test_path)
    return new_data
    
def get_call_tree(objects, par):
    for i in objects:
        keys = list(i.keys())
        type_ = keys[0]
        if keys[0] =='ClassDef':
            from_3 = i['ClassDef']['name']
            if from_3 in new_data.keys():
                new_data[par].append((from_3, 'ClassDef'))
            else:
                new_data[par] = [(from_3, 'ClassDef')]

        if keys[0] == 'FunctionDef':
            # print(i['FunctionDef']['name'], "defined in", par)
            retrieve_inner_functions(i, par)
        elif keys[0] == 'Expr':
#             print(i['Expr']['value']['Call']['func']['Name']['ctx'])
            retrieve_inner_expr(i, par)
        elif keys[0] == 'Assign':
#             print(i['Expr']['value']['Call']['func']['Name']['ctx'])
            retrieve_inner_assign(i, par)
        elif keys[0] == 'With':
#             print(i['Expr']['value']['Call']['func']['Name']['ctx'])
            retrieve_inner_with(i, par)
        elif keys[0] == 'ImportFrom':
#             print(i['Expr']['value']['Call']['func']['Name']['ctx'])
            retrieve_inner_imp_fro(i, par)
        elif keys[0] == 'Import':
#             print(i['Expr']['value']['Call']['func']['Name']['ctx'])
            retrieve_inner_imp(i, par)
        else:
            if 'body' in list(i[keys[0]].keys()):
                get_call_tree(i[keys[0]]['body'], par)
            
def retrieve_inner_imp_fro(obj, from_):
    
        
    from_2 = obj['ImportFrom']['module']
    
    if from_ in new_data.keys():
        new_data[from_].append((from_2, 'module'))
    else:
        new_data[from_] = [(from_2, 'module')]
        
    names = obj['ImportFrom']['names']
    for name in names:
        name_true = name['alias']['name']
        # print(name_true,"from", from_2,"into", from_)
        if from_2 in new_data.keys():
            new_data[from_2].append((name_true, 'func'))
        else:
            new_data[from_2] = [(name_true, 'func')]
    

            
def retrieve_inner_imp(obj, from_):
    names = obj['Import']['names']
    for name in names:
        name_true = name['alias']['name']
        # print(name_true,"into" , from_)
        if from_ in new_data.keys():
            new_data[from_].append((name_true, 'module'))
        else:
            new_data[from_] = [(name_true, 'module')]
            

def retrieve_inner_functions(obj, from_):
    from_func = obj['FunctionDef']['name']
    body_func = obj['FunctionDef']['body']

    print(from_func)

    if from_ in new_data.keys():
        new_data[from_].append((from_func, 'funcDef'))
    else:
        new_data[from_] = [(from_func, 'funcDef')]


    get_call_tree(body_func, from_func)


def retrieve_inner_expr(obj, from_):
    keys = list(obj.keys())
    if list(obj[keys[0]].keys())[0] == "value":
        new_obj = obj[keys[0]]['value']
        if list(new_obj.keys())[0] == "Call":
            new_obj = new_obj['Call']
            if list(new_obj.keys())[0] == "func":
                new_obj = new_obj['func']
                
                
                
                if list(new_obj.keys())[0] == "Attribute":
                    if 'Name' in list(new_obj['Attribute']['value'].keys()):
                        from_2 = new_obj['Attribute']['value']['Name']['id']
                        func_name_ = obj[keys[0]]['value']['Call']['func']['Attribute']['attr']
                        # print(func_name_, "from", from_2,"called in", from_)

#                         if from_ in new_data.keys():
#                             if (from_2, 'module') not in new_data[from_]:
#                                 new_data[from_].append((from_2, 'module'))
#                         else:
#                             new_data[from_] = [(from_2, 'module')]

                        if from_2 in new_data.keys():
                            if (func_name_, 'func') not in new_data[from_2]:
                                new_data[from_2].append((func_name_, 'func'))
                        else:
                            new_data[from_2] = [(func_name_, 'func')]
                    elif 'Attribute' in list(new_obj['Attribute']['value'].keys()):
                        pass
                        # print("\n\n",new_obj['Attribute']['value']['Attribute'].keys(), "\n\n")
                    
                    
                    
                elif list(new_obj.keys())[0] == "Name":
                    if from_ in new_data.keys():
                        if (new_obj['Name']['id'], 'func') not in new_data[from_]:
                            new_data[from_].append((new_obj['Name']['id'], 'func'))
                    else:
                        new_data[from_] = [(new_obj['Name']['id'], 'func')]
                    # print(new_obj['Name']['id'], "called from", from_)
    
    
    
def retrieve_inner_assign(obj, from_):
    keys = list(obj.keys())    
    keys2 = list(obj[keys[0]]['value'].keys())
    new_obj = obj[keys[0]]['value']
    key_ = keys2[0]
    
    if key_ == 'Call':
        new_obj = new_obj['Call']
        if list(new_obj.keys())[0] == "func":
            new_obj = new_obj['func']
            
            
#             print(new_obj.keys())
            
            
            if list(new_obj.keys())[0] == "Attribute":
                if 'Name' in list(new_obj['Attribute']['value'].keys()):
                    from_2 = new_obj['Attribute']['value']['Name']['id']
                    func_name_ = obj[keys[0]]['value']['Call']['func']['Attribute']['attr']
                    # print(func_name_, "from", from_2,"called in", from_)

#                         if from_ in new_data.keys():
#                             if (from_2, 'module') not in new_data[from_]:
#                                 new_data[from_].append((from_2, 'module'))
#                         else:
#                             new_data[from_] = [(from_2, 'module')]

                    if from_2 in new_data.keys():
                        if (func_name_, 'func') not in new_data[from_2]:
                            new_data[from_2].append((func_name_, 'func'))
                    else:
                        new_data[from_2] = [(func_name_, 'func')]
                elif 'Attribute' in list(new_obj['Attribute']['value'].keys()):
                    pass
                    # print("\n\n",new_obj['Attribute']['value']['Attribute'].keys(), "\n\n")

            elif list(new_obj.keys())[0] == "Name":
                pass
                # print(new_obj['Name']['id'], "called from", from_)
                
                if from_ in new_data.keys():
                    if (new_obj['Name']['id'], 'func') not in new_data[from_]:
                        new_data[from_].append((new_obj['Name']['id'], 'func'))
                else:
                    new_data[from_] = [(new_obj['Name']['id'], 'func')]

    


def retrieve_inner_with(obj, from_):
    keys = list(obj.keys())
#     print("\n\n",obj[keys[0]].keys(),"\n\n")
#     print("\n\n")
#     print(obj[keys[0]]['items'])
    # print("\n\n")
    # print(obj[keys[0]]['items'][0]['withitem'].keys())
    # print("\n\n")

    
    
    try:
        if obj[keys[0]]['items'][0]['withitem']['optional_vars']:
            file_name = obj[keys[0]]['items'][0]['withitem']['optional_vars']['Name']['id']
        else:
            file_name = "None"
    except:
        file_name = "file"
        # print("\n\n")
    
#     print(obj[keys[0]]['items'][0]['withitem']['context_expr']['Call']['args'][0]['Constant']['value'])
    # print("\n\n")
    
    if from_ in new_data.keys():
        if (file_name, 'file') not in new_data[from_]:
            new_data[from_].append((file_name, 'file'))
    else:
        new_data[from_] = [(file_name, 'file')]
    
    get_call_tree(obj[keys[0]]['body'], from_)
    

        
    
