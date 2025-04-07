import pickle

# in_tree_file = open('../ast_map.dat', 'rb')
# new_tree = pickle.load(in_tree_file)
# in_tree_file.close()

# in_dict_file = open('../directory_map.dat', 'rb')
# new_dict = pickle.load(in_dict_file)
# in_dict_file.close()

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2))))

def metric_c2c(in_tree):

    keys = list(in_tree.keys())
    values = {i:{k:{0:0,1:0,2:0} for k in range(len(keys))} for i in range(len(keys))}

    for i in range(len(keys)):
        for j in range(i,len(keys)):
            func = c2c(in_tree[keys[i]], in_tree[keys[j]], ['func', 'funcDef'])
            clas = c2c(in_tree[keys[i]], in_tree[keys[j]], ['ClassDef'])
            mode = c2c(in_tree[keys[i]], in_tree[keys[j]], ['module'])

            values[i][j][0] = round(func,2)
            values[j][i][0] = round(func,2)
            values[i][j][1] = round(clas, 2)
            values[j][i][1] = round(clas, 2)
            values[i][j][2] = round(mode, 2)
            values[j][i][2] = round(mode, 2)

    print(values)    
    return values



def c2c(c1, c2, str):
    ver_1 = []
    ver_2 = []
    for k in c1.keys():
        for l in c1[k]:
            for j in c1[k][l]:
                print(j)
                if(j[1] in str):
                    # print(j)
                    ver_1.append(j[0])
    
    for k in c2.keys():
        for l in c2[k]:
            for j in c2[k][l]:
                print(j)
                if(j[1] in str):
                    # print(j)
                    ver_2.append(j[0])

    

    add = Diff(ver_2, ver_1)
    rem = Diff(ver_1, ver_2)

    print(len(add))
    print(len(rem))
    print(len(ver_1))
    print(len(ver_2))
    print('--------------')

    return (1 - ((len(add) + len(rem))/(len(ver_1) + len(ver_2)))) * 100
                # print(j[1])


# metric_c2c(new_dict)

def metric_a2a(in_tree):

    keys = list(in_tree.keys())
    values = {i:{k:0 for k in range(len(keys))} for i in range(len(keys))}
    for i in range(len(keys)):
        for j in range(i,len(keys)):
            temp = a2a(in_tree[keys[i]], in_tree[keys[j]])
            # print(temp)
            values[i][j] = round(temp, 2)
            values[j][i] = round(temp, 2)

    print(values)
    return values

def a2a(a1, a2):
    ac1 = aco(a1)
    ac2 = aco(a2)
    mto1 = mto(a1, a2)

    return (1 - (mto1/(ac1 + ac2))) * 100

def aco(a1):
    l1 = list(a1.keys())
    added_folders = Diff(l1, [])
    add_com = 0
    
    for idir in l1:
        files = [i['value'] for i in a1[idir]]
        add_com += len(files)

    return add_com + len(added_folders)

def mto(a1, a2):
    l1 = list(a1.keys())
    l2 = list(a2.keys())

    removed_folders = Diff(l1, l2)
    added_folders = Diff(l2, l1)
    common_folders = [i for i in l1 if i in l2]

    rem_fol = len(removed_folders)
    add_fol = len(added_folders)

    add_comp = []
    rem_comp = []

    for idir in removed_folders:
        files = [i['value'] for i in a1[idir]]
        rem_comp += files
        # rem_com += len(files)

    for idir in added_folders:
        files = [i['value'] for i in a2[idir]]
        add_comp += files
        # add_com += len(files)

    for idir in common_folders:
        files1 = [i['value'] for i in a1[idir]]
        files2 = [i['value'] for i in a2[idir]]

        add_comp += files2
        rem_comp += files1

    # print("Removed folders: ", rem_fol)
    # print("Added folders: ", add_fol)

    # print("new_addc: ", len(Diff(add_comp, rem_comp)))
    # print("new_remc: ", len(Diff(rem_comp, add_comp)))

    return rem_fol + add_fol + len(Diff(add_comp, rem_comp)) + len(Diff(rem_comp, add_comp))
    

def Diff(li1, li2):
    return (list(list(set(li1)-set(li2))))

# metric_a2a(new_tree)


