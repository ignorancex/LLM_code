import numpy as np
import time
import gc


#Recebo um array que tem objectos do tipo

'''
{
 columns: []
 rows: []
}
0 14 17 60
1 15 54 60
0 14 17 60
'''

#array = [{'columns': ['60'], 'indexes': [0, 1, 2]}, {'columns': ['0', '14', '17', '60'], 'indexes': [0, 2]}]

# Acrescentar filtragem antes do merge
# Numero de clusters encontrados por partição
def merge_patterns(clusters, nr_lines, min_sup): #patterns, clusters, nr_lines, min_sup, total_patterns):
    f_c = []
    pointer_i = 0
    # Percorrer os clusters
    while pointer_i < len(clusters):
        temp = []
        pointer_j = pointer_i + 1
        # Os padrões do cluster são os iniciais
        for p_i in clusters[pointer_i]:
            temp.append(p_i)
        # Verificar os padrões dos clusters seguintes
        while pointer_j < len(clusters):
            current_temp = temp.copy()
            for pattern in current_temp:
                for pattern_j in clusters[pointer_j]:
                    indexes_intersected = np.intersect1d(pattern_j["indexes"], pattern["indexes"], assume_unique=True) #pattern_j["indexes"].intersection(pattern["indexes"])
                    if len(indexes_intersected)/nr_lines < min_sup:
                        continue
                    else:
                        columns_intersected = pattern_j["columns"].union(pattern["columns"])
                        temp.append({'columns': columns_intersected, 'indexes': indexes_intersected})
            pointer_j += 1
        to_be_added = []
        for p_temp in temp:
            can_be_added = True
            # check if it's closed within temp
            for p_aux in temp:
                if p_temp["columns"].issubset(p_aux["columns"]) and len(p_aux["columns"]) != len(p_temp["columns"]):
                    if len(p_temp["indexes"]) == len(p_aux["indexes"]):
                        can_be_added = False
                        break
            if not can_be_added:
                continue
            # check if its closed within p_f_c
            for p_f_c in f_c:
                if p_temp["columns"].issubset(p_f_c["columns"]) and len(p_f_c["columns"]) != len(p_temp["columns"]):
                    if len(p_temp["indexes"]) == len(p_f_c["indexes"]):
                        can_be_added = False
                        break
            if can_be_added:
                to_be_added.append(p_temp)
        for val in to_be_added:
            f_c.append(val)
        pointer_i += 1
    end = time.time_ns()
    return f_c


def merge_patterns_a(clusters, nr_lines, min_sup, intersections_merge, variables_subsets): #patterns, clusters, nr_lines, min_sup, total_patterns):
    print("#########################################################################################")
    print("Merge analysis:")
    total_inter = 0
    total_vars = 0
    start = time.time_ns()
    f_c = []
    pointer_i = 0
    # Percorrer os clusters
    while pointer_i < len(clusters):
        #######
        nr_intersections = 0
        #######
        temp = []
        pointer_j = pointer_i + 1
        # Os padrões do cluster são os iniciais
        for p_i in clusters[pointer_i]:
            temp.append(p_i)
        # Verificar os padrões dos clusters seguintes
        while pointer_j < len(clusters):
            current_temp = temp.copy()
            for pattern in current_temp:
                for pattern_j in clusters[pointer_j]:
                    ###################
                    nr_intersections += 1
                    ###################
                    indexes_intersected = np.intersect1d(pattern_j["indexes"], pattern["indexes"], assume_unique=True) #pattern_j["indexes"].intersection(pattern["indexes"])
                    if len(indexes_intersected)/nr_lines < min_sup:
                        continue
                    else:
                        columns_intersected = pattern_j["columns"].union(pattern["columns"])
                        temp.append({'columns': columns_intersected, 'indexes': indexes_intersected})
            pointer_j += 1
        #######
        total_inter += nr_intersections
        print("Number of interserctions "+str(nr_intersections))
        nr_var = 0
        #######
        to_be_added = []
        for p_temp in temp:
            can_be_added = True
            # check if it's closed within temp
            for p_aux in temp:
                ###########
                nr_var +=1
                ###########
                if p_temp["columns"].issubset(p_aux["columns"]) and len(p_aux["columns"]) != len(p_temp["columns"]):
                    if len(p_temp["indexes"]) == len(p_aux["indexes"]):
                        can_be_added = False
                        break
            if not can_be_added:
                continue
            # check if its closed within p_f_c
            for p_f_c in f_c:
                ###########
                nr_var +=1
                ###########
                if p_temp["columns"].issubset(p_f_c["columns"]) and len(p_f_c["columns"]) != len(p_temp["columns"]):
                    if len(p_temp["indexes"]) == len(p_f_c["indexes"]):
                        can_be_added = False
                        break
            if can_be_added:
                to_be_added.append(p_temp)
        #########
        total_vars += nr_var
        print("Number of variables checked for subsets: "+str(nr_var))
        #########
        for val in to_be_added:
            f_c.append(val)
        pointer_i += 1
    end = time.time_ns()
    print("End of merge analysis")
    print("#########################################################################################")
    variables_subsets.append(total_vars)
    intersections_merge.append(total_inter)
    return f_c
'''
clusters = [[{'columns': frozenset(['0']), 'indexes': frozenset([0, 2])}],
                [{'columns': frozenset(['14']), 'indexes': frozenset([0, 2])}],
                [{'columns': frozenset(['17']), 'indexes': frozenset([0, 2])}],
                [{'columns': frozenset(['60']), 'indexes': frozenset([0, 1, 2])}],
                [{'columns': frozenset(['1']), 'indexes': frozenset([1])}],
                [{'columns': frozenset(['15']), 'indexes': frozenset([1])}],
                [{'columns': frozenset(['17']), 'indexes': frozenset([1])}]
                ]

nr_lines = 3
min_sup = 0.1

y = merge_patterns(clusters, nr_lines, min_sup)

x=0
'''
