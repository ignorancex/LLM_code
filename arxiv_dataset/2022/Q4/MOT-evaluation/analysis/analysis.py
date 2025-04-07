
import os

import numpy as np
import pandas as pd


def search_correlation(tb, conditions, type_c):

    if (len(type_c) == 1) or (len(type_c) - 1 == len(conditions)):

        accept = []

        for seq_id in tb['Set id'].unique():

            seq_info = tb.loc[tb['Set id'] == seq_id]

            accept += search_seq(seq_info, conditions, type_c)


        # print(accept)
        return accept


    return None



def search_seq(tb, conditions, type_c):

    accept = []
    one_c = False

    if len(type_c) == 1:
        one_c = True
        cond_type = type_c[0]


    # print(len(tb))


    # Compare without repeating. 
    # for i, A in tb.iterrows():
    for i, (_, A) in enumerate(tb.iterrows()):
        # print('--------------------------')

        # for j, B in tb[i+1:].iterrows():
        for j, (_, B) in enumerate(tb[i+1:].iterrows()):

            # print(i, j)

            c1 = None

            # For conditions
            for idx, (k, e) in enumerate(conditions.items()):

                if not one_c: cond_type = type_c[idx]

                c2 = (abs(A[k] - B[k]) <= e)

                if idx == 0: c1 = c2

                c1 = apply_cond(c1, c2, cond_type)


            if c1:   accept.append([A, B])


    return accept




def apply_cond(e1, e2, type_c):

    if type_c == 'or': return e1 or e2
    if type_c == 'and': return e1 and e2

    return None





def load_histogram(file, path='../outputs/evaluation'):

    df = pd.read_csv(os.path.join(path, file))

    data = df.values.flatten()
    data = data[np.logical_not(np.isnan(data))]

    return data




def load_csv_metric():

    files_name_list = ['intra.csv', 'Qd.csv', 'Qt.csv', 'Id.csv', 'Nd.csv', 'It.csv', 'Nt.csv', 'inter.csv', 'Y.csv', 'C.csv', 'IDSW.csv']
    dict_data = {}

    for file in files_name_list:

        name = file.split('.')[0]

        print(name)

        dict_data[name] = load_histogram(file)


    return dict_data



