import itertools
import json
import argparse
import os
import copy
import random
import numpy as np

dataset_names=['posLen_tp_1_rand_c','posLen_tp_2_rand_c','posLen_tp_3_rand_c','posLen_tp_4_rand_c'
              ,'posLen_tp_5_rand_c']

data_detail_names=['position_length_type_1','position_length_type_2','position_length_type_3'
                   ,'position_length_type_4','position_length_type_5']

for i in range(len(data_detail_names)):
    dataset_name = dataset_names[i]
    data_detail_name = data_detail_names[i]

    if not os.path.isdir(dataset_name + '_markPosition_test'):
        os.mkdir(dataset_name + '_markPosition_test')
    if not os.path.isdir(dataset_name + '_markPosition_test/' + 'detail' ):
        os.mkdir(dataset_name + '_markPosition_test/' + 'detail')

    if 'posAngle' in dataset_name:
        path = 'detail/posAngle/{}.json'.format(data_detail_name)
    else:
        path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
    with open(path,'r') as load_f:
        original_dict = json.load(load_f)
        load_f.close()
        # print(original_dict)
    path1 = '{}.json'.format(dataset_name)
    with open(path1,'r') as load_f:
        original_dict1 = json.load(load_f)
        load_f.close()
        # print(original_dict1)
    original_dict1['trainPairCount'] = 0
    original_dict1['testPairCount'] = 0
    original_dict1['validPairCount'] = 1000

    a=original_dict
    b=original_dict1

    i=0
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False

    fp = open(dataset_name + '_markPosition_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_markPosition_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

    i=0

    for j in np.arange(-3,4,1):
        i+=1
        new_dict = copy.deepcopy(a)
        new_dict1 = copy.deepcopy(b)
        new_dict['train']=False
        new_dict['mark']['dotDeviation']=int(j)
        # if j==2 or j==3:
        #     new_dict['fixBarGap']=2
        fp = open(dataset_name + '_markPosition_test/detail/testdata_detail_{}.json'.format(i),'w')

        json.dump(new_dict,fp,indent=4)
        fp.close()
        # print(new_dict)

        new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
        fp = open('{}_markPosition_test/testdata_{}.json'.format(dataset_name,i),'w')
        json.dump(new_dict1,fp,indent=4)
        fp.close()

# dataset_names=['posLen_tp_1_rand_c','posLen_tp_3_rand_c','posLen_tp_7_rand_c','posLen_tp_2_rand_c','posLen_tp_10_rand_c','posLen_tp_11_rand_c'
#               ,'posLen_tp_12_rand_c','posAngle_pie','posAngle_bar']

# data_detail_names=['position_length_type_1','position_length_type_3','position_length_type_7','position_length_type_2','position_length_type_10'
#                    ,'position_length_type_11','position_length_type_12','position_angle_pie','position_angle_bar']

# for i in range(len(data_detail_names)):
#     dataset_name = dataset_names[i]
#     data_detail_name = data_detail_names[i]

#     if not os.path.isdir(dataset_name + '_randpos_test'):
#         os.mkdir(dataset_name + '_randpos_test')
#     if not os.path.isdir(dataset_name + '_randpos_test/' + 'detail' ):
#         os.mkdir(dataset_name + '_randpos_test/' + 'detail')

#     if 'posAngle' in dataset_name:
#         path = 'detail/posAngle/{}.json'.format(data_detail_name)
#     else:
#         path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
#     with open(path,'r') as load_f:
#         original_dict = json.load(load_f)
#         load_f.close()
#         # print(original_dict)
#     path1 = '{}.json'.format(dataset_name)
#     with open(path1,'r') as load_f:
#         original_dict1 = json.load(load_f)
#         load_f.close()
#         # print(original_dict1)
#     original_dict1['trainPairCount'] = 0
#     original_dict1['testPairCount'] = 0
#     original_dict1['validPairCount'] = 1000

#     a=original_dict
#     b=original_dict1

#     i=0
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False

#     fp = open(dataset_name + '_randpos_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_randpos_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()

#     i=0

#     for j in np.arange(0,1):
#         i+=1
#         new_dict = copy.deepcopy(a)
#         new_dict1 = copy.deepcopy(b)
#         new_dict['train']=False
#         new_dict['mark']['randPos']=True
#         # if j==2 or j==3:
#         #     new_dict['fixBarGap']=2
#         fp = open(dataset_name + '_randpos_test/detail/testdata_detail_{}.json'.format(i),'w')

#         json.dump(new_dict,fp,indent=4)
#         fp.close()
#         # print(new_dict)

#         new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#         fp = open('{}_randpos_test/testdata_{}.json'.format(dataset_name,i),'w')
#         json.dump(new_dict1,fp,indent=4)
#         fp.close()