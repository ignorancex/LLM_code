import itertools
import json
import argparse
import os
import copy
import random
from random_words import RandomWords

dataset_names=['posLen_tp_1_rand_c','posLen_tp_2_rand_c','posLen_tp_3_rand_c','posLen_tp_4_rand_c'
              ,'posLen_tp_5_rand_c']

data_detail_names=['position_length_type_1','position_length_type_2','position_length_type_3'
                   ,'position_length_type_4','position_length_type_5']

for i in range(len(data_detail_names)):
    dataset_name = dataset_names[i]
    data_detail_name = data_detail_names[i]

    if not os.path.isdir(dataset_name + '_human_test'):
        os.mkdir(dataset_name + '_human_test')
    if not os.path.isdir(dataset_name + '_human_test/' + 'detail' ):
        os.mkdir(dataset_name + '_human_test/' + 'detail')

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
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    new_dict['train']=False
    json.dump(new_dict,fp,indent=4)
    fp.close()

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()


    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['TitlePosition']='left'
    new_dict['TitlePaddingLeft']=float(0.05)
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['lineThickness']=2
    if 'posLen_tp_1_rand_c' in dataset_name or 'posLen_tp_7_rand_c' in 'posLen_tp_13_rand_c' in dataset_name or dataset_name or 'posLen_tp_15_rand_c' in dataset_name:
        
        new_dict['fixBarGap']=2
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    if 'posLen_tp_1_rand_c' in dataset_name or 'posLen_tp_7_rand_c' in dataset_name or 'posLen_tp_13_rand_c' in dataset_name or 'posLen_tp_15_rand_c' in dataset_name:
        new_dict['barWidth']=5
        pass
    else:
        new_dict['stackWidth']=27
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['bgcolorL_perturbation']=-25
    new_dict['LABperturbation']=True
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['strokecolorL_perturbation']=25
    new_dict['strokeLABperturbation']=True
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['barcolorL_perturbation']=-15
    new_dict['barLABperturbation']=True
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()
    
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['values']['valueRange']=[1,10,93,100]
    # new_dict['changeTargetOnly']=True
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()
    
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    if 'posLen_tp_13_rand_c' in dataset_name or 'posLen_tp_15_rand_c' in dataset_name:
        new_dict['mark']['bottomValue']=2
        pass
    else:
        new_dict['mark']['dotDeviation']=3
    fp = open(dataset_name + '_human_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_human_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()