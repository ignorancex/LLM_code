import itertools
import json
import argparse
import os
import copy
import random
import numpy as np
from random_words import RandomWords
import sys
sys.path.append("../../../util")
from color_pool import *

dataset_name = 'posLen_tp_1_rand_c_mask'
data_detail_name = 'position_length_type_1_mask'

# if not os.path.isdir(dataset_name + '_barWidth_test'):
#     os.mkdir(dataset_name + '_barWidth_test')
# if not os.path.isdir(dataset_name + '_barWidth_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_barWidth_test/' + 'detail')

# path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
# with open(path,'r') as load_f:
#     original_dict = json.load(load_f)
#     load_f.close()
#     # print(original_dict)
# path1 = '{}.json'.format(dataset_name)
# with open(path1,'r') as load_f:
#     original_dict1 = json.load(load_f)
#     load_f.close()
#     # print(original_dict1)
# original_dict1['trainPairCount'] = 0
# original_dict1['testPairCount'] = 0
# original_dict1['validPairCount'] = 1000

# a=original_dict
# b=original_dict1

# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_barWidth_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_barWidth_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(5,9):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['barWidth']=int(j)
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_barWidth_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_barWidth_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()


if not os.path.isdir(dataset_name + '_strokeWidth_test'):
    os.mkdir(dataset_name + '_strokeWidth_test')
if not os.path.isdir(dataset_name + '_strokeWidth_test/' + 'detail' ):
    os.mkdir(dataset_name + '_strokeWidth_test/' + 'detail')

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
fp = open(dataset_name + '_strokeWidth_test/detail/testdata_detail_{}.json'.format(i),'w')
json.dump(new_dict,fp,indent=4)
fp.close()

new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
fp = open('{}_strokeWidth_test/testdata_{}.json'.format(dataset_name,i),'w')
json.dump(new_dict1,fp,indent=4)
fp.close()

i=0

for j in np.arange(0,4):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['lineThickness']=int(j)
    if j==2 or j==3:
        new_dict['fixBarGap']=2
    fp = open(dataset_name + '_strokeWidth_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_strokeWidth_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

# if not os.path.isdir(dataset_name + '_color_test'):
#     os.mkdir(dataset_name + '_color_test')
# if not os.path.isdir(dataset_name + '_color_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_color_test/' + 'detail')

# path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
# with open(path,'r') as load_f:
#     original_dict = json.load(load_f)
#     load_f.close()
#     # print(original_dict)
# path1 = '{}.json'.format(dataset_name)
# with open(path1,'r') as load_f:
#     original_dict1 = json.load(load_f)
#     load_f.close()
#     # print(original_dict1)
# original_dict1['trainPairCount'] = 0
# original_dict1['testPairCount'] = 0
# original_dict1['validPairCount'] = 1000

# a=original_dict
# b=original_dict1

# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_color_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_color_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(-15,20,5):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['bgcolor_pertubation']=int(j)
#     # new_dict['bgcolor']="pertubation"
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_color_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_color_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()

# for j in np.arange(-15,20,5):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['barcolor_pertubation']=int(j)
#     # new_dict['barcolor']="pertubation"
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_color_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_color_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()

# for j in np.arange(-15,20,5):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['strokecolor_pertubation']=int(j)
#     # new_dict['strokecolor']="pertubation"
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_color_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_color_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()


# if not os.path.isdir(dataset_name + '_title_test'):
#     os.mkdir(dataset_name + '_title_test')
# if not os.path.isdir(dataset_name + '_title_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_title_test/' + 'detail')

# path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
# with open(path,'r') as load_f:
#     original_dict = json.load(load_f)
#     load_f.close()
#     # print(original_dict)
# path1 = '{}.json'.format(dataset_name)
# with open(path1,'r') as load_f:
#     original_dict1 = json.load(load_f)
#     load_f.close()
#     # print(original_dict1)
# original_dict1['trainPairCount'] = 0
# original_dict1['testPairCount'] = 0
# original_dict1['validPairCount'] = 1000

# a=original_dict
# b=original_dict1

# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_title_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_title_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(0.2,0.9,0.1):
#     # for j in np.arange(0,1,0.1):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['TitlePosition']='left'
#     new_dict['TitlePaddingLeft']=j
#     fp = open(dataset_name + '_title_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_title_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()

# if not os.path.isdir(dataset_name + '_lightness_test'):
#     os.mkdir(dataset_name + '_lightness_test')
# if not os.path.isdir(dataset_name + '_lightness_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_lightness_test/' + 'detail')

# path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
# with open(path,'r') as load_f:
#     original_dict = json.load(load_f)
#     load_f.close()
#     # print(original_dict)
# path1 = '{}.json'.format(dataset_name)
# with open(path1,'r') as load_f:
#     original_dict1 = json.load(load_f)
#     load_f.close()
#     # print(original_dict1)
# original_dict1['trainPairCount'] = 0
# original_dict1['testPairCount'] = 0
# original_dict1['validPairCount'] = 1000

# a=original_dict
# b=original_dict1

# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_lightness_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_lightness_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(-30,40,10):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['lightness_pertubation']=int(j)
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_lightness_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_lightness_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()

# if not os.path.isdir(dataset_name + '_titleFontSize_test'):
#     os.mkdir(dataset_name + '_titleFontSize_test')
# if not os.path.isdir(dataset_name + '_titleFontSize_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_titleFontSize_test/' + 'detail')

# path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
# with open(path,'r') as load_f:
#     original_dict = json.load(load_f)
#     load_f.close()
#     # print(original_dict)
# path1 = '{}.json'.format(dataset_name)
# with open(path1,'r') as load_f:
#     original_dict1 = json.load(load_f)
#     load_f.close()
#     # print(original_dict1)
# original_dict1['trainPairCount'] = 0
# original_dict1['testPairCount'] = 0
# original_dict1['validPairCount'] = 1000

# a=original_dict
# b=original_dict1

# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_titleFontSize_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_titleFontSize_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(9,16,1):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['TitleFontSize']=int(j)
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_titleFontSize_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_titleFontSize_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()
    
    
# if not os.path.isdir(dataset_name + '_maskLength_test'):
#     os.mkdir(dataset_name + '_maskLength_test')
# if not os.path.isdir(dataset_name + '_maskLength_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_maskLength_test/' + 'detail')

# path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
# with open(path,'r') as load_f:
#     original_dict = json.load(load_f)
#     load_f.close()
#     # print(original_dict)
# path1 = '{}.json'.format(dataset_name)
# with open(path1,'r') as load_f:
#     original_dict1 = json.load(load_f)
#     load_f.close()
#     # print(original_dict1)
# original_dict1['trainPairCount'] = 0
# original_dict1['testPairCount'] = 0
# original_dict1['validPairCount'] = 1000

# a=original_dict
# b=original_dict1

# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_maskLength_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_maskLength_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(1,0,-0.1):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['mask']['maskLength']=j
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_maskLength_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_maskLength_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()


# if not os.path.isdir(dataset_name + '_maskgap_test'):
#     os.mkdir(dataset_name + '_maskgap_test')
# if not os.path.isdir(dataset_name + '_maskgap_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_maskgap_test/' + 'detail')

# path = 'detail/posLen_random_c/{}.json'.format(data_detail_name)
# with open(path,'r') as load_f:
#     original_dict = json.load(load_f)
#     load_f.close()
#     # print(original_dict)
# path1 = '{}.json'.format(dataset_name)
# with open(path1,'r') as load_f:
#     original_dict1 = json.load(load_f)
#     load_f.close()
#     # print(original_dict1)
# original_dict1['trainPairCount'] = 0
# original_dict1['testPairCount'] = 0
# original_dict1['validPairCount'] = 1000

# a=original_dict
# b=original_dict1

# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_maskgap_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_maskgap_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(1,10,2):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['mask']['maskgap']=int(j)
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_maskgap_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_maskgap_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()