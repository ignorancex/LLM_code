import itertools
import json
import argparse
import os
import copy
import random
import numpy as np
from random_words import RandomWords

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datasetName',type=str, default='posLen_tp_1_rand_c_mask')
parser.add_argument('--dataNum',type=int, default=1000)
args = parser.parse_args()

dataset_name = args.datasetName
data_detail_name = 'position_length_type_1_mask'

# if not os.path.isdir(dataset_name + '_markSize_test'):
#     os.mkdir(dataset_name + '_markSize_test')
# if not os.path.isdir(dataset_name + '_markSize_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_markSize_test/' + 'detail')

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
# original_dict1['validPairCount'] = args.dataNum

# a=original_dict
# b=original_dict1

# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_markSize_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_markSize_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(0,3):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['mark']['markSize']=int(j)
#     fp = open(dataset_name + '_markSize_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_markSize_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()

if not os.path.isdir(dataset_name + '_data_test'):
        os.mkdir(dataset_name + '_data_test')
if not os.path.isdir(dataset_name + '_data_test/' + 'detail' ):
    os.mkdir(dataset_name + '_data_test/' + 'detail')

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

fp = open(dataset_name + '_data_test/detail/testdata_detail_{}.json'.format(i),'w')
json.dump(new_dict,fp,indent=4)
fp.close()

new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
fp = open('{}_data_test/testdata_{}.json'.format(dataset_name,i),'w')
json.dump(new_dict1,fp,indent=4)
fp.close()

i=0

for j in np.arange(3):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    if int(j)==0:
        new_dict['values']['valueRange']=[1,10]
    elif int(j)==1:
        new_dict['values']['valueRange']=[93,100]
    else:
        new_dict['values']['valueRange']=[1,10,93,100]
    # if j==2 or j==3:
    #     new_dict['fixBarGap']=2
    fp = open(dataset_name + '_data_test/detail/testdata_detail_{}.json'.format(i),'w')

    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_data_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

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

for j in np.arange(2,9,1):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['mark']['bottomValue']=int(j)
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

if not os.path.isdir(dataset_name + '_barWidth_test'):
    os.mkdir(dataset_name + '_barWidth_test')
if not os.path.isdir(dataset_name + '_barWidth_test/' + 'detail' ):
    os.mkdir(dataset_name + '_barWidth_test/' + 'detail')

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
original_dict1['validPairCount'] = args.dataNum

a=original_dict
b=original_dict1

i=0
new_dict = copy.deepcopy(a)
new_dict1 = copy.deepcopy(b)
new_dict['train']=False
fp = open(dataset_name + '_barWidth_test/detail/testdata_detail_{}.json'.format(i),'w')
json.dump(new_dict,fp,indent=4)
fp.close()

new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
fp = open('{}_barWidth_test/testdata_{}.json'.format(dataset_name,i),'w')
json.dump(new_dict1,fp,indent=4)
fp.close()

i=0

for j in np.arange(5,10):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['barWidth']=int(j)
    if j==8:
        new_dict['fixBarGap']=1
    if j==9:
        new_dict['fixBarGap']=0
    fp = open(dataset_name + '_barWidth_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_barWidth_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()


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

for j in np.arange(0,3):
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

# for j in np.arange(-60,80,20):
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


if not os.path.isdir(dataset_name + '_title_test'):
    os.mkdir(dataset_name + '_title_test')
if not os.path.isdir(dataset_name + '_title_test/' + 'detail' ):
    os.mkdir(dataset_name + '_title_test/' + 'detail')

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
fp = open(dataset_name + '_title_test/detail/testdata_detail_{}.json'.format(i),'w')
json.dump(new_dict,fp,indent=4)
fp.close()

new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
fp = open('{}_title_test/testdata_{}.json'.format(dataset_name,i),'w')
json.dump(new_dict1,fp,indent=4)
fp.close()

i=0

for j in np.arange(0.05,1,0.15):
    # for j in np.arange(0,1,0.1):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['TitlePosition']='left'
    new_dict['TitlePaddingLeft']=j
    fp = open(dataset_name + '_title_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_title_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

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

# for j in np.arange(-15,20,5):
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

if not os.path.isdir(dataset_name + '_titleFontSize_test'):
    os.mkdir(dataset_name + '_titleFontSize_test')
if not os.path.isdir(dataset_name + '_titleFontSize_test/' + 'detail' ):
    os.mkdir(dataset_name + '_titleFontSize_test/' + 'detail')

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
fp = open(dataset_name + '_titleFontSize_test/detail/testdata_detail_{}.json'.format(i),'w')
json.dump(new_dict,fp,indent=4)
fp.close()

new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
fp = open('{}_titleFontSize_test/testdata_{}.json'.format(dataset_name,i),'w')
json.dump(new_dict1,fp,indent=4)
fp.close()

i=0

for j in np.arange(9,16,1):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['TitleFontSize']=int(j)
    # if j==2 or j==3:
    #     new_dict['fixBarGap']=2
    fp = open(dataset_name + '_titleFontSize_test/detail/testdata_detail_{}.json'.format(i),'w')
    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_titleFontSize_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()
    

# if not os.path.isdir(dataset_name + '_imagePadding_test'):
#     os.mkdir(dataset_name + '_imagePadding_test')
# if not os.path.isdir(dataset_name + '_imagePadding_test/' + 'detail' ):
#     os.mkdir(dataset_name + '_imagePadding_test/' + 'detail')

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
# original_dict1['validPairCount'] = args.dataNum

# a=original_dict
# b=original_dict1


# variable_name=['Horizontal_layout_original']

# # Direction=['Horizontal','vertical']
# Horizontal_layout_original=[[35,55]] #14

# results=[]
# # results.append(AxisIntervals)
# # results.append(Axislabel)
# # results.append(Yaxisvalue)

# # results.append(Direction)
# results.append(Horizontal_layout_original)


# i=0
# new_dict = copy.deepcopy(a)
# new_dict1 = copy.deepcopy(b)
# new_dict['train']=False
# fp = open(dataset_name + '_imagePadding_test/detail/testdata_detail_{}.json'.format(i),'w')
# json.dump(new_dict,fp,indent=4)
# fp.close()

# new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
# fp = open('{}_imagePadding_test/testdata_{}.json'.format(dataset_name,i),'w')
# json.dump(new_dict1,fp,indent=4)
# fp.close()

# i=0

# for j in np.arange(17,24,1):
#     i+=1
#     new_dict = copy.deepcopy(a)
#     new_dict1 = copy.deepcopy(b)
#     new_dict['train']=False
#     new_dict['imagePadding']=int(j)
#     # if j==2 or j==3:
#     #     new_dict['fixBarGap']=2
#     fp = open(dataset_name + '_imagePadding_test/detail/testdata_detail_{}.json'.format(i),'w')
#     json.dump(new_dict,fp,indent=4)
#     fp.close()
#     # print(new_dict)

#     new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
#     fp = open('{}_imagePadding_test/testdata_{}.json'.format(dataset_name,i),'w')
#     json.dump(new_dict1,fp,indent=4)
#     fp.close()  

if not os.path.isdir(dataset_name + '_strokeColorL_test'):
        os.mkdir(dataset_name + '_strokeColorL_test')
if not os.path.isdir(dataset_name + '_strokeColorL_test/' + 'detail' ):
    os.mkdir(dataset_name + '_strokeColorL_test/' + 'detail')

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

fp = open(dataset_name + '_strokeColorL_test/detail/testdata_detail_{}.json'.format(i),'w')
json.dump(new_dict,fp,indent=4)
fp.close()

new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
fp = open('{}_strokeColorL_test/testdata_{}.json'.format(dataset_name,i),'w')
json.dump(new_dict1,fp,indent=4)
fp.close()

i=0

for j in np.arange(-5,30,5):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['strokecolorL_perturbation']=int(j)
    new_dict['strokeLABperturbation']=True
    # if j==2 or j==3:
    #     new_dict['fixBarGap']=2
    fp = open(dataset_name + '_strokeColorL_test/detail/testdata_detail_{}.json'.format(i),'w')

    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_strokeColorL_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()

if not os.path.isdir(dataset_name + '_barColorL_test'):
    os.mkdir(dataset_name + '_barColorL_test')
if not os.path.isdir(dataset_name + '_barColorL_test/' + 'detail' ):
    os.mkdir(dataset_name + '_barColorL_test/' + 'detail')

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

fp = open(dataset_name + '_barColorL_test/detail/testdata_detail_{}.json'.format(i),'w')
json.dump(new_dict,fp,indent=4)
fp.close()

new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
fp = open('{}_barColorL_test/testdata_{}.json'.format(dataset_name,i),'w')
json.dump(new_dict1,fp,indent=4)
fp.close()

i=0

for j in np.arange(-15,20,5):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['barcolorL_perturbation']=int(j)
    new_dict['barLABperturbation']=True
    # if j==2 or j==3:
    #     new_dict['fixBarGap']=2
    fp = open(dataset_name + '_barColorL_test/detail/testdata_detail_{}.json'.format(i),'w')

    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_barColorL_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()
    
if not os.path.isdir(dataset_name + '_bgColorL_test'):
    os.mkdir(dataset_name + '_bgColorL_test')
if not os.path.isdir(dataset_name + '_bgColorL_test/' + 'detail' ):
    os.mkdir(dataset_name + '_bgColorL_test/' + 'detail')

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

fp = open(dataset_name + '_bgColorL_test/detail/testdata_detail_{}.json'.format(i),'w')
json.dump(new_dict,fp,indent=4)
fp.close()

new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
fp = open('{}_bgColorL_test/testdata_{}.json'.format(dataset_name,i),'w')
json.dump(new_dict1,fp,indent=4)
fp.close()

i=0

for j in np.arange(-25,10,5):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['bgcolorL_perturbation']=int(j)
    new_dict['LABperturbation']=True
    # if j==2 or j==3:
    #     new_dict['fixBarGap']=2
    fp = open(dataset_name + '_bgColorL_test/detail/testdata_detail_{}.json'.format(i),'w')

    json.dump(new_dict,fp,indent=4)
    fp.close()
    # print(new_dict)

    new_dict1['param'] = '{$include}' + ' detail/testdata_detail_{}.json'.format(i)
    fp = open('{}_bgColorL_test/testdata_{}.json'.format(dataset_name,i),'w')
    json.dump(new_dict1,fp,indent=4)
    fp.close()