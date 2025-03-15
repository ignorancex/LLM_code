import itertools
import json
import argparse
import os
import copy
import random
import numpy as np
from random_words import RandomWords

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--datasetName',type=str, default='posLen_tp_2_rand_c')
parser.add_argument('--dataNum',type=int, default=50)
args = parser.parse_args()

dataset_name = args.datasetName
data_detail_name = 'position_length_type_2'


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

for j in np.arange(27,34,1):
    i+=1
    new_dict = copy.deepcopy(a)
    new_dict1 = copy.deepcopy(b)
    new_dict['train']=False
    new_dict['stackWidth']=int(j)
    # if j==2 or j==3:
    #     new_dict['fixBarGap']=2
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
original_dict1['validPairCount'] = args.dataNum

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
original_dict1['validPairCount'] = args.dataNum

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
original_dict1['validPairCount'] = args.dataNum

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
original_dict1['validPairCount'] = args.dataNum

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

 
if not os.path.isdir(dataset_name + '_strokeColorL_test'):
    os.mkdir(dataset_name + '_strokeColorL_test')
if not os.path.isdir(dataset_name + '_strokeColorL_test/' + 'detail' ):
    os.mkdir(dataset_name + '_strokeColorL_test/' + 'detail')


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
original_dict1['validPairCount'] = args.dataNum

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
original_dict1['validPairCount'] = args.dataNum

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