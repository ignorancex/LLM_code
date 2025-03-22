from math import log2
import math
import cv2
import os
import json
import numpy as np
import Dataset.UtilIO as uio

#/home/zhenxing/graph_perception/test_result_4/raw_result/resnet152_ele_angle/tests/ele_angle_testdata_0/Iter_69000/resultinfo.json
#/home/zhenxing/graph_perception/test_result_4/dataset/ele_angle_testdata_0/valid
# path="./test_result_4/raw_result/"
# target_path="./test_result_4/dataset/"
# # predicts=[]
# folders=os.listdir(path)
# for i in folders:
#     tests=path+i+"/tests/"
#     result_folders=os.listdir(tests)
#     for j in result_folders:
#         dataListPath=target_path+j+"/valid/list.json"
#         dataTargetFolder=target_path+j+"/valid/target/"
#         iters=tests+j+"/"
#         iters_folders=os.listdir(iters)
#         predicts=[]
#         for k in iters_folders:
#             result=iters+k+"/"
#             file_name=result+"resultInfo.json"
#             with open(file_name,'r') as f:
#                 pv=json.load(f)
#             for v in pv:
#                 predicts.append(v['pred_n'][0][0])
#             # print(file_name)
#             # print(len(predicts))
#             dataFileList = uio.load(dataListPath,"json")
#             dataFile = [os.path.join(dataTargetFolder,x[1]["num"]) for x in dataFileList]
#             target=[]
#             for d in dataFile:
#                 f = open(d,"r")
#                 obj = json.load(f)
#                 f.close()
#                 target.append(obj)
#             # print(len(target)) 
#             sum=0
#             for index in range(len(target)):
#                 if isinstance(predicts[index],int):
#                     sum+=math.log2(abs(target[index]-predicts[index])+0.125)
#                 elif isinstance(predicts[index],list):
#                     pass
#             mlae=sum/len(target)
#             print(mlae)


# dataListPath = os.path.join(config.data.validPath,"list.json")
# dataFileList = uio.load(config.mlae.dataListPath,"json")
# config.mlae.dataTargetFolder = os.path.join(config.data.validPath,config.data.outputFolder)

# dataListPath="./test_result_4/dataset/ele_angle_testdata_0/valid/list.json"
# dataTargetFolder="./test_result_4/dataset/ele_angle_testdata_0/valid/target/"
# path="./test_result_4/raw_result/resnet152_ele_angle/tests/ele_angle_testdata_0/Iter_69000/resultInfo.json"
# savePath=path="./test_result_4/raw_result/resnet152_ele_angle/final/"

# with open(path,'r') as f:
#     pv=json.load(f)
# predicts=[]
# for v in pv:
#     predicts.append(v['pred_n'][0])
# dataFileList = uio.load(dataListPath,"json")
# dataFile = [os.path.join(dataTargetFolder,x[1]["num"]) for x in dataFileList]
# target=[]
# for d in dataFile:
#     f = open(d,"r")
#     obj = json.load(f)
#     f.close()
#     target.append(obj)
# print(len(target)) 
# sum=0
# for index in range(len(target)):
#     if isinstance(predicts[index],int):
#         sum+=math.log2(abs(target[index]-predicts[index])+0.125)
#     elif isinstance(predicts[index],list):
#         lossTotal=0
#         lossCount=0
#         for ind in range(len(predicts[index])): #1 or 4
#             lossTotal+=math.log2(abs(target[index][ind]-predicts[index][ind])*100+0.125)
#             lossCount+=1
#         sum+=lossTotal/lossCount
#         pass
# mlae=sum/len(target)
# print(mlae)

path="./test_result_4/raw_result/"
target_path="./test_result_4/dataset/"
# predicts=[]
folders=os.listdir(path)
for i in folders:
    # if not 'posL' in i and not 'posA' in i:
    #     continue
    if not 'resnet152_posLen_tp_15_rand_c_lw' in i:
        continue
    print(i)
    # if not 'posAngle_pie' in i:
    #     continue
    tests=path+i+"/tests/"
    result_folders=os.listdir(tests)
    for j in result_folders:
        # if not 'bgColorL' in j and not 'dataRange' in j and not 'markPosition' in j:
        if not 'human' in j:
            continue
        print(j)
        dataListPath=target_path+j+"/valid/list.json"
        dataTargetFolder=target_path+j+"/valid/target/"
        iters=tests+j+"/"
        iters_folders=os.listdir(iters)
        predicts=[]
        for k in iters_folders:
            result=iters+k+"/"
            file_name=result+"resultInfo.json"
            with open(file_name,'r') as f:
                pv=json.load(f)
            predicts=[]
            for v in pv:
                predicts.append(v['pred_n'][0])
            dataFileList = uio.load(dataListPath,"json")
            dataFile = [os.path.join(dataTargetFolder,x[1]["num"]) for x in dataFileList]
            target=[]
            for d in dataFile:
                f = open(d,"r")
                obj = json.load(f)
                f.close()
                target.append(obj)
            # print(len(target)) 
            # print(len(target)) 
            # print(result)

            sum=0
            sums=[]
            for index in range(len(target)):
                if isinstance(predicts[index],int):
                    sum+=math.log2(abs(target[index]-predicts[index])+0.125)
                    sums.append(math.log2(abs(target[index]-predicts[index])+0.125))
                elif isinstance(predicts[index],list):
                    lossTotal=0
                    lossCount=0
                    for ind in range(len(predicts[index])): #1 or 4
                        # print(target[index][ind])
                        # print(index)
                        # lossTotal+=math.log2(abs(target[index][ind]-predicts[index][ind])/target[index][ind]*100+0.125)
                        lossTotal+=math.log2(abs(target[index][ind]-predicts[index][ind])*100+0.125)
                        lossCount+=1
                    sum+=lossTotal/lossCount
                    sums.append(lossTotal/lossCount)
                    pass
            mlae=sum/len(target)
            
            sorted_arr=sorted(sums)
            quarter=len(sorted_arr)//4
            data=sorted_arr[quarter:-quarter]

            MLAE_True=np.mean(data)

            savePath=path+i+"/final_2/"
            resultPath=path+i+"/final_3/"
            if not os.path.isdir(savePath):
                os.mkdir(savePath)
            savename=savePath+j+"_"+k+".json"
            result_name=resultPath+j+"_"+k+"_result.json"
            result_={}
            result_['predict']=predicts
            result_['true']=target
            finalResult=[]
            finalResult.append({"MLAE":mlae,"MLAE_True":MLAE_True})
            # print("save",savePath)
            uio.save(savename,finalResult,"json_format")
            uio.save(result_name,result_,"json")
    # break

