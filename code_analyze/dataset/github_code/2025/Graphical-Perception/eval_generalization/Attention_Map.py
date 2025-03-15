import numpy as np
import torch
import torchvision
import torch.nn as nn
import os
import argparse
from Networks import *
import torchvision.transforms as transforms
import logging
import Dataset
import Dataset.UtilIO as uio
import sys
import util.parallelUtil as para
import random
import util
import util.pyutils as upy
from util.pyutils import multiImport
from util import Config
from util.Config import ConfigObj
import util.shareCode
from util.shareCode import programInit, globalCatch
import GenDataset
import MLAECompute
import json
import os.path as osp
import matplotlib.cm as cm
import cv2

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from grad_cam.utils import GradCAM, show_cam_on_image, center_crop_img
from grad_cam_ import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

def collate_fn_inner(data,dic):
    for k,v in data.items():
        if isinstance(v,dict):
            if k in dic.keys():
                collate_fn_inner(v,dic[k])
            else:
                subdic={}
                collate_fn_inner(v,subdic)
                dic[k]=subdic
        elif isinstance(v,list):
            if k in dic.keys():
                dic[k]+=v
            else:
                dic[k]=v
        else: # tensor
            if k in dic.keys():
                torch.stack((dic[k],v.unsqueeze(0)))
            else:
                dic[k]=v.unsqueeze(0)

def collate_fn(all_data):
    dic={}
    for x in all_data:
        collate_fn_inner(x,dic)
    return dic
    
    #
class TestHelper:

    def __init__(self,device,config):
        #global config
        self.config=config
        self.device=device

        if len(config.refer.__dict__)>4:
            logging.info("Detect related config, borrow related config, except dataset config")
            self.config.model = config.refer.model
            self.config.testOption.model.basicPath = config.refer.modelOutput.modelName
            self.testResultOutputPath = config.refer.test.testResultOutputPath
            if self.config.utility.mainPath != 'result':
                self.config.testOption.model.basicPath = self.config.testOption.model.basicPath.replace(self.config.utility.mainPath,'result')
                self.testResultOutputPath = self.testResultOutputPath.replace(self.config.utility.mainPath,'result')

            self.config.trainParam = config.refer.trainParam
            self.name = config.refer.name


        ConfigObj.default(self.config,"utility.debug",False)
        ConfigObj.default(self.config,"cuda.parallel",False)
        ConfigObj.default(self.config,"cuda.use_gpu",[0,1])
        ConfigObj.default(self.config,"continueTrain.enableContinue",False)
        ConfigObj.default(self.config,"test.displayImageIndex",0)
        ConfigObj.default(self.config,"test.storeImageIndex",[0])
        ConfigObj.default(self.config,"utility.globalSeed",0)
        ConfigObj.default(self.config,"data.trainSeed",100)
        ConfigObj.default(self.config,"data.validSeed",10000)
        ConfigObj.default(self.config,"data.inputFolder","input")
        ConfigObj.default(self.config,"data.outputFolder","target")
        ConfigObj.default(self.config,"data.orgFolder","org")
        ConfigObj.default(self.config,"trainParam.batchSize",1)
        ConfigObj.default(self.config,"trainParam.learnType","Adam")
        ConfigObj.default(self.config,"trainParam.learnRate",0.001)
        ConfigObj.default(self.config,"trainParam.adam_weight_decay",0.0005)
        ConfigObj.default(self.config,"trainParam.learnRateMulti","1.0")
        ConfigObj.default(self.config,"trainParam.clipNorm",5)
        ConfigObj.default(self.config,"testOption.dataListPath","")
        ConfigObj.default(self.config,"testOption.outputResult",r"{utility.mainPath}/raw_result/{name}")

        # gener = GenDataset.GenDataset(self.config)
        # gener.genData()
        self.model=None
        

    def initLeft(self):
        config = self.config
        device = self.device
        # load model
        singleModelName = self.config.model.name.split(".")[-1]
        modelName = "Networks.%s.%s"%(self.config.model.name,singleModelName)
        logging.info("Create Model %s"%modelName)
        modelClass=multiImport(modelName)
        self.model=None
        try:
            logging.info("Try to pass dict")
            self.model = modelClass(**Config.obj2dic(config.model.param))
        except:
            logging.info("Direct pass params")
            self.model = modelClass(config.model.param)
            # with open('a.json','w') as f:
            #     json.dump(config,f)
            print(str(config))

        self.model.setConfig(config,device)

        self.setSeed(config.utility.globalSeed)

        # load data
        dataListPath = self.config.testOption.dataListPath
        
        if not isinstance(dataListPath,str) or len(dataListPath)==0:
            dataListPath = os.path.join(self.config.data.validPath,"list")
            logging.info("Data list path is None, use %s instead"%dataListPath)
        paths = uio.load(dataListPath,"json")

        paths = uio.load(os.path.join(self.config.data.validPath,"list"),"json")

        datasetManagerName = "Dataset.%s.%s"%(config.data.manageType,config.data.manageType)
        self._dataManagerClass = multiImport(datasetManagerName)
        
        self.testData = self._dataManagerClass(config.data.validPath,config,paths,"exp")

        self.testDataLoader = torch.utils.data.DataLoader(self.testData,batch_size=1,shuffle=False,pin_memory=True,num_workers=0,collate_fn=collate_fn)

        self.debug = self.config.utility.debug

        self.setSeed(config.utility.globalSeed)


    def setSeed(self,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def load(self, path):
        self.model = self.model.cpu()
        modelName = path#self.config.testOption.modelPath 
        logging.info("Load model %s"%modelName)

        self.model.load_state_dict(torch.load(modelName))
        self.model=self.model.to(self.device)
        print(modelName)

        self.model.setConfig(self.config,self.device)

    def findModels(self):
        pth = self.config.testOption.model.basicPath
        if pth is None or len(pth)==0:
            logging.error("basicPath is none")
        folder, filebasicName = os.path.split(pth)
        otherName = filebasicName.strip().replace(".pkl","").replace("%d","")
        flist=[]
        for root,dirs,files in os.walk(folder):
            for f in files:
                if f.endswith(".pkl"):
                    temp = f.replace(otherName,"").replace(".pkl","").strip()
                    iterValue = 0
                    try:
                        iterValue=int(temp)
                    except BaseException as e:
                        logging.warning("Ignore model file (not match) %s"%f)
                        continue
                    if iterValue<self.config.testOption.model.minIter:
                        logging.warning("Skip model file (low iter) %s"%f)
                        continue
                    realPath=os.path.join(root,f)
                    flist.append((iterValue,realPath))
                    logging.info("Detect model file %s"%realPath)
        logging.info("Discover %d models"%len(flist))
        if len(flist)==0:
            logging.warning("Cannot find any models, please check the configuration")
            logging.warning("path %s"%pth)
        return flist

    def test(self,storePath):
        os.makedirs(storePath,exist_ok=True)
        self.model.eval()
        lossInfo=[]
        lossTotalInfo={}
        resultInfo=[]
        imageInfo=[]
        testIter=0
        testCount = len(self.testDataLoader)
        logging.info("Test Begin! Num: %d"%testCount)
        for x in self.testDataLoader:
            testIter+=1
            uio.logProgress(testIter,testCount,"Test Model")
            vv = self.model.test(x)
            # process loss
            vloss=vv["loss"]
            vloss2={}
            for k,v in vloss.items():
                vi=v.detach().cpu().item()
                vloss2[k]=vi
                if k not in lossTotalInfo:
                    lossTotalInfo[k]=vi
                else:
                    lossTotalInfo[k]+=vi
            lossInfo.append(vloss2)
            # process result
            resultDic={}
            for k,v in vv["result"].items():
                if v is not None:
                    resultDic[k]=v.detach().cpu().numpy().tolist()
                else:
                    resultDic[k]=None
            # print(resultDic)
            resultInfo.append(resultDic)

            if "images" in vv.keys():
                imageInfo.append(v["images"])
        #average
        for k in lossTotalInfo.keys():
            lossTotalInfo[k]/=testCount
        
        # visualize loss
        s="Test Result |"
        for k,v in lossTotalInfo.items():
            s+=" %s %8s |"%(k,str(v))
        logging.info(s+"    ")

        # store test content
        logging.debug("Test complete, store test results at %s"%storePath)
        os.makedirs(storePath,exist_ok=True)
        # store loss
        uio.save(os.path.join(storePath,"lossInfo"),lossInfo,"json")
        uio.save(os.path.join(storePath,"lossTotalInfo"),lossTotalInfo,"json")
        # store result data
        uio.save(os.path.join(storePath,"resultInfo"),resultInfo,"json")
        # store images
        for i in self.config.test.storeImageIndex:
            if i>=len(imageInfo):
                continue
            imInfo = imageInfo[i]
            for k,v in imInfo.items():
                imgStorePath = os.path.join(storePath,"test_img_%5d_%s.png"%(i,k))
                transforms.ToPILImage()(v.detach().cpu()).save(imgStorePath)
        

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    # cv2.imwrite(filename, np.uint8(gcam))
    plt.imshow(np.uint8(gcam)[:,:,::-1])
    plt.savefig(filename)
    plt.show()


def runTest(config,device):
    th = TestHelper(device,config)
    models = th.findModels()
    sorted(models)
    # models=[(1,"/home/disk0/graph_perception/result/resnet152_posLen_tp_1_rand_c_lw/model_1260000.pkl")]
    
    config.mlae.computeList=[]

    if config.testOption.model.minIter==0:
        logging.info("Try to find model with minimum valid loss to test")
        minLoss=None
        minIter=0
        minIndex=0
        i=0
        for iters,modelPath in models:
            folder = os.path.join(th.testResultOutputPath,config.refer.test.testResultIterOutputFolder%iters)
            print(folder)
            f = os.path.join(folder,"lossTotalInfo")
            obj = uio.load(f,"json")
            maxLoss=-1.0
            for k,v in obj.items():
                maxLoss=max(v,maxLoss)
            logging.info("Model loss %f"%maxLoss)
            if minLoss is None or minLoss > maxLoss:
                minLoss=maxLoss
                minIter=iters
                minIndex=i
            i+=1

        logging.info("Decide to use Model (Iter %d), min loss %f"%(minIter,minLoss))

        newModels = [models[minIndex]]
        print(newModels)
        for iters, modelPath in models:
            resultFile=os.path.join(config.testOption.outputResult%iters,"resultInfo.json")
            if os.path.exists(resultFile) and iters!=newModels[0][0]:
                logging.info("Iter %d is completed, decide to compute loss!"%iters)
                newModels.append((iters,modelPath))
        models=newModels[1:]
        models.append(newModels[0])
    print(models)
    init=False
    pointIndex=0
    for iter,modelPath in models:
        resultFile=os.path.join(config.testOption.outputResult%iter,"resultInfo.json")
        # if os.path.exists(resultFile):
        #     logging.warning("Detect result file, skip test %s"%(config.testOption.outputResult%iter))
        #     logging.info("Related result file %s"%resultFile)
        # else:
        if not init:
            th.initLeft()
            init=True
        th.load(modelPath)
        model=th.model
        layer={
            # "layer1":th.model.network.layer1,
            # "layer2":th.model.network.layer2,
            # "layer3":th.model.network.layer3,
            "layer4":th.model.network.layer4
        }
        if not os.path.isdir('./hhh/'+th.config.data.validPath.split('/')[2]):
            os.mkdir('./hhh/'+th.config.data.validPath.split('/')[2])
        
        print(th.config.data.validPath.split('/')[2])
        for k,v in layer.items():

            # print(th.model.network)
            target_layers = [v]
            d=None
            count=0
            # for pointIndex in range(1000):

            for x in th.testDataLoader:
                pointIndex+=1
                # d=x
                # if count==pointIndex:
                #     d=x
                #     # print(x)
                #     # print(x['input']['img'].shape)
                #     break
                # count+=1

                img_path=os.path.join(th.config.data.validPath,"input",uio.load(os.path.join(th.config.data.validPath,"list"),"json")[count][0]['img'])
                # print("img_path",img_path)
                # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
                img = Image.open(img_path,'r').convert('RGB').resize((224,224),Image.ANTIALIAS)
                img = np.array(img, dtype=np.uint8)
                target_layer = layer["layer4"]
                target_class = 243  # "bull mastif"
                gcam = GradCAM(model=model)
                ids = gcam.forward(x)
                print("lllllllllllll",ids)
                ids_ = x['target']['num'].to(device)
                print("jjjjjjjjjj","====",str(pointIndex),ids_)
                loss=(ids-ids_)*(ids-ids_)
                gcam.backward(ids=loss)
                # gcam.backward(ids=ids)

                print("Generating Grad-CAM @")

                # Grad-CAM
                regions = gcam.generate(target_layer='network.layer4')

                for j in range(1):
                    # print(
                    #     "uuuuuuuuuuuuu"
                    # )

                    # save_gradcam(
                    #     filename=osp.join(
                    #         "./hhh/"+th.config.data.validPath.split('/')[2],
                    #         str(th.config.data.validPath.split('/')[2])+'-'+str(pointIndex)+'-'+k+'.svg'
                    #         # "{}-{}-gradcam-{}-{}.png".format(
                    #         #     j, "resnet152", 'target_layer', 'classes[target_class]'
                    #         # ),
                    #     ),
                    #     gcam=regions[j, 0],
                    #     raw_image=img.astype(dtype=np.float32),
                    # )
                    save_gradcam(
                        filename=osp.join(
                            "./hhh/"+th.config.data.validPath.split('/')[2],
                            str(th.config.data.validPath.split('/')[2])+'-'+str(pointIndex)+'-'+k+'.png'
                            # "{}-{}-gradcam-{}-{}.png".format(
                            #     j, "resnet152", 'target_layer', 'classes[target_class]'
                            # ),
                        ),
                        gcam=regions[j, 0],
                        raw_image=img.astype(dtype=np.float32),
                    )
            
            '''
            img_tensor=th.testData[0]['input']['img']
            # data_transform = transforms.Compose([transforms.ToTensor()])
            # data_transform = transforms.Compose([transforms.Resize(size=224),transforms.ToTensor()])

            # # load image
            img_path = "./grad_cam/bar_type1_59.png"
            print("th.config.data.validPath",th.config.data.validPath.split('/')[2])
            img_path=os.path.join(th.config.data.validPath,"input",uio.load(os.path.join(th.config.data.validPath,"list"),"json")[count][0]['img'])
            print("img_path",img_path)
            # assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path,'r').convert('RGB').resize((224,224),Image.ANTIALIAS)
            img = np.array(img, dtype=np.uint8)
            # # img = center_crop_img(img, 224)

            # # [C, H, W]
            # img_tensor = data_transform(img)
            # print(img_tensor.shape)
            # # expand batch dimension
            # # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
            target_category = 281  # tabby, tabby cat
            # target_category = 254  # pug, pug-dog

            tar = d['target']['num']
            print('tar',tar)
            grayscale_cam = cam(input_tensor=d,target=tar)
            # grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
            # grayscale_cam = cam(input_tensor=d)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                            grayscale_cam,
                                            use_rgb=True)
            plt.imshow(visualization)
            # plt.savefig('./grad_cam/'+str(th.config.data.validPath.split('/')[2])+'-'+str(pointIndex)+'-'+k+'.png')
            plt.savefig('./grad_cam/'+str(th.config.data.validPath.split('/')[2])+'-'+str(pointIndex)+'-'+k+'.svg')
            plt.show()
            '''
            


def main():
    #parse params
    config = programInit()
    print(config.model)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda.detectableGPU
    device = torch.device("cpu")
    logging.info(config.cuda.detectableGPU)
    if torch.cuda.is_available():
        logging.info("Detect GPU, Use gpu to train the model")
        device = torch.device("cuda")

    runTest(config,device)

    




if __name__ == '__main__':
    globalCatch(main)