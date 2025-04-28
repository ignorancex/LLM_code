import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import test_images, test_single_nii
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import SimpleITK as sitk
from data_reader import H5DataLoader
import h5py
import uuid
import xml.etree.ElementTree as ET


def read_perf_xml(xmlfile):
    # Read XML for brain atlas
    xml_root = ET.parse(xmlfile).getroot()

    names = list()
    ids = list()


    for i in range(len(xml_root[2])):
        names.append(xml_root[2][i].get('name'))
        ids.append(np.float16(xml_root[2][i].get('value')))

    mydict = {names[i]: ids[i]
                for i in range(len(names))}

    return mydict





if __name__ == "__main__":

    xmlfile = 'out2.xml'
    
    a= read_perf_xml(xmlfile)



