# This is the File for training the CNN encoder.

###############################################################################
# Importing necessary libraries

import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
import time
from Encoder.Models import ConfigurationHolder as CH
from Encoder.Models import CNN_encoder
from Encoder.Models import DataExtractor as DE
from torchsummary import summary
###############################################################################
from Encoder.Models.DataExtractor import SquarePad
from utils.annotation_parser import JsonParser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CNN_selection = 1 # 0 = original one, 1 = one with pooling

if CNN_selection == 0:
    from Models import CNN_encoder
elif CNN_selection == 1:
    from Models import CNN_encoder_pooling as CNN_encoder
###############################################################################
# Retrieving path to current folder and to configuration file.
# Configuration file contains all the information regarding parameters.

# Current folder
project_dir = os.path.dirname(os.path.dirname(__file__))
pathCurrentFolder = os.path.join(project_dir, "Encoder")
# Path to configuration file (containing all paramters of nets)
pathConfigurationFile = pathCurrentFolder + '/Configuration/ConfigurationFile.json'
configHolder = CH.ConfigurationHolder()
configHolder.LoadConfigFromJSONFile(pathConfigurationFile)
# Path to locations file (containing the paths to the inputs and if to use GPU/CPU)
pathLocationsFile = pathCurrentFolder + '/Configuration/LocationsFile.json'
locationsHolder = CH.ConfigurationHolder()
locationsHolder.LoadConfigFromJSONFile(pathLocationsFile)

# Path to output Folder
outputFolder = locationsHolder.config['output_folder']

# Take out the path to the annotations datasets
pathAnnotationsFileTraining = locationsHolder.config['annotations_folder_training']
pathAnnotationsFileValidation = locationsHolder.config['annotations_folder_validation']
# Create a configuration holder for the annotations
annotationsHolderTraining = CH.ConfigurationHolder()
annotationsHolderTraining.LoadConfigFromJSONFile(pathAnnotationsFileTraining)
annotationsHolderValidation = CH.ConfigurationHolder()
annotationsHolderValidation.LoadConfigFromJSONFile(pathAnnotationsFileValidation)

# Take out the path to the images
pathImagesFileTraining = locationsHolder.config['images_folder_training']
pathImagesFileValidation = locationsHolder.config['images_folder_validation']

###############################################################################
# Creating data holders for training and validation data
# Training
dataExtractorTraining = DE.DataExtractor(annotationsHolderTraining, pathImagesFileTraining, configHolder)
# Take the mins and maxs of training and pass them to normalize the validation
# trainingMinsAndMaxs = dataExtractorTraining.minAverageDistance, dataExtractorTraining.maxAverageDistance, \
#                       dataExtractorTraining.minMass, dataExtractorTraining.maxMass
# dataExtractorTraining.minWidthTop, dataExtractorTraining.maxWidthTop, \
# dataExtractorTraining.minWidthBottom, dataExtractorTraining.maxWidthBottom, \
# dataExtractorTraining.minHeight, dataExtractorTraining.maxHeight
trainingMinsAndMaxs = dataExtractorTraining.minAverageDistance, dataExtractorTraining.maxAverageDistance, \
    dataExtractorTraining.minRatioWidth, dataExtractorTraining.maxRatioWidth, \
        dataExtractorTraining.minRatioHeight, dataExtractorTraining.maxRatioHeight, \
            dataExtractorTraining.minMass, dataExtractorTraining.maxMass
# Validation/Testing
dataExtractorValidation = DE.DataExtractor(annotationsHolderValidation, pathImagesFileValidation, configHolder,
                                           trainingMinsAndMaxs)

###############################################################################

# Initialize CNN and print it
if CNN_selection == 0:
    CNN = CNN_encoder.CNN_encoder(image_size=configHolder.config['x_size'],
                                  dim_filters=configHolder.config['dim_filters'],
                                  kernel=configHolder.config['kernels_size'],
                                  stride=configHolder.config['stride'],
                                  number_of_neurons_middle_FC=configHolder.config['number_of_neurons_middle_FC'],
                                  number_of_neurons_final_FC=configHolder.config['number_of_neurons_final_FC'],
                                  number_of_cameras=configHolder.config['number_of_cameras'],
                                  minValuesOutput=dataExtractorTraining.minValuesOutput,
                                  maxValuesOutput=dataExtractorTraining.maxValuesOutput)
elif CNN_selection == 1:
    CNN = CNN_encoder.CNN_encoder(minValuesOutput=dataExtractorTraining.minValuesOutput,
                                  maxValuesOutput=dataExtractorTraining.maxValuesOutput)


CNN.load_state_dict(
    torch.load(os.path.join(project_dir, os.path.join("demo","Encoder_pool_152.torch"))))
CNN.eval()

ann = JsonParser()
ann.load_json(pathAnnotationsFileValidation)
transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((112, 112), interpolation=F.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])
time_list = []
for i in range(0, len(ann.image_name)):
    file_name = ann.image_name[i]
    imgCurr = Image.open(os.path.join(pathImagesFileValidation, file_name))

    imgCurrTransformed = transform(imgCurr)
    # cv2.imshow("", imgCurrTransformed.numpy().transpose(1, 2, 0)[:,:,::-1])
    # cv2.waitKey(0)

    inputImages = torch.zeros(1, 3,
                              112, 112)
    inputImages[0, :, :, :] = imgCurrTransformed

    inputSingleValues = torch.zeros(1, 3)

    inputSingleValues[0, 0] = (ann.ar_w[i] - dataExtractorValidation.minRatioWidth) / (dataExtractorValidation.maxRatioWidth - dataExtractorValidation.minRatioWidth)
    inputSingleValues[0, 1] = (ann.ar_h[i] - dataExtractorValidation.minRatioHeight) / (dataExtractorValidation.maxRatioHeight - dataExtractorValidation.minRatioHeight)
    inputSingleValues[0, 2] = (ann.avg_d[i] - dataExtractorValidation.minAverageDistance) / (dataExtractorValidation.maxAverageDistance - dataExtractorValidation.minAverageDistance)
    if 20 <= i < 320:
        start_time = time.time()
    predictedValues = CNN(inputImages, inputSingleValues)
    if 20 <= i < 320:
        elapsed_time = time.time() - start_time
        time_list.append(elapsed_time)
    annSingleValues = torch.zeros(1, 3)
    annSingleValues[0, 0] = ann.mass[i]
    print("true: {}".format(ann.mass[i]))
    print("prediction: {}".format(
        int(max(0, CNN.CalculateOutputValueDenormalized(predictedValues, 1).cpu().detach().numpy()[0][0]))))

print("Images = {}".format(len(time_list)))
print("Mean time = {}".format(np.mean(np.array(time_list))))
print("Std dev time = {}".format(np.std(np.array(time_list))))