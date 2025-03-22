# A class for extracting the dataset

import os
import torch
import glob
import numpy as np
import cv2
from PIL import Image
import random
import torchvision.transforms.functional as F
from torchvision import transforms


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

class BinaryImage:
    def __call__(self, image):
        
        print('image SHAPE')
        print(image.size)
        
        pixels = image.load()
        
        
        print('pixels SHAPE')
        print(pixels.shape)
        
        
        for i in range(image.size[0]):  # for every pixel:
            for j in range(image.size[1]):
                pixels[i, j] *= 255
        return image


class DataExtractor(object):

    def __init__(self, annotationsHolder, pathImagesFile, configHolder, trainingMinsAndMaxs=None):

        #######################################################################
        # Define parameters
        self.number_of_cameras = 1  # <---------------------------------------------------- For now used one camera only

        # Number of singled valued parameters
        self.numberOfInputSingleValues = 3 * self.number_of_cameras
        self.numberOfOutputSingleValues = 1 * self.number_of_cameras
        self.image_channels = 3 * self.number_of_cameras # 1

        self.batch_size = configHolder.config['batch_size']
        # Size of images
        self.x_size = configHolder.config['x_size']
        self.y_size = configHolder.config['y_size']

        ######################################################################

        if trainingMinsAndMaxs != None:
            # self.minAverageDistance, self.maxAverageDistance, self.minWidthTop, self.maxWidthTop, \
            # self.minWidthBottom, self.maxWidthBottom = trainingMinsAndMaxs
            self.minAverageDistance, self.maxAverageDistance, \
                self.minRatioWidth, self.maxRatioWidth, \
                    self.minRatioHeight, self.maxRatioHeight, \
                        self.minMass, self.maxMass = trainingMinsAndMaxs

                
        self.FindNumberOfImages(pathImagesFile)
        self.FindNumberOfBatches()

        #######################################################################
        # Extraction of the data

        # Extract the images
        inputImages = self.ExtractImagesFromPath(pathImagesFile, annotationsHolder)

        # Extact the normalization values for the single input elements
        if trainingMinsAndMaxs == None:
            self.ExtractMinAndMaxValuesOfInputsAndOutputs(annotationsHolder)
        else:
            self.CompactMaxsAndMinsValuesOutputs()

        # Extract the single input elements (i.e., annotation inputs and outputs)
        inputSingleValues, outputSingleValues = self.ExtractAnnotationInputsAndOutputs(
            annotationsHolder)
        # Shuffle the data
        inputImagesShuffle, inputSingleValuesShuffle, outputSingleValuesShuffle = self.ShuffleData(
            inputImages, inputSingleValues, outputSingleValues)
        # Now separate to create batches (i.e., divide the first dimension in 
        # 'number of batches' and 'batch size')
        # The final data attributes are extracted here.
        self.DivideTheDataInBatches(inputImagesShuffle, inputSingleValuesShuffle,
                                    outputSingleValuesShuffle)

        return

        ###########################################################################

    # Functions for defining parameters

    def FindNumberOfImages(self, pathImagesFile):

        imageFiles = os.listdir(pathImagesFile)
        self.numberOfImages = len(imageFiles)

        return

    def FindNumberOfBatches(self):

        # floor, so the last uncomplete batch is thrown out
        self.numberOfBatches = int(np.floor(self.numberOfImages / self.batch_size))

        return

    ###########################################################################
    # Functions for data extraction and reshaping

    def ExtractImagesFromPath(self, pathImagesFile, annotationsHolder):

        # Preparing the space for the input images
        inputImages = torch.zeros(self.numberOfImages, self.image_channels,
                                  self.x_size, self.y_size)

        # # Define transform
        # transform = transforms.Compose([
        #     transforms.Resize((self.x_size, self.y_size)),
        #     transforms.ToTensor(),
        # ])

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 90), expand=True, interpolation=F.InterpolationMode.BILINEAR),
            SquarePad(),
            transforms.Resize((self.x_size, self.y_size)),
            transforms.ToTensor(),
        ])

        # Extracting the images
        countInImageFolder = 0
        # for img in glob.glob(pathImagesFile + "/" + "*.png"):
        for i in range(0, len(glob.glob(pathImagesFile + "/" + "*.png"))):
            img = os.path.join(pathImagesFile, annotationsHolder.config['annotations'][i]['image_name'])

            imgCurr = Image.open(img)
            imgCurrTransformed = transform(imgCurr)

            # Check resize visually
            # cv2.imshow("", imgCurrTransformed.numpy().transpose(1, 2, 0)[:,:,::-1])
            # cv2.waitKey(0)

            inputImages[countInImageFolder, :, :, :] = imgCurrTransformed

            countInImageFolder += 1

        return inputImages

    # Find min and max value of an annotation in the annotations holder
    @staticmethod
    def FindMinAndMaxInAnnotations(annotationsHolder, annotationName):

        lengthOfAnnotations = len(annotationsHolder.config['annotations'])
        minValue = 100000000
        maxValue = 0
        for i in range(lengthOfAnnotations):

            currentAverageDistance = annotationsHolder.config['annotations'][i][annotationName]
            if currentAverageDistance < minValue:
                minValue = currentAverageDistance
            if currentAverageDistance > maxValue:
                maxValue = currentAverageDistance

        return minValue, maxValue

    def CompactMaxsAndMinsValuesOutputs(self):

        # Save the min and max values of output (to denormalize it after calling the CNN)
        minValues = torch.zeros(self.numberOfOutputSingleValues)
        # minValues[0] = self.minWidthTop
        # minValues[1] = self.minWidthBottom
        minValues[0] = self.minMass
        self.minValuesOutput = minValues

        maxValues = torch.zeros(self.numberOfOutputSingleValues)
        # maxValues[0] = self.maxWidthTop
        # maxValues[1] = self.maxWidthBottom
        maxValues[0] = self.maxMass
        self.maxValuesOutput = maxValues

        return

    def ExtractMinAndMaxValuesOfInputsAndOutputs(self, annotationsHolder):

        # Extracting the min and max of average distance and width, to perform normalization

        # Min and max of average distance, width top and width bottom
        self.minRatioWidth, self.maxRatioWidth = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder,
                                                                                                    'aspect ratio width')
        print("Min ratio x: {}".format(self.minRatioWidth))
        print("Max ratio x: {}".format(self.maxRatioWidth))
        
        self.minRatioHeight, self.maxRatioHeight = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder,
                                                                                                    'aspect ratio height')       
        print("Min ratio y: {}".format(self.minRatioHeight))
        print("Max ratio y: {}".format(self.maxRatioHeight))
        
        self.minAverageDistance, self.maxAverageDistance = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder,
                                                                                                    'average distance')
        print("Min dist: {}".format(self.minAverageDistance))
        print("Max dist: {}".format(self.maxAverageDistance))
        # self.minWidthTop, self.maxWidthTop = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder, 'width top')
        # self.minWidthBottom, self.maxWidthBottom = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder,
        #                                                                                     'width bottom')
        self.minMass, self.maxMass = DataExtractor.FindMinAndMaxInAnnotations(annotationsHolder,
                                                                              'mass')
        print("Min mass: {}".format(self.minMass))
        print("Max mass: {}".format(self.maxMass))
        self.CompactMaxsAndMinsValuesOutputs()

        return

    def ExtractAnnotationInputsAndOutputs(self, annotationsHolder):

        # Preparing the space for the input images, the input values and the output prediction
        inputSingleValues = torch.zeros(self.numberOfImages, self.numberOfInputSingleValues)
        outputSingleValues = torch.zeros(self.numberOfImages, self.numberOfOutputSingleValues)

        # Extracting the annotation inputs and outputs (normalizing them, except for the image ratio)
        for i in range(self.numberOfImages):
            
            # Extract aspect ratio, average distance (inputs)
            currentAspectRatioWidth = annotationsHolder.config['annotations'][i]['aspect ratio width']
            currentAspectRatioHeight = annotationsHolder.config['annotations'][i]['aspect ratio height']
            currentAverageDistance = annotationsHolder.config['annotations'][i]['average distance']
            # Extract width width top and bottom (inputs)
            currentWidthTop = annotationsHolder.config['annotations'][i]['width top']
            currentWidthBottom = annotationsHolder.config['annotations'][i]['width bottom']
            currentMass = annotationsHolder.config['annotations'][i]['mass']

            ## CLIP ??
            #currentAverageDistance = min(currentAverageDistance, self.maxAverageDistance)
            #currentMass = min(currentMass, self.maxMass)

            # Normalize the data
            currentAverageRatioWidthNorm = (currentAspectRatioWidth - self.minRatioWidth) / (
                    self.maxRatioWidth - self.minRatioWidth)
            currentAverageRatioHeightNorm = (currentAspectRatioHeight - self.minRatioHeight) / (
                    self.maxRatioHeight - self.minRatioHeight)
            currentAverageDistanceNorm = (currentAverageDistance - self.minAverageDistance) / (
                    self.maxAverageDistance - self.minAverageDistance)
            # currentWidthTopNorm = (currentWidthTop - self.minWidthTop) / (self.maxWidthTop - self.minWidthTop)
            # currentWidthBottomNorm = (currentWidthBottom - self.minWidthBottom) / (
            #         self.maxWidthBottom - self.minWidthBottom)
            currentMassNorm = (currentMass - self.minMass) / (self.maxMass - self.minMass)

            # Put the inputs and outputs in their arrays
            inputSingleValues[i, 0] = currentAverageRatioWidthNorm
            inputSingleValues[i, 1] = currentAverageRatioHeightNorm
            inputSingleValues[i, 2] = currentAverageDistanceNorm

            # outputSingleValues[i, 0] = currentWidthTopNorm
            # outputSingleValues[i, 1] = currentWidthBottomNorm
            outputSingleValues[i, 0] = currentMassNorm

        return inputSingleValues, outputSingleValues

    def ShuffleData(self, inputImages, inputSingleValues, outputSingleValues):

        # First we must define the indices of shuffling
        # indicesOrder = np.arange(self.numberOfImages)  # All indices in order
        #indicesOrder = np.arange(self.numberOfImages // 5)

        indicesOrder = np.arange(self.numberOfImages)
        # We shuffle the above indices
        # newIndices = indicesOrder
        newIndices = np.arange(self.numberOfImages)
        for i in range(len(indicesOrder)):
            pickValue = random.choice(indicesOrder)
            indexPickedValue = np.where(indicesOrder == pickValue)
            indicesOrder = np.delete(indicesOrder, (indexPickedValue[0]), axis=0)
            newIndices[i] = pickValue
            '''
            pickValue = random.choice(indicesOrder)
            indexPickedValue = np.where(indicesOrder == pickValue)
            # indicesOrder = np.delete(indicesOrder, (indexPickedValue[0]), axis=0)
            newIndices[i*5] = pickValue
            newIndices[i*5+1] = pickValue + 1
            newIndices[i * 5 + 2] = pickValue + 2
            newIndices[i * 5 + 3] = pickValue + 3
            newIndices[i * 5 + 4] = pickValue + 4
            indicesOrder = np.delete(indicesOrder, (indexPickedValue[0]), axis=0)
            '''

        # Now we shuffle the data based on the above indices
        inputImagesShuffle = inputImages[newIndices, :, :, :]
        inputSingleValuesShuffle = inputSingleValues[newIndices, :]
        outputSingleValuesShuffle = outputSingleValues[newIndices, :]

        return inputImagesShuffle, inputSingleValuesShuffle, outputSingleValuesShuffle

    def DivideTheDataInBatches(self, inputImagesShuffle, inputSingleValuesShuffle,
                               outputSingleValuesShuffle):

        inputImagesFinal = torch.zeros(self.numberOfBatches, self.batch_size, self.image_channels,
                                       self.x_size, self.y_size)
        inputSingleValuesFinal = torch.zeros(self.numberOfBatches, self.batch_size, self.numberOfInputSingleValues)
        outputSingleValuesFinal = torch.zeros(self.numberOfBatches, self.batch_size, self.numberOfOutputSingleValues)

        beginBatch = 0
        endBatch = self.batch_size
        for i in range(self.numberOfBatches):
            inputImagesFinal[i, :, :, :, :] = inputImagesShuffle[beginBatch:endBatch, :, :, :]
            inputSingleValuesFinal[i, :, :] = inputSingleValuesShuffle[beginBatch:endBatch, :]
            outputSingleValuesFinal[i, :, :] = outputSingleValuesShuffle[beginBatch:endBatch, :]

            beginBatch += self.batch_size
            endBatch += self.batch_size

        self.inputImagesBatched = inputImagesFinal
        self.inputSingleValuesBatched = inputSingleValuesFinal
        self.outputSingleValuesBatched = outputSingleValuesFinal

        return
