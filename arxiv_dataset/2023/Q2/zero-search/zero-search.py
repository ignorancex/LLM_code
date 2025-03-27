from tkinter import *
import tkinter as tk
from tkinter import filedialog, Text
from tkinter.ttk import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import time
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import requests
from PIL import Image, ImageTk
from keras.preprocessing import image
from PIL import Image
from random import sample
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess   # CHANGE FOR PRETRAINED
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_preprocess
from tensorflow.keras.models import Model
from pathlib import Path
import matplotlib.pyplot as plt
Image.LOAD_TRUNCATED_IMAGES = True
import requests
from io import BytesIO
import cv2
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import glob, random

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

class FeatureExtractor:
    
    # Constructor
    def __init__(self, arch='VGG'): # CHANGE FOR PRETRAINED
        
        self.arch = arch
        
        # Using VGG -16 as the architecture with ImageNet weights
        if self.arch == 'VGG' :  # CHANGE FOR PRETRAINED
            base_model = VGG16(weights = 'imagenet')
            self.model = Model(inputs = base_model.input, outputs = base_model.get_layer('fc1').output)
        
        # Using the ResNet 50 as the architecture with ImageNet weights
        elif self.arch == 'ResNet':
            base_model = ResNet50(weights = 'imagenet')
            self.model = Model(inputs = base_model.input, outputs = base_model.get_layer('avg_pool').output)
        
        # Using the Xception as the architecture with ImageNet weights
        elif self.arch == 'Xception':
            base_model = Xception(weights = 'imagenet')
            self.model = Model(inputs = base_model.input, outputs = base_model.get_layer('avg_pool').output)
            
    # Method to extract image features
    def extract_features(self, img):
        
        # The VGG 16 & ResNet 50 model has images of 224,244 as input while the Xception has 299, 299
        if self.arch == 'VGG' or self.arch == 'ResNet': # CHANGE FOR PRETRAINED
            img = img.resize((224, 224))
        elif self.arch == 'Xception':
            img = img.resize((299, 299))
        
        # Convert the image channels from to RGB
        img = img.convert('RGB')
        
        # Convert into array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        if self.arch == 'VGG':
            # Proprocess the input as per vgg 16
            x = vgg_preprocess(x)
            
        elif self.arch == 'ResNet':
            # Proprocess the input as per ResNet 50
            x = resnet_preprocess(x)
            
        elif self.arch == 'Xception':
            # Proprocess the input as per ResNet 50
            x = xception_preprocess(x)
        
        
        # Extract the features
        features = self.model.predict(x) 
        
        # Scale the features
        features = features / np.linalg.norm(features)
        
        return features

            # Method to extract image features
    def extract_query_features(self, img):
        
        # Convert into array
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        if self.arch == 'VGG': # CHANGE FOR PRETRAINED
            # Proprocess the input as per vgg 16
            x = vgg_preprocess(x)
            
        elif self.arch == 'ResNet':
            # Proprocess the input as per ResNet 50
            x = resnet_preprocess(x)
            
        elif self.arch == 'Xception':
            # Proprocess the input as per ResNet 50
            x = xception_preprocess(x)
        
        
        # Extract the features
        features = self.model.predict(x) 
        
        # Scale the features
        features = features / np.linalg.norm(features)
        
        return features

def owl(image, prompt, threshold=0.1):
    # image = Image.open(requests.get(url, stream=True).raw)
    texts = [[prompt]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]])
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process(outputs=outputs, target_sizes=target_sizes)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    # score_threshold = 0.1
    best_box = []
    best_conf = -100000
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        if score >= threshold:
            if score > best_conf:
                best_box = box
                best_conf = score.item()
    print(f"Detected {text[label]} with confidence {round(best_conf, 3)} at location {best_box}")
    # image[int(best_box[1]):int(best_box[1]+best_box[3]), int(best_box[0]):int(best_box[0]+best_box[2])]
    return best_box

def choose_file():
    global modelImages, file_path, image_features_vgg, vgg_feature_extractor
    file_path = filedialog.askdirectory()
    print(file_path)
    modelImages = os.listdir(file_path)
    print(modelImages)
    
    vgg_feature_extractor = FeatureExtractor(arch='VGG')
    count = 0
    # dictionary to store the features and index of the image
    image_features_vgg = {}
    for i, img_path in enumerate(modelImages):
        
        # Extract features and store in a dictionary
        try:
            img = image.load_img(os.path.join(file_path,img_path))
            feature = vgg_feature_extractor.extract_features(img)
            image_features_vgg[img_path] = feature
            print(count)
        except:
            print(img_path)
        count += 1
    # count = 0
    # img_list = []
    # image_features_vgg = {}
    # for img in modelImages:
    #     img_arr = cv2.imread(os.path.join(file_path,modelImages))
    #     img_list.append(img_arr)
        # feature = vgg_feature_extractor.extract_features(img_arr)
        # image_features_vgg[idx] = feature
        # # call feature extractor 


        # cv2.resize(img_arr,(240,240))
        # cv2.imshow(f'Image_{count}',img_arr)
        # print(count)
        # count = count + 1
    # print(f'The Count is',count)   

def input():
    global search, thresh
    search = inputtxt.get(1.0, "end-1c")
    thresh = s1.get()
    print(search, thresh)
    
def get_similar():
    # threshold = 0.0105
    global img_plot 
    x = random.randint(0, len(modelImages))
    print(modelImages[x])
    test = image.load_img(os.path.join(file_path,modelImages[x]))
    output = owl(test, search, thresh)
    c = 0
    while (len(output)==0) and (c<10):
        c+=1
        print("we got here")
        x = random.randint(0, len(modelImages))
        print(modelImages[x])
        test = image.load_img(os.path.join(file_path,modelImages[x]))
        
        output = owl(test, search, thresh)

    output = [0 if i < 0 else i for i in output]

    test_img = cv2.imread(os.path.join(file_path,modelImages[x]))
    roi_cropped = test_img[int(output[1]):int(output[1]+output[3]), int(output[0]):int(output[0]+output[2])]
    roi_cropped = roi_cropped[...,::-1].astype(np.float32)
    roi_cropped = cv2.resize(roi_cropped, ((224,224)))  # CHANGE FOR PRETRAINED

    queryFeatures_Vgg = vgg_feature_extractor.extract_query_features(roi_cropped)

    similarity_images_vgg = {}
    for idx, feat in image_features_vgg.items():
        
        # Compute the similarity using Euclidean Distance
        similarity_images_vgg[idx] = np.sum((queryFeatures_Vgg - feat)**2) ** 0.5
        
    similarity_vgg_sorted = sorted(similarity_images_vgg.items(), key = lambda x : x[1], reverse=False)
    top_10_indexes_vgg = [idx for idx, _ in similarity_vgg_sorted][ : 10]
    # count = 0

    # _, axs = plt.subplots(5, 2, figsize=(12, 12))
    # axs = axs.flatten()
    # c = 1
    # for i, ax in zip(top_10_indexes_vgg, axs):
    #     img = cv2.imread(os.path.join(file_path,i))
    #     ax.imshow(img)
    # plt.show()
    # for i in top_10_indexes_vgg:
    #     img = cv2.imread(os.path.join(file_path,i))
    #     cv2.imshow(str(i), img)
    #     cv2.waitKey(10000)

    # _, axs = plt.subplots(5, 2, figsize=(12, 12))
    # axs = axs.flatten()
    # c = 1
    # for i, ax in zip(top_10_indexes_vgg, axs):
    #     img = cv2.imread(os.path.join(file_path,i))
    #     # img = cv2.resize(img, (450, 625))
    #     ax.imshow(img)
    # plt.savefig('plot.png')
    # plt.show()

    height, width = 450, 625

    # Create an empty numpy array to hold the collage image
    collage = np.zeros((5*height, 2*width, 3), dtype=np.uint8)

    # Load each image and resize it to match the height and width of the individual images
    images = [cv2.imread(os.path.join(file_path,path)) for path in top_10_indexes_vgg]
    resized_images = [cv2.resize(img, (width, height)) for img in images]

    # Iterate over the resized images and place them in the empty numpy array
    for i in range(5):
        for j in range(2):
            index = i*2 + j
            if index < len(resized_images):
                collage[i*height:(i+1)*height, j*width:(j+1)*width, :] = resized_images[index]

    # cv2.imshow("implot", collage)
    # collage = cv2.resize(collage, (520,520))
    color_coverted = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    img_plot = ImageTk.PhotoImage(pil_image.resize((475,500)))
    my_label = tk.Label(frame,image=img_plot)
    my_label.grid(row=0,column=0)

    # img_plot = ImageTk.PhotoImage(Image.open("plot.png").resize((520,520)))
    # my_label = tk.Label(frame,image=img_plot)
    # my_label.grid(row=0,column=0)


    # cv2.imshow('im_plot',img_plot)
    # cv2.imshow('img_show',img_plot )
    print('got it')


# def forward(image_number):
#     global my_label
#     global button_forward
#     global button_backward

#     # my_label = Label(frame, image=img_list[image_number-1])
#     cv2.imshow(f'Image{image_number}',img_arr[image_number-1]) 
#     button_forward = Button(canvas,text='Next',command=lambda: forward(image_number+1))
#     button_backward = Button(canvas,text='Previous',command=lambda: back(image_number-1))

#     if image_number == len(images):
#         button_forward = Button(canvas,text='Next',state= DISABLED)


# def back(image_number):
#     global my_label
#     global button_forward
#     global button_backward

#     my_label = Label(frame, image=img_arr[image_number-1]) 
#     my_label.image = images[image_number-1]
#     button_forward = Button(canvas,text='Next',command=lambda: forward(image_number+1))
#     button_backward = Button(canvas,text='Previous',command=lambda: back(image_number-1))


def exit_app():
    root.quit()

root = tk.Tk()

canvas = tk.Canvas(root,height = 800,width = 800,bg = '#323333') 
canvas.pack()

frame = tk.Frame(root,bg = 'white')
frame.place(relx = 0.2,rely = 0.05,relwidth =0.598,relheight =0.625)

openfile = tk.Button(canvas,text = 'Select a Folder',fg = 'blue',padx = 10,pady = 5, command=choose_file)
openfile.place(x = 100 , y = 600)

inputtxt = tk.Text(canvas,height = 1.5,width = 14)
inputtxt.place(x = 250 , y = 600)

printButton = tk.Button(canvas,text = "Enter",fg = 'green',padx = 10,pady = 5, command = input)
printButton.place(x = 530 , y = 600)

# button_forward = tk.Button(canvas, text = "Next",fg = 'blue',padx = 10,pady = 5, command = lambda: forward(2))
# button_forward.place(x = 700 , y = 400)

# button_backward = tk.Button(canvas, text = "Previous",fg = 'blue',padx = 10,pady = 5, command = back, state= DISABLED)
# button_backward.place(x = 70 , y = 400)

s1 = Scale(canvas, from_ = 0, to = 0.1, orient = HORIZONTAL) 
s1.place(x = 400 , y = 600)

get_similar_button = tk.Button(canvas,text = 'Get Similar Images',fg = 'brown',padx = 10,pady = 5, command = get_similar)
get_similar_button.place(x = 600, y = 600)

exit_btn = tk.Button(canvas,text = 'Exit',fg = 'red',padx = 10,pady = 5, command = exit_app)
exit_btn.place(x = 700, y = 700)


root.mainloop()