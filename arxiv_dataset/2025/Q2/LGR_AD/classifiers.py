#In this file, several models based on text-to-image generation might be included. 

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, DenseNet121, MobileNetV2,Xception
from tensorflow.keras.models import load_model
import pandas as pd

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess

from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
test_dir = '/home/nassim/Desktop/GoD/tiny-imagenet-200/val'

# Initialize ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)

# Load test images from a directory
test_generator = datagen.flow_from_directory(test_dir,
                                             target_size=(224, 224),
                                             batch_size=32
                                             )




# Define pre-trained models
models = {
    'ResNet50': '',
    'VGG16': '',
    'Xception': '',
    'DenseNet121': '',
    'MobileNetV2': ''
}




with tf.device('/GPU:0'):
    
    # Make predictions using each model
    for name in models:
            results_df = pd.DataFrame()

            if name == 'ResNet50':
                with tf.device('/GPU:0'):

                    model,preprocess=(ResNet50(weights='imagenet'),resnet_preprocess)
            if name == 'VGG16':
    
                    model,preprocess=(VGG16(weights='imagenet'),vgg_preprocess)
            if name == 'Xception':
    
                    model,preprocess=(Xception(weights='imagenet'),xception_preprocess)
            if name == 'DenseNet121':
    
                    model,preprocess=(DenseNet121(weights='imagenet'),densenet_preprocess)
            if name == 'MobileNetV2':
                with tf.device('/GPU:0'):
                    model,preprocess=(MobileNetV2(weights='imagenet'),mobilenet_preprocess)
            print(f"Making predictions with {name}...")
        

            predictions = model.predict(test_generator,use_multiprocessing=True)
        
            # Get the index of the highest score for each prediction
            predicted_labels = np.argmax(predictions, axis=1)
            # Get filenames (remove directory path from filenames)
            filenames = [name.split('/')[-1] for name in test_generator.filenames]
            
            # Store the results in a temporary DataFrame
            temp_df = pd.DataFrame({
                'Filename': filenames,
                f'{name}_Prediction': predicted_labels
            })
            
            # If results_df is empty, copy temp_df to results_df
            if results_df.empty:
                results_df = temp_df
            else:
                # Merge results_df and temp_df on 'Filename'
                results_df = pd.merge(results_df, temp_df, on='Filename')
            results_df.to_csv(f'model_predictions_{name}.csv', index=False)

            print(f"Predicted labels for first 10 images: {predicted_labels[:10]}")
            print("------")
