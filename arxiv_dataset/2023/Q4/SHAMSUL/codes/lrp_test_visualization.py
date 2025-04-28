from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import classification_report

from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import csv

import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score
from utility import get_sample_counts
import pandas as pd

import innvestigate
import innvestigate.utils

from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

def create_dataset(data_list, df, image_dimension, output_dir):
    
    original=[]
    transformed=[]
    for i in data_list:
        
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
    
        image_name= output_dir+'/'+df['Path'][i]
        image = Image.open(image_name)
        image_array = np.asarray(image.convert("RGB"))
        image_array = image_array / 255.
        image_array = resize(image_array, (image_dimension, image_dimension))
        image_array = image_array[None, :, :, :]
        original.append(image_array)
        image_array = (image_array - imagenet_mean) / imagenet_std #for testing has to open
        transformed.append(image_array)
    
    original_images=np.vstack(original)
    transformed_images=np.vstack(transformed)
    
    return original_images, transformed_images

def transparent_cmap(cmap, intensity, N=255):
    
    #Copy colormap and set alpha values
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, intensity, N+4)
    return mycmap

def plot_heatmap_only(heatmap, cmap, save_path):
    
    plt.imshow(heatmap, cmap=cmap, clim=(-1, 1))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def plot_original_image(image, save_path):
    
    fig, ax = plt.subplots(1, 1)
    #ax.axis('off')
    ax.imshow(image)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    #plt.show()
    plt.close()

def main():
    # parser config
    config_file = "config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")

    # train config
    image_dimension = cp["TRAIN"].getint("image_dimension")

    # test config
    batch_size = cp["TEST"].getint("batch_size")
    test_steps = cp["TEST"].get("test_steps")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    # parse weights file path
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    weights_path = os.path.join(output_dir, output_weights_name)
    best_weights_path = os.path.join(output_dir, "best_{output_weights_name}")

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "valid", class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError("""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    
    if use_best_weights:
        model_weights_path = best_weights_path
    else:
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path, input_shape=(image_dimension, image_dimension, 3))
        
    test_data_df = pd.read_csv(os.path.join(output_dir, "valid.csv"))
    test_data_list = np.array([i for i in range(len(test_data_df))])
    
    original_images, transformed_images = create_dataset(test_data_list, test_data_df, image_dimension, output_dir)

    np_names = np.array(class_names)
    y_hat = model.predict(transformed_images)
    y=test_data_df[np_names].as_matrix()
    
    
    
    lrp_methods = ["lrp.z", "lrp.alpha_2_beta_1", "lrp.epsilon", "deep_taylor"]
            
    # Strip softmax layer
    #model = innvestigate.utils.model_wo_softmax(model)
    #print("removed softmax")
    
    header=['file_name']
    header.extend(['rank_'+str(i) for i in range(1,len(np_names)+1)])
    header_line=','.join(header)
    with open('lrp_result_ranks.csv', 'a') as f:
        f.write(header_line + '\n')  
    f.close()
    
    lrp_heatmaps_dict = dict()
    
    for lrp_method in lrp_methods:
        
        lrp_heatmap_method_dict = dict()
        
        # Create analyzer
        if lrp_method=="lrp.epsilon":
            optional_method = {'epsilon': 1}
        else:
            optional_method = {}
            
        '''
        {'input': <class 'innvestigate.analyzer.misc.Input'>, 
            'random': <class 'innvestigate.analyzer.misc.Random'>, 
            'gradient': <class 'innvestigate.analyzer.gradient_based.Gradient'>, 
            'gradient.baseline': <class 'innvestigate.analyzer.gradient_based.BaselineGradient'>, 
            'input_t_gradient': <class 'innvestigate.analyzer.gradient_based.InputTimesGradient'>, 
            'deconvnet': <class 'innvestigate.analyzer.gradient_based.Deconvnet'>, 
            'guided_backprop': <class 'innvestigate.analyzer.gradient_based.GuidedBackprop'>, 
            'integrated_gradients': <class 'innvestigate.analyzer.gradient_based.IntegratedGradients'>, 
            'smoothgrad': <class 'innvestigate.analyzer.gradient_based.SmoothGrad'>, 
            'lrp': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRP'>, 
            'lrp.z': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZ'>, 
            'lrp.z_IB': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZIgnoreBias'>, 
            'lrp.epsilon': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilon'>, 
            'lrp.epsilon_IB': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPEpsilonIgnoreBias'>, 
            'lrp.w_square': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPWSquare'>, 
            'lrp.flat': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPFlat'>, 
            'lrp.alpha_beta': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlphaBeta'>, 
            'lrp.alpha_2_beta_1': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha2Beta1'>, 
            'lrp.alpha_2_beta_1_IB': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha2Beta1IgnoreBias'>, 
            'lrp.alpha_1_beta_0': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0'>, 
            'lrp.alpha_1_beta_0_IB': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0IgnoreBias'>, 
            'lrp.z_plus': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZPlus'>, 
            'lrp.z_plus_fast': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPZPlusFast'>, 
            'lrp.sequential_preset_a': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetA'>, 
            'lrp.sequential_preset_b': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetB'>, 
            'lrp.sequential_preset_a_flat': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetAFlat'>, 
            'lrp.sequential_preset_b_flat': <class 'innvestigate.analyzer.relevance_based.relevance_analyzer.LRPSequentialPresetBFlat'>, 
            'deep_taylor': <class 'innvestigate.analyzer.deeptaylor.DeepTaylor'>, 
            'deep_taylor.bounded': <class 'innvestigate.analyzer.deeptaylor.BoundedDeepTaylor'>, 
            'deep_lift.wrapper': <class 'innvestigate.analyzer.deeplift.DeepLIFTWrapper'>, 
            'pattern.net': <class 'innvestigate.analyzer.pattern_based.PatternNet'>, 
            'pattern.attribution': <class 'innvestigate.analyzer.pattern_based.PatternAttribution'>}
        '''
            
        analyzer = innvestigate.create_analyzer(lrp_method, model, neuron_selection_mode="index", **optional_method)
            
        for test_data_index in range(len(test_data_df)):
            
            lrp_class_heatmaps_for_one_instance_dict = dict()
            
            ind = y_hat[test_data_index].argsort()[::-1]
            
            if lrp_method=="lrp.z":
            
                original_image_path='original_images/'+test_data_df['Path'][test_data_index].replace("/","_")+'.png'
    
                #plot_original_image(transformed_images[test_data_index], original_image_path)
                
                ranked_class_list=[test_data_df['Path'][test_data_index]]
                ranked_class_list.extend(np.take(np_names, ind))
                ranked_class_list_line=','.join(ranked_class_list)
                with open('lrp_result_ranks.csv', 'a') as f:
                    f.write(ranked_class_list_line + '\n')  
                f.close()
                
            for class_index in ind:
            
                class_name = np_names[class_index]
                
                # Apply analyzer w.r.t. maximum activated output-neuron
                a = analyzer.analyze(np.expand_dims(transformed_images[test_data_index].copy(), axis=0), neuron_selection=class_index)

                # Aggregate along color channels and normalize to [-1, 1]
                a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
                a /= np.max(np.abs(a))

                
                
                heatmap = a[0]
                
                lrp_class_heatmaps_for_one_instance_dict[class_name]=heatmap
            
                #heatmap_only_save_path = 'results/LRP/heatmaps/'+test_data_df['Path'][test_data_index].replace("/","_")+'_heatmap_only_using_'+lrp_method+'_method_class_name_'+class_name+'.png'
    
                #plot_heatmap_only(heatmap, 'seismic', heatmap_only_save_path)
                
            lrp_heatmap_method_dict[test_data_index]=lrp_class_heatmaps_for_one_instance_dict
            
        lrp_heatmaps_dict[lrp_method]=lrp_heatmap_method_dict
        
        if lrp_method=="deep_taylor":
            print(str(test_data_index+1)+' test images completed.')
            #break
        
    with open("results/LRP/lrp_heatmaps_dict", "wb") as fp:
        pickle.dump(lrp_heatmaps_dict, fp)
    
                    
if __name__ == "__main__":
    main()

