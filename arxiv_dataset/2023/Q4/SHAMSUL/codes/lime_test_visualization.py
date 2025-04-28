import numpy as np
import os
from configparser import ConfigParser
from generator import AugmentedImageSequence
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score, average_precision_score
from utility import get_sample_counts
import pickle
import pandas as pd
from PIL import Image
from skimage.transform import resize
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

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

def plot_heatmap_with_image_plus_original_image(heatmap, image, cmap, intensity, save_path):
    
    #just a test
    w, h = heatmap.shape
    y, x = np.mgrid[0:h, 0:w]
    
    mycmap = transparent_cmap(plt.cm.get_cmap(cmap), intensity)
    
    
    fig = plt.figure()

    ax1 = fig.add_subplot(1,2,1, aspect = "equal")
    ax2 = fig.add_subplot(1,2,2, aspect = "equal")

    ax1.axis('off')
    ax1.imshow(image)
    
    ax2.imshow(image)
    cb = ax2.contourf(x, y, heatmap, 4, cmap=mycmap)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)

    #Create and remove the colorbar for the first subplot
    cbar1 = fig.colorbar(cb, cax = cax1)
    fig.delaxes(fig.axes[2])

    #Create second colorbar
    cbar2 = fig.colorbar(cb, cax = cax2)

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    #plt.show()
    plt.close()

def plot_heatmap_with_image(heatmap, image, cmap, intensity, save_path):
    
    w, h = heatmap.shape
    y, x = np.mgrid[0:h, 0:w]
    
    mycmap = transparent_cmap(plt.cm.get_cmap(cmap), intensity)
    
    fig, ax = plt.subplots(1, 1)
        
    ax.imshow(image)
    cb = ax.contourf(x, y, heatmap, 4, cmap=mycmap)
    plt.colorbar(cb)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
        
    
def plot_heatmap_only(heatmap, cmap, save_path):
    
    plt.imshow(heatmap, cmap = cmap, vmin  = -heatmap.max(), vmax = heatmap.max())
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
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    np_names = np.array(class_names)
    
    test_data_list = np.array([i for i in range(len(test_data_df))])
    
    original_images, transformed_images = create_dataset(test_data_list, test_data_df, image_dimension, output_dir)
    
    explainer = lime_image.LimeImageExplainer()
    
    header=['file_name']
    header.extend(['rank_'+str(i) for i in range(1,len(np_names)+1)])
    header_line=','.join(header)
    with open('lime_result_ranks.csv', 'a') as f:
        f.write(header_line + '\n')  
    f.close()
    
    lime_heatmaps_dict = dict()
    
    for test_data_index in range(len(test_data_df)):
        
        # Hide color is the color for a superpixel turned OFF. Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels
        explanation = explainer.explain_instance(transformed_images[test_data_index], model.predict, top_labels=len(np_names), hide_color=0, num_samples=1000)
        
        all_ind = explanation.top_labels
        
        ranked_class_list=[test_data_df['Path'][test_data_index]]
        ranked_class_list.extend(np.take(np_names, all_ind))
        ranked_class_list_line=','.join(ranked_class_list)
        with open('lime_result_ranks.csv', 'a') as f:
            f.write(ranked_class_list_line + '\n')  
        f.close()
        
        original_image_path='original_images/'+test_data_df['Path'][test_data_index].replace("/","_")+'.png'
    
        #plot_original_image(original_images[test_data_index], original_image_path)
        
        lime_class_heatmaps_for_one_instance_dict = dict()
        
        for class_index in all_ind:
            
            class_name = np_names[class_index]
            
            #Map each explanation weight to the corresponding superpixel
            dict_heatmap = dict(explanation.local_exp[class_index])
            heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
            lime_class_heatmaps_for_one_instance_dict[class_name]=heatmap
            
            heatmap_with_image_save_path = 'results/LIME/heatmap_and_images/'+test_data_df['Path'][test_data_index].replace("/","_")+'_heatmap_with_image_using_lime_method_class_name_'+class_name+'.png'
            
            plot_heatmap_with_image(heatmap, original_images[test_data_index], 'inferno', 0.8, heatmap_with_image_save_path)
            
            heatmap_only_save_path = 'results/LIME/heatmaps/'+test_data_df['Path'][test_data_index].replace("/","_")+'_heatmap_only_using_lime_method_class_name_'+class_name+'.png'
            
            plot_heatmap_only(heatmap, 'inferno', heatmap_only_save_path)
            
        print(str(test_data_index+1)+' test images completed.')
        
        lime_heatmaps_dict[test_data_index]=lime_class_heatmaps_for_one_instance_dict
            
        #break
        
    with open("results/LIME/lime_heatmaps_dict", "wb") as fp:
        pickle.dump(lime_heatmaps_dict, fp)
    


if __name__ == "__main__":
    main()
