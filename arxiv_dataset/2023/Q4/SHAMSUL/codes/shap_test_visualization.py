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
from skimage.segmentation import slic
import shap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pylab as pl

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
    
# define a function that depends on a binary mask representing if an image region is hidden
def mask_image(zs, segmentation, image, background=None):
    if background is None:
        background = image.mean((0,1))
    out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
    for i in range(zs.shape[0]):
        out[i,:,:,:] = image
        for j in range(zs.shape[1]):
            if zs[i,j] == 0:
                out[i][segmentation == j,:] = background
    return out
    
def f(z):
    return model.predict(mask_image(z, segments_slic, img_orig, 255))
    
def fill_segmentation(values, segmentation):
    out = np.zeros(segmentation.shape)
    for i in range(len(values)):
        out[segmentation == i] = values[i]
    return out

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
train_data_df = pd.read_csv(os.path.join(output_dir, "train.csv"))
np_names = np.array(class_names)
    
#train_data_list = np.random.choice(len(train_data_df), 1000, replace=False)
#x_train_orig, x_train = create_dataset(train_data_list, train_data_df, image_dimension, output_dir)

test_data_list = np.array([i for i in range(len(test_data_df))])
    
x_test_orig, x_test = create_dataset(test_data_list, test_data_df, image_dimension, output_dir)

header=['file_name']
header.extend(['rank_'+str(i) for i in range(1,15)])
header_line=','.join(header)
with open('shap_result_ranks.csv', 'a') as g:
    g.write(header_line + '\n')  
g.close()

shap_heatmaps_dict = dict()

for test_data_index in range(len(test_data_df)):
    
    img = x_test[test_data_index]
    img_orig = x_test_orig[test_data_index]

    # segment the image so we don't have to explain every pixel
    segments_slic = slic(img, n_segments=50, compactness=30, sigma=3)

    # get the top predictions from the model
    preds = model.predict(np.expand_dims(img.copy(), axis=0))
    top_preds = np.argsort(-preds)
    inds = top_preds[0]

    # use Kernel SHAP to explain the network's predictions
    explainer = shap.KernelExplainer(f, np.zeros((1,50)))
    shap_values = explainer.shap_values(np.ones((1,50)), nsamples=1000) # runs VGG16 1000 times
    
    original_image_path='original_images/'+test_data_df['Path'][test_data_index].replace("/","_")+'.png'
    
    #plot_original_image(img_orig, original_image_path)
    
    ranked_class_list=[test_data_df['Path'][test_data_index]]
    
    ranked_class_list.extend(np.take(np_names, inds))
    ranked_class_list_line=','.join(ranked_class_list)
    with open('shap_result_ranks.csv', 'a') as g:
        g.write(ranked_class_list_line + '\n')  
    g.close()
    
    shap_class_heatmaps_for_one_instance_dict = dict()

    for class_index in range(len(inds)):
    
        class_name=np_names[inds[class_index]]
    
        heatmap = fill_segmentation(shap_values[inds[class_index]][0], segments_slic)
        
        shap_class_heatmaps_for_one_instance_dict[class_name]=heatmap
    
        heatmap_with_image_save_path = 'results/SHAP/heatmap_and_images/'+test_data_df['Path'][test_data_index].replace("/","_")+'_heatmap_with_image_using_shap_method_class_name_'+class_name+'.png'
    
        plot_heatmap_with_image(heatmap, img_orig, 'inferno', 0.8, heatmap_with_image_save_path)
    
        heatmap_only_save_path = 'results/SHAP/heatmaps/'+test_data_df['Path'][test_data_index].replace("/","_")+'_heatmap_only_using_shap_method_class_name_'+class_name+'.png'
    
        plot_heatmap_only(heatmap, 'inferno', heatmap_only_save_path)
        
    print(str(test_data_index+1)+' test images completed.')
    
    shap_heatmaps_dict[test_data_index]=shap_class_heatmaps_for_one_instance_dict
            
    #break
    
with open("results/SHAP/shap_heatmaps_dict", "wb") as fp:
    pickle.dump(shap_heatmaps_dict, fp)
    
