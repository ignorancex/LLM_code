################################################################################################### 
#                               Load modules
###################################################################################################
import random, os, json, cv2, shutil
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_style("whitegrid", {'axes.grid' : False})

from skimage.metrics import structural_similarity as SSIM





################################################################################################### 
#                               Helper functions
###################################################################################################
def makedir(path, delete_folder = False):
    if os.path.exists(path):
        if delete_folder:
            shutil.rmtree(path)
    else:            
        os.makedirs(path)


def to_json(path, filename, list_of_dicts):
    with open(path + filename + '.json', 'w', encoding = 'utf-8') as json_file:
        json.dump(list_of_dicts, json_file, indent = 4)



def compute_ssim(image_1, image_2):
    ssim_value = SSIM(image_1, image_2, multichannel = True)
    return ssim_value

def mean_ssim(X_test, X_decode):
    results = []
    for i in range(X_test.shape[0]):
        current_result = compute_ssim(X_test[i, ...], X_decode[i, ...])
        results.append(current_result)
    results = np.mean(np.array(results))
    return results

    
def load_diff_dim_images_from_folder(folder, random_state, nr = -1, resize_val = None):
    random.seed(random_state)
    images = []
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        
        if img is not None:
            if img.ndim > 2:
                if resize_val is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, dsize = (resize_val, resize_val), interpolation = cv2.INTER_CUBIC)
                images.append(img)
                
    if nr > 0:
        images = random.sample(images, nr)
    return np.asarray(images)





def load_images_from_subfolders(subfolders, random_state, nr, resize_val = None):
    random.seed(random_state)
    all_images = []
    total_count = 0
    
    for path in subfolders:
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            
            if img is not None:
                if resize_val is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, dsize = (resize_val, resize_val), interpolation = cv2.INTER_CUBIC)
                all_images.append(img)
                total_count += 1

    if nr < total_count :          
        all_images = random.sample(all_images, nr)
    return np.asarray(all_images)




def load_train_test_images(opts):
    
    if opts["test_path"] is not None:
        X_train = load_diff_dim_images_from_folder(opts["train_path"], opts["random_state"], opts["N_train"], opts["resize_val"])
        X_test = load_diff_dim_images_from_folder(opts["test_path"], opts["random_state"], opts["N_test"], opts["resize_val"])
    else:
        X_train = load_diff_dim_images_from_folder(opts["train_path"], opts["random_state"], 
                                                            opts["N_train"] + opts["N_test"], opts["resize_val"])
        X_test = None

    width = X_train.shape[1]
    height = X_train.shape[2]
    channels = X_train.shape[3]

    return X_train, X_test, width, height, channels





def load_images(opts):

    subfolders = [x[0] for count, x in enumerate(os.walk(opts["train_path"])) if count != 0]
    X_train = load_images_from_subfolders(subfolders, opts["random_state"], opts["N_train"] + opts["N_test"], opts["resize_val"])
        
    width = X_train.shape[1]
    height = X_train.shape[2]
    channels = X_train.shape[3]

    return X_train, width, height, channels




def reshape_images(X_train, X_test, channels):        
    if channels == 1:
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
    else:
        X_train = X_train.reshape((X_train.shape[0], -1, channels))
        X_test = X_test.reshape((X_test.shape[0], -1, channels))

    return X_train, X_test



def to_numpy(X):
    if X is None:
        return None
    else:
        if isinstance(X, pd.DataFrame):
            return X.values
        else: 
            return X


def convert_data(X_train, X_test, y_train, y_test):
    X_train = to_numpy(X_train)
    X_test = to_numpy(X_test)
    y_train = to_numpy(y_train)
    y_test = to_numpy(y_test)
    return X_train, X_test, y_train, y_test




################################################################################################### 
#                               Plot functions
###################################################################################################
def plot_sample(X_test, w, h, c, color = False):
    all_idx = X_test.shape[0]
    idx = list(range( min(10, all_idx) ))
    
    stacked_array_1 = np.hstack(([ X_test[i, :].reshape(w, h, c) for i in idx[:5] ]))
    stacked_array_2 = np.hstack(([ X_test[i, :].reshape(w, h, c) for i in idx[5:] ]))
    stacked_array = np.vstack((stacked_array_1, stacked_array_2))
    stacked_array = stacked_array.astype(np.uint8)
    
    print(stacked_array_1.shape, stacked_array.shape)
    cmap = None    
    if not color:
        cmap = plt.cm.Greys_r

    plt.figure(figsize = (5, 5))
    plt.imshow(stacked_array, cmap = cmap)
    plt.tight_layout()
    plt.show()





def plot_results_tabular(datas, feat_numbers, limits = None, path = None, save = False):
    X_decoded, X_test = datas

    plt.figure()
    plt.tick_params(axis = 'both', which = 'minor', labelsize = 12)

    count = 0
    for _ in range(2):
        for _ in range(2):
            plt.subplot(2, 2, count + 1)
            plt.plot(X_test[:, feat_numbers[count]], 'b*-', label = 'X_test [' + str(feat_numbers[count]) + ']', 
                                                                                    linewidth = 2.5, markersize = 10)
            plt.plot(X_decoded[:, feat_numbers[count]], 'ro-', label = 'X_decoded [' + str(feat_numbers[count]) + ']', 
                                                                                    linewidth = 2.5, markersize = 5)
            plt.grid('on')
            
            if limits is not None:
                plt.xlim(limits[count])

            plt.legend()

            count += 1

    plt.tight_layout()
    
    if save:
        plt.savefig(path + ".jpg", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".png", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".eps", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".pdf", dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()







def plot_results_images(datas, channels, path = None, save = False):
    
    cmap = None    
    if channels == 1:
        cmap = plt.cm.Greys_r

    plt.figure()
    plt.imshow(datas, cmap = cmap)
    
    plt.axis("off")
    plt.tight_layout()
    
    if save:
        plt.savefig(path + ".jpg", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".png", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".eps", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".pdf", dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()




def make_boxplot_estimators(results, cols, path = None, save = False):
    sns.set(font_scale = 1.5)

    sns.catplot(data = results, x = cols[0], y = cols[1], hue = cols[2], kind = "box")
    plt.grid('on')

    if save:
        plt.savefig(path + ".jpg", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".png", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".eps", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".pdf", dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()


def make_barplot_time(results, cols, path = None, save = False):
    sns.set(font_scale = 2)

    if len(cols) < 3:
        sns.catplot(data = results, x = cols[0], y = cols[1], kind = "bar")        
    else:
        sns.catplot(data = results, x = cols[0], y = cols[1], hue = cols[2], kind = "bar")

    plt.grid('on')

    if save:
        plt.savefig(path + ".jpg", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".png", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".eps", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".pdf", dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()




def plot_results_tabular_misc(n_test_decode, mean_results, std_results, title_list, path = None, save = False):
    xlabel, ylabel = title_list
    mean_results, std_results = np.array(mean_results), np.array(std_results)

    plt.figure()
    plt.rcParams['font.size'] = '20'

    plt.plot(n_test_decode, mean_results, 'b*-')
    plt.fill_between(n_test_decode, mean_results - std_results, mean_results + std_results, color = 'b', alpha = 0.2)

    plt.grid('on')
    plt.tight_layout()

    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)

    if save:
        plt.savefig(path + ".jpg", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".png", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".eps", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".pdf", dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()




def basic_plot_results(x, y, title_list, path = None, save = False):
    xlabel, ylabel = title_list
    x, y = np.array(x), np.array(y)

    plt.figure()
    plt.rcParams['font.size'] = '20'

    plt.plot(x, y, 's-', markerfacecolor = "green", markeredgecolor = "black", markersize = 7)
 
    plt.grid('on')
    plt.tight_layout()

    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    
    if save:
        plt.savefig(path + ".jpg", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".png", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".eps", dpi = 300, bbox_inches = 'tight')
        plt.savefig(path + ".pdf", dpi = 300, bbox_inches = 'tight')
    else:
        plt.show()

