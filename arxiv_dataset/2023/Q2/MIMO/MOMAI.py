import numpy as np
import pandas as pd
import os


def read_excel(folder_path):
    dice_list = []
    HD_list = []
    conf_list = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.xlsx'):
            file_path = os.path.join(folder_path, filename)
            print(f'Reading {file_path}...')
            
            xls = pd.ExcelFile(file_path)
            df = xls.parse(xls.sheet_names[0],dtypes=float)
            
            if len(xls.sheet_names) >= 3:
                dice_tmp = xls.parse(xls.sheet_names[0]).to_numpy()
                HD_tmp = xls.parse(xls.sheet_names[1]).to_numpy()
                conf_tmp = xls.parse(xls.sheet_names[2]).to_numpy()
                
                dice_list.append(dice_tmp)
                HD_list.append(HD_tmp)
                conf_list.append(conf_tmp)
            else:
                print(f'File {filename} does not have enough sheets.')
    dice = np.array(dice_list) if dice_list else np.array([])
    HD = np.array(HD_list) if HD_list else np.array([])
    conf = np.array(conf_list) if conf_list else np.array([])
    return dice,HD,conf

def get_percentile(statistics, percentile):
    # Sorts the statistics list in ascending order and assigns it to the 'ordered' variable
    ordered = np.sort(statistics)
    return np.percentile(ordered, percentile)

# Bootstrapping for threshold
def threshold(dice,HD,num):
    """
     Calculates threshold values for dice and Hausdorff distance metrics.

    Args:
        dice (ndarray): Dice coefficient used to calculate threshold.
        HD (ndarray): Hausdorff distance used to calculate threshold.
        num (int): number of organs.

    Returns:
        tuple: A tuple of two lists representing the calculated dice and HD thresholds.

    Raises:
        ValueError: If the number of samples used between dice and HD are not equal.
    """

    # Reshape dice and HD arrays
    temp_1 = np.reshape(dice,(-1,dice.shape[2]))
    temp_1 = temp_1[: , :num]  
    temp_2 = np.reshape(HD,(-1,HD.shape[2]))
    temp_2 = temp_2[: , :num]
    # Initialize dice_threshold and HD_threshold lists
    dice_threshold, HD_threshold = [], []
    # Check that the number of samples is equal between dice and HD
    if temp_1.shape[0] != temp_2.shape[0]:
        raise ValueError("The num of samples used between dice and HD are not equal")
    # Iterate over each organ
    for i in range(temp_1.shape[1]):
        pool_1 = temp_1[ : ,i]; 
        pool_2 = temp_2[ : ,i]; 
        statistics_1, statistics_2 = [], []

        # Generate 100 random samples and calculate means
        for j in range(0,100):
            poolB_1 = np.random.choice(pool_1, 2)
            poolB_2 = np.random.choice(pool_2, len(pool_2)-1)
            stat_1 = np.mean(poolB_1)
            stat_2 = np.mean(poolB_2)
            statistics_1.append(stat_1)
            statistics_2.append(stat_2)

        # Get the median of the means and round to 4 decimal places
        medium_1 = get_percentile(statistics_1, 50) # default is 50,can be changed
        medium_2 = get_percentile(statistics_2, 50)
        dice_threshold.append(round(medium_1, 4))
        HD_threshold.append(round(medium_2, 4))

    print("Dice:",dice_threshold)
    print("HD:",HD_threshold)
    return dice_threshold, HD_threshold


def data_calculating(Dice,HD,conf,threshold_dice,threshold_HD,Num):
    """
    Args:
        Dice (ndarray): Dice coefficient.
        HD (ndarray): Hausdorff distance.
        conf (ndarray): Confidence.
        threshold_dice (list): The threshold of Dice coefficient of each organ.
        threshold_HD (list): The threshold of Hausdorff distance of each organ.
        Num (int): number of organs.
    """

    # Sort the confidence scores in descending order
    index = np.argsort((-conf),axis=0)
    # Initialize 
    tau_all = np.ones(Num)
    id_all  = [[] for _ in range(Num)]
    unqualified = [[] for _ in range(Num)]
    id_num  = 0  
    p_total = 0

    for i in range(Num):
        # Extract scores for each organ
        conf_temp  = conf[ :,i]
        index_temp = index[:,i]
        Dice_temp  = Dice[: ,i]
        HD_temp    = HD[ : , i]

        # Extract scores for each organ
        for id in range(len(index_temp)):
            # Create pools of Dice and HD scores up to the current ID
            pool_1 = Dice_temp[index_temp[:id+1]]; 
            pool_2 = HD_temp[index_temp[:id+1]]; 
            # Initialize lists for collecting statistics
            statistics_1, statistics_2 = [], []

            # Perform bootstrapping to estimate the mean and confidence intervals of the pools
            for j in range(0,100):
                poolB_1 = np.random.choice(pool_1, len(pool_1));    
                poolB_2 = np.random.choice(pool_2, len(pool_2)); 
                stat_1 = np.mean(poolB_1); # can be of other statistics
                stat_2 = np.mean(poolB_2); 
                statistics_1.append(stat_1)
                statistics_2.append(stat_2)

            # Get the lower bounds of the confidence intervals
            lower_1 = get_percentile(statistics_1, 5)
            lower_2 = get_percentile(statistics_2, 95)

            # Check if the current organ meets the performance thresholds
            if lower_1 >= threshold_dice[i] and lower_2 <= threshold_HD[i]:
                tau_all[i] = np.round(conf_temp[index_temp[id]],3)  
                id_all[i].append(index_temp[id])        # qualified organs
            else:
                unqualified[i].append(index_temp[id])   # unqualified organs
        # Update the number of qualified organ IDs
        id_num += len(id_all[i])
    # Calculate the total performance metric for all organs
    p_total = np.round((id_num/(conf.size)),3)
    return p_total


if __name__ == '__main__':
    path = r'./data'     # the path of data
    model_Num = 8                       # the number of model used for testing
    sample_Num = 6                      # the number of smaples
    organ_Num = 13                      # the number of organs 

# The information of organ(BTCV)
    organ = { 0:"spleen",         1:"right kidney",   2:"left kidney",
              3:"gallbladder",    4:"esophagus",      5:"liver",
              6:"stomach",        7:"aorta",          8:"inferior vena cava",
              9:"portal and splenic veins",          10:"pancreas",
              11:"right adrenal gland",              12:"left adrenal gland"}
#The information of model
    model ={ 0:"Attention-Unet",   1:"nnUNet",   2:"SwinUNETR",
             3:"Unet",        4:"UNETR",     5:"V-Net"} 

# data preparing
    Dice,HD_dis,conf = read_excel(path)
    Dice_threshold, HD_threshold = threshold(Dice,HD_dis,organ_Num)
    
# data calculating
    p_list = [0 for i in range(model_Num)]
    for i in range(model_Num):
        conf_temp = conf[i, : , : organ_Num]
        dice_temp = Dice[i, : , : organ_Num]
        hd_temp   = HD_dis[i, : , : organ_Num]
        p_list[i] = data_calculating(dice_temp,hd_temp,conf_temp,Dice_threshold,HD_threshold,organ_Num)
        print("For",model[i],":",p_list[i])
    print("Total:",p_list)
