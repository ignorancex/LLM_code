import cv2
from tqdm import tqdm
import csv
#%%
file_train = "../VOC2007/2007_train.txt"
val_set = ['000024', '000057']
z_offsets = ['z0', 'z400', 'z-400', 'z800', 'z-800', 'z1200', 'z-1200', 'z1600', 'z-1600', 'z2000', 'z-2000']

#%%
def prediction_patch(w=80, z_expand=False):
    """    
    Arguments:
    w -- window size for patches
    z_expand -- if expand 11 patches along z-axis for each position
    
    Returns:
    total_num -- total number of predicted nucleus patches
    """       
    total_num = 0
    
    if z_expand == False:
        for img_name in tqdm(val_set):
            img = cv2.imread('../VOC2007/JPEGImages/' + img_name + '.jpg')
            with open('../FCRN/Results/Results_' + img_name + '.csv') as f:
                f_csv = csv.reader(f)
                headers = next(f_csv)
                
                i = 0
                for row in f_csv:
                    center = [round(float(row[-2])), round(float(row[-1]))]
                    box = [center[0] - int(w/2), center[1] - int(w/2), 
                           center[0] + int(w/2), center[1] + int(w/2)]
                    cropped = img[box[1]:box[1]+w, box[0]:box[0]+w]    # [Ymin:Ymax , Xmin:Xmax]
                    cv2.imwrite("./Patches/Prediction/{}_{}.jpg"
                                .format(img_name, '0'*(4 - len(str(i))) + str(i)), 
                                cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    i += 1
                total_num += (f_csv.line_num - 1)
                
    else:
        for img_name in tqdm(val_set):
            img_i = str(int(int(img_name) / 16) + 1)
            img_j = str(int(img_name) % 16)
            
            for z_offset in tqdm(z_offsets):
                img_z_name = ('01-2019-02-19_16.23.01_x40_' + z_offset + '_' + 
                              'i' + '0'*(2 - len(img_i)) + img_i + 
                              'j' + '0'*(2 - len(img_j)) + img_j + '.jpg')
                img = cv2.imread('../01-2019-02-19_16.23.01_x40/' + img_z_name)
                
                with open('../FCRN/Results/Results_' + img_name + '.csv') as f:
                    f_csv = csv.reader(f)
                    headers = next(f_csv)
                    i = 0
                    for row in f_csv:
                        center = [round(float(row[-2])), round(float(row[-1]))]
                        box = [center[0] - int(w/2), center[1] - int(w/2), 
                               center[0] + int(w/2), center[1] + int(w/2)]
                        cropped = img[box[1]:box[1]+w, box[0]:box[0]+w]    # [Ymin:Ymax , Xmin:Xmax]
                        cv2.imwrite("./Patches/Z_expanded/{}_{}_{}.jpg"
                                    .format(img_name, '0'*(4 - len(str(i))) + str(i), z_offset), 
                                    cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        i += 1
                    total_num += (f_csv.line_num - 1)
                
    return total_num

#%%
def groundtruth_patch(w=80):
    """    
    Arguments:
    w -- window size for patches
    
    Returns:
    total_num -- total number of nucleus patches
    """     
    total_num = 0
    
    with open(file_train, 'r') as f:
        for line in f.readlines():      # for each image
            img_path = line.split()[0]
            img_name = img_path.split('/')[-1].split('.')[0]
            img = cv2.imread(img_path)    # read the image
            boxes = line.split()[1:]    # extract the bounding boxes
            boxes = [list(map(int, box.split(',')[:-1])) for box in boxes]   # convert to int
            
            # generate nuclei patches according to the bounding boxes
            i = 0
            for box in boxes:     # for each patch in each image
#                box = list(map(int, boxes[i].split(',')[:-1]))   # convert to int
                cropped = img[box[1]:box[1]+w, box[0]:box[0]+w]    # [Ymin:Ymax , Xmin:Xmax]
                cv2.imwrite("./Patches/GroundTruth/{}_{}.jpg".format(img_name, '0'*(4 - len(str(i))) + str(i)), 
                            cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                i += 1
                
            total_num += len(boxes)

    return total_num#, locations

#%%
if __name__ == '__main__':
    total_num = prediction_patch(z_expand=True)
#    total_num = groundtruth_patch()
    
