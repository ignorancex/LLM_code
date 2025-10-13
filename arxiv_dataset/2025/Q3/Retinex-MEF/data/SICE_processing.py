import os
import cv2
from tqdm import tqdm


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path)
    assert mode == 'RGB' or mode == 'Gray' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'Gray':  
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def image_save(path,img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path,img)

patchsize=128
stride=128

folderpath=r"data\SICE_training"

rawdata_path=r"data\SICE_sorted"

os.makedirs(folderpath,exist_ok=True)
patch_num=0
file_name_list=os.listdir(rawdata_path) 
for k in tqdm(range(len(file_name_list))):

    sub_folder=os.path.join(rawdata_path,file_name_list[k])
    subimgnum=len(os.listdir(sub_folder))

    for j in range(1,subimgnum+1):
        img1=image_read_cv2(os.path.join(sub_folder,str(1)+".JPG"))
        img2=image_read_cv2(os.path.join(sub_folder,str(subimgnum)+".JPG"))
        img3=image_read_cv2(os.path.join(sub_folder,str(j)+".JPG"))


        H,W,_=img1.shape
        p_H_num=(H-patchsize)//stride + 1
        p_W_num=(W-patchsize)//stride + 1

        for kk in range(p_H_num*p_W_num):
            
            os.makedirs(os.path.join(folderpath,str(patch_num)),exist_ok=True)


            a0=kk//p_W_num
            a1=kk-a0*p_W_num

            img1_patch=img1[a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize,:]
            img2_patch=img2[a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize,:]
            img3_patch=img3[a0*stride:a0*stride+patchsize,a1*stride:a1*stride+patchsize,:]

            image_save(os.path.join(folderpath,str(patch_num),"img_1.png"), img1_patch)
            image_save(os.path.join(folderpath,str(patch_num),"img_2.png"), img2_patch)
            image_save(os.path.join(folderpath,str(patch_num),"img_3.png"), img3_patch)

            patch_num+=1

print('training set, # samples %d\n' % (patch_num))       





