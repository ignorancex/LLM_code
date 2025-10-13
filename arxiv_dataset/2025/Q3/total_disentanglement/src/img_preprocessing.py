from re import A
import cv2
import numpy as np
import pickle
from tqdm import tqdm


def preprocessing(img,img_size=64,margin=0,threshold_w=0,threshold_h=0,out_channel=3):
    # if len(img) > 2:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_white = cv2.bitwise_not(img)#np.where(img_org>0,0,255) #文字領域白
    y_min = 0
    y_max = 0
    x_min = 0
    x_max = 0
    # 行の処理
    for i in range(img_white.shape[0]):
        if np.sum(img_white[i,:]) > 0:
            y_min = i
            break
    for i in reversed(range(img_white.shape[0])):
        if np.sum(img_white[i,:]) > 0:
            y_max = i+1 #rangeが0~n-1なので，arrayのインデックス調整で+1する　例：img[0:N] これも0~N-1になっているので+1しておかないとバグる
            break
    # 列の処理
    for i in range(img_white.shape[1]):
        if np.sum(img_white[:,i]) > 0:
            x_min = i
            break
    for i in reversed(range(img_white.shape[1])):
        if np.sum(img_white[:,i]) > 0:
            x_max = i+1
            break
    img = img_white[y_min:y_max,x_min:x_max]
    h = img.shape[0]
    w = img.shape[1]
    if (h<threshold_h) or (w<threshold_w):
        # print('error')
        return 0
    if margin>0:
        img = np.pad(img,[(margin,margin),(margin,margin)],'constant')
    size = max(w,h)
    ratio = img_size/size #何倍すれば良いか
    img_resize = cv2.resize(img, (int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_CUBIC)
    # img_resize = cv2.bitwise_not(img_resize) #文字領域黒
    #0埋めの幅を決める
    if w > h:
        pad = int((img_size - h*ratio)/2)
        #np.pad()の第二引数[(上，下),(左，右)]にpaddingする行・列数
        img_resize = np.pad(img_resize,[(pad,pad),(0,0)],'constant')
    elif h > w:
        pad = int((img_size - w*ratio)/2)
        img_resize = np.pad(img_resize,[(0,0),(pad,pad)],'constant')
    #最終的にきれいに100x100にresize
    img_resize = cv2.resize(img_resize,(img_size,img_size),interpolation=cv2.INTER_CUBIC)
    img_resize = cv2.bitwise_not(img_resize)#np.where(img_resize!=0,0,255)
    if out_channel == 3:
        img_resize_ = np.dstack((img_resize,img_resize))
        img_resize = np.dstack((img_resize,img_resize_))
    return img_resize

def main():
    img_path = '../dataset/myfont/fontimage/music-sheets_VV.png'##'../dataset/myfont/fontimage/yuletide_AA.png'
    # img_path = '../dataset/myfont/fontimage/1-up_AA.png'
    img = preprocessing(img_path,img_size=64)
    cv2.imwrite('sample_v2.png',img)

if __name__ == '__main__':
    main()