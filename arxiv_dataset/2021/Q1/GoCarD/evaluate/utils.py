import os
from load_data import get_anchors

def create_data_name(train1, test, train2, train2_fraction, mix_devel):
    # mix_car
    # muse_part
    if train2 is None:
        if train1 == test:
            dic_name = train1
        elif train1 != test:
            dic_name = train1 + '-X-' + test
    elif mix_devel:
        dic_name = train1 + '-+-' + str(train2_fraction) + "-" + train2 + '-X-' + train1 + '_' + train2 + '-X-' + test
    else:
        dic_name = train1 + '-+-' + str(train2_fraction) + "-" + train2 + '-X-' + test

    return dic_name
    
    mix_devel

def anchor_file(CONFIG_PATH, backbone):

    if backbone =="darknet53" or backbone =="squeezenet":
        anchors_path = os.path.join(CONFIG_PATH,'darknet53_anchors.txt')
    elif  backbone =="tinydarknet":
        anchors_path = os.path.join(CONFIG_PATH,'tiny_yolo_anchors.txt')
    else:
        # missing squeeznet
        print('Anchorfile not known')
        exit()
    return get_anchors(anchors_path)