import copy
import json
import numpy as np

data={
    "fileName": "bar_type1_%d",
    "outputWidth": 100,
    "outputHeight": 100,
    "barCount": 10,
    "barWidth": 7,
    "color": {
        "colorDiffLeast": 5,
        "useFixSettingChance": 1.0,
        "fillRectChance": 1.0,
        "fixColorBar": {
            "colorIsFixed": False,
            "color": [
                0,
                0,
                0
            ]
        },
        "background": {
            "colorIsFixed": False,
            "color": [
                255,
                255,
                255
            ]
        },
        "fixStroke": {
            "colorIsFixed": False
        }
    },
    "lineThickness": 1,
    "spacePaddingLeft": 2,
    "spacePaddingRight": 1,
    "spacePaddingTop": 0,
    "spacePaddingBottom": 1,
    "labelValue": 1,
    "midGap": 4,
    "fixBarGap": -1,
    "values": {
        "valueRange": [
            10,
            93
        ],
        "pixelValue": True,
        "useSpecialGen": False
    },
    "mark": {
        "dotColor": [
            0,
            0,
            0
        ],
        "genFix": 2,
        "fix": [],
        "ratio": {
            "ratioMarkOnly": False,
            "ratio2Only": False,
            "ratioNotMarkOnly": False,
            "ratio2MarkOnly": True
        },
        "bottom":True,
        "bottomValue":5
    }
}
# for i in range(4,9):
#     for j in range(1,11,2):
#         data_copy=copy.deepcopy(data)
#         data_copy['barWidth']=int(i)
#         if i>7:
#             data_copy['fixBarGap']=1
#         data_copy['mark']['bottomValue']=int(j)
#         with open('position_length_type_1_barWidth_dotpos_'+str(i)+'_'+str(j)+'.json','w') as f:
#             json.dump(data_copy,f,indent=4)
# for i in range(4,9):
#     for j in np.arange(0.3,0.8,0.1):
#         data_copy=copy.deepcopy(data)
#         data_copy['barWidth']=int(i)
#         if i>7:
#             data_copy['fixBarGap']=1
#         # data_copy['mark']['bottomValue']=int(i)
#         data_copy["TitlePosition"]="left"
#         data_copy["TitlePaddingLeft"]=float(j)
#         with open('position_length_type_1_barWidth_title_'+str(i)+'_'+str(int(j*10))+'.json','w') as f:
#             json.dump(data_copy,f,indent=4)
# for i in range(1,11,2):
#     for j in np.arange(0.3,0.8,0.1):
#         data_copy=copy.deepcopy(data)
#         data_copy['mark']['bottomValue']=int(i)
#         data_copy["TitlePosition"]="left"
#         data_copy["TitlePaddingLeft"]=float(j)
#         with open('position_length_type_1_dotpos_title_'+str(i)+'_'+str(int(j*10))+'.json','w') as f:
#             json.dump(data_copy,f,indent=4)

for i in range(4,9):
    for j in range(1,11,2):
        data_copy=copy.deepcopy(data)
        data_copy['barWidth']=int(i)
        if i>7:
            data_copy['fixBarGap']=1
        data_copy['mark']['bottomValue']=int(j)
        data_copy['values']['valueRange']=[1,100]
        with open('position_length_type_1_barLength_barWidth_dotpos_'+str(i)+'_'+str(j)+'.json','w') as f:
            json.dump(data_copy,f,indent=4)