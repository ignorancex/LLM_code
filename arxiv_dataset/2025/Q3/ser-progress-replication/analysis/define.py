models = [
    {
        "name": r"\emph{AST}",
        "data": "wav",
        "dir": "ASTModel-T",
        "group": "'16- Supervised TL",
        "year": 2021,
    },
    {
        "name": r"\cnn",
        "data": "mel-16k",
        "dir": "Cnn14-16k-T",
        "group": "'16- Supervised TL",
        "year": 2020,
    },
    {
        "name": r"\emph{ETDNN}",
        "data": "wav",
        "dir": "TDNNFFNN-T",
        "group": "'16- Supervised TL",
        "year": 2020,
    },
    {
        "name": r"\emph{EffNet}",
        "data": "mel-16k",
        "dir": "EfficientNet-B0-T",
        "group": "'12- ImageNet",
        "year": 2019,
    },
    {
        "name": r"\emph{Resnet50}",
        "dir": "ResNet-50-T",
        "data": "mel-16k",
        "group": "'12- ImageNet",
        "year": 2015,
    },
    {
        "name": r"\wemo",
        "dir": "w2v2-l-emo",
        "data": "wav",
        "group": "'20- Audio SSL",
        "year": 2023,
    },
    {
        "name": r"\wlarge",
        "dir": "w2v2-l",
        "group": "'20- Audio SSL",
        "year": 2020,
        "data": "wav",
    },
    {
        "name": r"\wrobust",
        "dir": "w2v2-l-rob",
        "group": "'20- Audio SSL",
        "year": 2021,
        "data": "wav",
    },
    {
        "name": r"\wbase",
        "dir": "w2v2-b",
        "group": "'20- Audio SSL",
        "year": 2020,
        "data": "wav",
    },
    {
        "name": r"\wvox",
        "dir": "w2v2-l-100k",
        "group": "'20- Audio SSL",
        "year": 2021,
        "data": "wav",
    },
    {
        "name": r"\hbase",
        "dir": "hubert-b-ls960",
        "group": "'20- Audio SSL",
        "year": 2021,
        "data": "wav",
    },
    {
        "name": r"\hlarge",
        "dir": "hubert-l-ll60k",
        "group": "'20- Audio SSL",
        "year": 2021,
        "data": "wav",
    },
    {
        "name": r"\emph{IS09-s-mlp}",
        "dir": "FFNN-IS09",
        "group": "'09- openSMILE",
        "year": 2009,
        "data": "IS09-func"
    },
    {
        "name": r"\emph{IS09-d-lstm}",
        "dir": "Seq-FFNN-IS09",
        "group": "'09- openSMILE",
        "year": 2009,
        "data": "IS09-llds"
    },
    {
        "name": r"\emph{IS10-s-mlp}",
        "dir": "FFNN-IS10",
        "group": "'09- openSMILE",
        "year": 2010,
        "data": "IS10-func"
    },
    {
        "name": r"\emph{IS10-d-lstm}",
        "dir": "Seq-FFNN-IS10",
        "group": "'09- openSMILE",
        "year": 2010,
        "data": "IS10-llds"
    },
    {
        "name": r"\emph{IS11-s-mlp}",
        "dir": "FFNN-IS11",
        "group": "'09- openSMILE",
        "year": 2011,
        "data": "IS11-func"
    },
    {
        "name": r"\emph{IS11-d-lstm}",
        "dir": "Seq-FFNN-IS11",
        "group": "'09- openSMILE",
        "year": 2011,
        "data": "IS11-llds"
    },
    {
        "name": r"\emph{IS12-s-mlp}",
        "dir": "FFNN-IS12",
        "group": "'09- openSMILE",
        "year": 2012,
        "data": "IS12-func"
    },
    {
        "name": r"\emph{IS12-d-lstm}",
        "dir": "Seq-FFNN-IS12",
        "group": "'09- openSMILE",
        "year": 2012,
        "data": "IS12-llds"
    },
    {
        "name": r"\emph{IS13-s-mlp}",
        "dir": "FFNN-IS13",
        "group": "'09- openSMILE",
        "year": 2013,
        "data": "IS13-func"
    },
    {
        "name": r"\emph{IS13-d-lstm}",
        "dir": "Seq-FFNN-IS13",
        "group": "'09- openSMILE",
        "year": 2013,
        "data": "IS13-llds"
    },
    {
        "name": r"\emph{eGeMAPS-d-lstm}",
        "dir": "Seq-FFNN-eGeMAPS",
        "group": "'09- openSMILE",
        "year": 2015,
        "data": "eGeMAPS-llds"
    },
    {
        "name": r"\emph{eGeMAPS-s-mlp}",
        "dir": "FFNN-eGeMAPS",
        "group": "'09- openSMILE",
        "year": 2015,
        "data": "eGeMAPS-func"
    },
    {
        "name": r"\emph{IS16-s-mlp}",
        "dir": "FFNN-IS16",
        "group": "'09- openSMILE",
        "year": 2016,
        "data": "ComParE-func"
    },
    {
        "name": r"\emph{IS16-d-lstm}",
        "dir": "Seq-FFNN-IS16",
        "group": "'09- openSMILE",
        "year": 2016,
        "data": "ComParE-llds"
    },
    {
        "name": r"\emph{CRNN}$^{18}$",
        "dir": "End2You-emo18",
        "group": "'16- End-to-end",
        "year": 2018,
        "data": "wav"
    },
    {
        "name": r"\emph{CRNN}$^{19}$",
        "dir": "End2You-zhao19",
        "group": "'16- End-to-end",
        "year": 2019,
        "data": "wav"
    },
    {
        "name": r"\emph{Whisper}$^t$",
        "dir": "Whisper-FFNN-Tiny-T",
        "group": "'16- Supervised TL",
        "year": 2023,
        "data": "wav"
    },
    {
        "name": r"\emph{Whisper}$^b$",
        "dir": "Whisper-FFNN-Base-T",
        "group": "'16- Supervised TL",
        "year": 2023,
        "data": "wav"
    },
    {
        "name": r"\emph{Whisper}$^s$",
        "dir": "Whisper-FFNN-Small-T",
        "group": "'16- Supervised TL",
        "year": 2023,
        "data": "wav"
    },
    {
        "name": r"\emph{VGG}$^{11}$",
        "dir": "VGG-11-T",
        "group": "'12- ImageNet",
        "year": 2016,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{VGG}$^{13}$",
        "dir": "VGG-13-T",
        "group": "'12- ImageNet",
        "year": 2016,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{VGG}$^{16}$",
        "dir": "VGG-16-T",
        "group": "'12- ImageNet",
        "year": 2016,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{VGG}$^{19}$",
        "dir": "VGG-19-T",
        "group": "'12- ImageNet",
        "year": 2016,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{Swin}$^t$",
        "dir": "Swin-T-T",
        "group": "'12- ImageNet",
        "year": 2021,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{Swin}$^b$",
        "dir": "Swin-B-T",
        "group": "'12- ImageNet",
        "year": 2021,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{Swin}$^s$",
        "dir": "Swin-S-T",
        "group": "'12- ImageNet",
        "year": 2021,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{AlexNet}",
        "dir": "AlexNet-T",
        "group": "'12- ImageNet",
        "year": 2012,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{ConvNeXt}$^t$",
        "dir": "ConvNeXt-Tiny-T",
        "group": "'12- ImageNet",
        "year": 2020,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{ConvNeXt}$^b$",
        "dir": "ConvNeXt-Base-T",
        "group": "'12- ImageNet",
        "year": 2020,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{ConvNeXt}$^s$",
        "dir": "ConvNeXt-Small-T",
        "group": "'12- ImageNet",
        "year": 2020,
        "data": "mel-16k"
    },
    {
        "name": r"\emph{ConvNeXt}$^l$",
        "dir": "ConvNeXt-Large-T",
        "group": "'12- ImageNet",
        "year": 2020,
        "data": "mel-16k"
    },
]

def decode(x):
    if x == 0:
        return "anger"
    elif x == "A":
        return "anger"
    elif x == 1:
        return "happiness"
    elif x == "H":
        return "happiness"
    elif x == 2:
        return "neutral"
    elif x == "N":
        return "neutral"
    elif x == 3:
        return "sadness"
    elif x == "S":
        return "sadness"
    else:
        raise NotImplementedError(x)