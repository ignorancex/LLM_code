import numpy as np
import torch

import fcd_utils.classifiers as classifiers
from source.models.architectures import basicVGG2D_04layers, SqueezeNet_source0

classes = ["Background", "Four chamber"]

verbose = False
net = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = []


def get_classes():
    return classes


def initialize(input_size: int, clip_duration: int, n_classes: int, model_path: str = None, modelname: str = None, verb: bool = False):
    """
    Load the model and initialize the network
    model_path: path where the config file and the model weights are.

    """
    global device
    global net
    global verbose

    verbose = verb
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # n_frames_per_clip = 10
    # net = classifiers.SimpleVideoClassifier(input_size, n_frames_per_clip, n_classes)
    # net = basicVGG2D_04layers(in_channels=NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP,
    #                                       num_classes=n_classes,
    #                                       n_frames_per_clip=NUMBER_OF_FRAMES_PER_SEGMENT_IN_A_CLIP)
    net = SqueezeNet_source0(num_classes= n_classes,
                            in_channels=clip_duration)
                            # Total params: 733,580 ## Training curves look good
    net.to(device)
    net.eval()
    if verbose:
        print(net)
        print("the model is : " + modelname)

    net_params = torch.load('{}/{}'.format(model_path, modelname))
    net.load_state_dict(net_params)
    net.eval()
    return True



def dowork(frames: np.array, verbose: int = 0):
    with torch.no_grad():
        print(f' -------------------- FourCHDetection_worker:dowork() -------------------------')
        print(f' FourCHDetection_worker:dowork(): frames.size() {frames.shape}')
        frames = torch.from_numpy(frames).type(torch.float).to(device).unsqueeze(0) / 255.0
        print(f' FourCHDetection_worker:dowork(): torch.from_numpy(frames).size() {frames.shape}')

        try:
            out = net(frames)
            print(f' out {out}')
            out_index = torch.argmax(out, dim=1)
            print(f' out_index {out_index}')

        except Exception as ex:
            print('[Python exception caught] FourChDetection_worker::do_work() - {}{}'.format(ex,
                                                                                              ex.__traceback__.tb_lineno))
            exit(-1)

    # out = torch.nn.functional.softmax(out, dim=1).cpu().numpy()
    out = out.cpu().numpy()
    return out
