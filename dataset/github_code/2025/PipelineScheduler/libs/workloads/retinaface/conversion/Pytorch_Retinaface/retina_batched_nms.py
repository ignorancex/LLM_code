
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import onnx_graphsurgeon as gs
import onnx

import argparse
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
import cv2
from libs.workloads.retinaface.conversion.Pytorch_Retinaface.models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('-i', '--image_path', default='/home/hihi/code/detection/various_models/Pytorch_Retinaface/img_raw.jpg')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--device', default='cuda:0', help='Inference hardware')
parser.add_argument('--confidence_threshold', default=0.2, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--opset', default=17, type=int, help='ONNX opset version')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class ProcModel(nn.Module):
    def __init__(self, net, cfg, img_size, device='cuda:0'):
        super(ProcModel, self).__init__()
        self.device = device
        _, _, img_height, img_width = img_size
        self.scale = torch.Tensor([img_width, img_height, img_width, img_height]).to(device)
        self.priors_data = PriorBox(cfg, image_size=(img_height, img_width)).forward().data
        self.priors_data = self.priors_data.unsqueeze(0).to(device)
        self.variances = torch.FloatTensor(cfg['variance']).to(device)
        self.retina_model = net.to(device)

    def forward(self, x):
        loc, conf, _ = self.retina_model(x)
        
        print(loc.shape, self.priors_data.shape)
        boxes = torch.cat((self.priors_data[:, :, :2] + loc[:, :, :2] * self.variances[0] * self.priors_data[:, :, 2:],
        self.priors_data[:, :, 2:] * torch.exp(loc[:, :, 2:] * self.variances[1])), 2)
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
        
        boxes = boxes * self.scale
        
        boxes = boxes.unsqueeze(2)
        
        scores = conf[:, :, 1:2]
        # print(boxes[:, torch.argmax(scores, dim=1)])
        
        return boxes, scores

        # return [boxes, scores]
    
def create_attrs(topK, keepTopK):
    attrs = {}
    attrs["shareLocation"] = 1
    attrs["backgroundLabelId"] = -1
    attrs["numClasses"] = 1
    attrs["topK"] = topK
    attrs["keepTopK"] = keepTopK
    attrs["scoreThreshold"] = 0.5
    attrs["iouThreshold"] = 0.7
    attrs["isNormalized"] = False
    attrs["clipBoxes"] = False
    # 001 is the default plugin version the parser will search for, and therefore can be omit
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    return attrs


def add_nmsplugin_to_onnx(model_file, output_names=('bbox_out', 'score_out'), topk=200, keepTopK=100):
    graph = gs.import_onnx(onnx.load(model_file))  # load onnx model
    batch_size = graph.inputs[0].shape[0]
    tensors = graph.tensors()
    boxes_tensor = tensors[output_names[0]] # match with onnx model output name
    confs_tensor = tensors[output_names[1]] # match with onnx model output name
    num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1])
    nmsed_boxes = gs.Variable(name="nmsed_boxes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK, 4])
    nmsed_scores = gs.Variable(name="nmsed_scores").to_variable(dtype=np.float32, shape=[batch_size, keepTopK])
    nmsed_classes = gs.Variable(name="nmsed_classes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK])
    new_outputs = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes] # do not change
    nms_node = gs.Node( # define nms plugin
        op="BatchedNMSDynamic_TRT",  # match with batchedNMSPlugn
        attrs=create_attrs(topk, keepTopK),  # set attributes for nms plugin
        inputs=[boxes_tensor, confs_tensor],
        outputs=new_outputs
    )
    graph.nodes.append(nms_node)  # nms plugin added
    graph.outputs = new_outputs
    graph = graph.cleanup().toposort()
    
    onnx.save(gs.export_onnx(graph), model_file)  # save model
    return model_file



if __name__ == '__main__':
    device = args.device
    
    image_path = args.image_path
    
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_raw = cv2.resize(img_raw, (320, 288), img_raw)
    img = np.float32(img_raw)
    
    im_height, im_width, _ = img.shape
    img -= (104, 117, 123)
    
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.device)
    net = ProcModel(net, cfg=cfg, img_size=img.shape, device=args.device)
    net.eval()
    print('Finished loading model!')
    
    net = net.to(device)
    
    # out = net(img)
    
    # print(out)
    
    
        
    input_names = ['images']
    output_names = ['bbox_out', 'score_out']
    train = False
    opset_version = 12
    
    dynamic = {'images': {0: 'batch'}}
    dynamic['bbox_out'] = {0: 'batch'}
    dynamic['score_out'] = {0: 'batch'}

    torch.onnx.export(net, img, 'retina_single_face.onnx', verbose=False, opset_version=opset_version,
                      training=torch.onnx.TrainingMode.EVAL,
                      export_params=True,
                      do_constant_folding=False,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic)
    
    add_nmsplugin_to_onnx('retina_single_face.onnx', topk=20, keepTopK=5)