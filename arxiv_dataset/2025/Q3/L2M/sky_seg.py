import os
import copy
import time
import argparse
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm
import imutils

def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype('float32')

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype('uint8')

    return onnx_result

    from argparse import ArgumentParser

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--debug", action='store_true', default=False)
    args, _ = parser.parse_known_args()

    onnx_session = onnxruntime.InferenceSession("checkpoints/skyseg.onnx")
    base = args.base
    img_base = os.path.join(base, "image1")
    out_base = os.path.join(base, "debug2")
    depth_base = os.path.join(base, "depth1")

    if not os.path.exists(out_base):
        os.mkdir(out_base)
        
    count = 0
        
    for img in tqdm(os.listdir(img_base)):

        # image = cv2.imread("../zeb/eth3do/playground-DSC_0585.png")
        image = cv2.imread(os.path.join(img_base, img))
        H, W = image.shape[:2]

        while(image.shape[0] >= 640 and image.shape[1] >= 640):
            image = cv2.pyrDown(image)
        result_map = run_inference(onnx_session,[320,320],image)
        result_map = imutils.resize(result_map, height=H, width=W)

        # cv2.imwrite(os.path.join(out_base, img), result_map)
        depth = np.load(os.path.join(depth_base, img[:-4]+".npy"))
        
        disp_ = 1 / (depth + 1)
        disp2_ = 1 / (depth + 1)
        
        depth[result_map > 255 * 0.5] = 0.0
        np.save(os.path.join(depth_base, img[:-4]+".npy"), depth)
        
        if args.debug and count < 100:
            
            disp2_[result_map > 255 * 0.5] = 0.0
            
            disp_ = ((disp_ - disp_.min()) / (disp_.max() - disp_.min()) * 255).astype(np.uint8)
            disp2_ = ((disp2_ - disp2_.min()) / (disp2_.max() - disp2_.min()) * 255).astype(np.uint8)
            
            cv2.imwrite(os.path.join(out_base, img), np.hstack([disp_, disp2_, result_map]))
        
        # import IPython
        # IPython.embed()
        # exit()
        count += 1