import argparse
import torch
# from torch2trt_dynamic import torch2trt_dynamic
from model_irse import IR_50


def export_onnx(model, im, file, opset=12, dynamic=True):
    # YOLOv5 ONNX export

    output_names = ['output']
    if dynamic:
        file = file.split('.')[0] + '_dynamic'
    f = file + '.onnx'
    if dynamic:
        # dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        dynamic = {'input': {0: 'batch'}}
        dynamic['output'] = {0: 'batch'}

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic or None)

    # Checks
    # model_onnx = onnx.load(f)  # load onnx model
    # onnx.checker.check_model(model_onnx)  # check onnx model

    # # Metadata
    #mer d = {'stride': int(max(model.stride)), 'names': model.names}
    # for k, v in d.items():
    #     meta = model_onnx.metadata_props.add()
    #     meta.key, meta.value = k, str(v)
    # onnx.save(model_onnx, f)
    # Simplify
    return f


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='backbone_ir50_asia.pth', type=str)
parser.add_argument(
    '-o', '--output', default='../../weight/torch/arc/ir50_asia-l2norm-db.onnx', type=str)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('--width', type=int, default=112)
parser.add_argument('--height', type=int, default=112)
parser.add_argument('-d', '--enable_dynamic_axes',
                    action="store_true", default=False)

args = parser.parse_args()
input_size = [args.height, args.width]
dummy_input = torch.randn(
    [args.batch_size, 3, args.height, args.width], device='cuda')
model = IR_50(input_size)
model.load_state_dict(torch.load(args.model))
model.cuda()
model.eval()
print(model)
# model(dummy_input)
# exit(0)

# export
input_names = ["input"]
output_names = ["output"]

# convert to TensorRT feeding sample data as input
opt_shape_param = [
    [
        [1, 3, args.height, args.width],   # min
        [1, 3, args.height, args.width],   # opt
        [1, 3, args.height, args.width]    # max
    ]
]

inputs = torch.randn(1, 3, args.height, args.width)
export_onnx(model, inputs, 'arcface.onnx')
# print('torch2trt_dynamic')
# model_trt = torch2trt_dynamic(model, [dummy_input], fp16_mode=True,
#                               opt_shape_param=opt_shape_param, input_names=input_names, output_names=output_names)
# save_path = f'arcface-ir50_asia-{args.height}x{args.width}-b1-fp16.engine'
# print('Saving')
# with open(save_path, 'wb') as f:
#     f.write(model_trt.engine.serialize())
