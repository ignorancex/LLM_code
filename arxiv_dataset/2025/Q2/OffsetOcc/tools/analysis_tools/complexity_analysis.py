import argparse

import torch
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, DictAction
from mmengine.analysis import parameter_count, parameter_count_table, flop_count
from offsetocc.registry import MODELS
from offsetocc.utils import register_all_modules
from offsetocc.structures import OccDataSample

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    register_all_modules()

    # define model
    model = MODELS.build(cfg.model)

    print("PARAMETER COUNT")
    print(parameter_count(model))
    print(parameter_count_table(model))

    # print("FLOPS")
    # inputs = {'imgs': torch.rand(1, 6, 3, 900, 1600)}
    # flops count does not work with OccDataSample
    # data_samples = [OccDataSample(metainfo={'img_shape': (900, 1600),
    #                             'input_shape': (900, 1600),
    #                             'lidar2img': [[[1241.920166015625, 842.0457153320312, 37.17252731323242, -351.5425720214844],
    #                                             [-16.829845428466797, 540.2156372070312, -1224.0548095703125, -652.0930786132812],
    #                                             [-0.012994648888707161, 0.9983007311820984, 0.05680442228913307, -0.427213191986084],
    #                                             [0.0, 0.0, 0.0, 0.0]],
    #                                         [[1365.5628662109375, -617.9367065429688, -39.84254455566406, -457.8063049316406],
    #                                             [380.9697265625, 323.6235656738281, -1238.4705810546875, -696.0205078125],
    #                                             [0.8428741097450256, 0.5369926691055298, 0.034670840948820114, -0.6073117852210999],
    #                                             [0.0, 0.0, 0.0, 0.0]],
    #                                         [[29.973461151123047, 1502.9454345703125, 82.4765853881836, -309.6894226074219],
    #                                             [-387.53155517578125, 323.80841064453125, -1237.137939453125, -690.0399169921875],
    #                                             [-0.8243371844291687, 0.5645750761032104, 0.04151139408349991, -0.535756528377533],
    #                                             [0.0, 0.0, 0.0, 0.0]],
    #                                         [[-803.6638793945312, -850.9808959960938, -27.80868911743164, -874.9713134765625],
    #                                             [-10.141632080078125, -444.4715270996094, -815.3507080078125, -711.7434692382812],
    #                                             [-0.00795462541282177, -0.9991635680198669, -0.040110912173986435, -1.020156741142273],
    #                                             [0.0, 0.0, 0.0, 0.0]],
    #                                         [[-1186.6087646484375, 923.2051391601562, 53.290367126464844, -620.9562377929688],
    #                                             [-462.5159606933594, -102.38880920410156, -1252.530029296875, -562.7839965820312],
    #                                             [-0.9475737810134888, -0.319522500038147, 0.0030462073627859354, -0.4341076612472534],
    #                                             [0.0, 0.0, 0.0, 0.0]],
    #                                         [[286.2206115722656, -1468.9803466796875, -61.87546920776367, -274.95916748046875],
    #                                             [446.3054504394531, -120.27753448486328, -1250.0810546875, -591.083984375],
    #                                             [0.9243063926696777, -0.38163694739341736, -0.0032980209216475487, -0.46312063932418823],
    #                                             [0.0, 0.0, 0.0, 0.0]]
    #                                         ],
    #                             'lidar_points': {
    #                                 'lidar2ego': [
    #                                     [-0.0005427949945442379, 0.9989306926727295, 0.046229466795921326, 0.9857929944992065],
    #                                     [-0.9999954700469971, -0.0004056931647937745, -0.0029750114772468805, 0.0],
    #                                     [-0.002953075338155031, -0.046230874955654144, 0.9989264011383057, 1.840190052986145],
    #                                     [0.0, 0.0, 0.0, 1.0]
    #                                 ]
    #                             }})]
    # print(flop_count(model, (inputs, data_samples)))

if __name__ == '__main__':
    main()
