import argparse


def add_shared_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--random-seed", type=int, default=None,
                        help="set random seed; default to not set")
    parser.add_argument("--save-dir", type=str, default="./temp",
                        help="set which dictionary to save the experiment result")
    parser.add_argument("--save-name", type=str, default="temp-name",
                        help="set the save name of the experiment result")
    parser.add_argument("--exp-config", type=str, default=None)
    parser.add_argument("--exp-type", type=str, default=None,
                        choices=["embed", "train", "atk", "eval", "furatk", "eval-furatk"])
    parser.add_argument("--cpu-eval", action="store_true")


def add_cls_task_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--cls-num", type=int, default=None)
    parser.add_argument("--embed-bs", type=int, default=None)

    parser.add_argument("--trainset-path", type=str, default=None)
    parser.add_argument("--testset-path", type=str, default=None)

    parser.add_argument("--classifier-path", type=str, default=None)


def add_img_exp_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--encoder", type=str, default=None,
                        choices=["rn34-hf", "rn50-hf", "mobilenetv3-hf", "rn34-ms", "rn50-ms",
                                 "clip-vit-l14", "clip-vit-l14-336", "clip-vit-b16", "clip-vit-b32",
                                 "openclip-vit-b32", "openclip-vit-l14", "openclip-vit-h14"],
                        help="set the pre-trained encoder")

    parser.add_argument("--dataset", type=str, default=None,
                        choices=["cifar10", "cifar100", "svhn", "stl10", "food101"])

    parser.add_argument("--atk-target-img", type=str, default=None)

    parser.add_argument("--jpeg-def", action="store_true")
    parser.add_argument("--resize-atk", type=int, default=None)



def add_text_exp_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--encoder", type=str, default=None,
                        choices=["bertbase", "bertlarge",
                                 "t5small", "t5base", "robertabase", "clip-vit-l14"])
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["yelp", "sst2", "sst5", "agnews", "trec"])

    parser.add_argument("--atk-target-text", type=str, default=None)
    parser.add_argument("--atkdata-path", type=str, default=None)

    parser.add_argument("--def-flip-rate", type=float, default=None)


def add_img2text_exp_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--encoder", type=str, default=None,
                        choices=["clip-vit-l14", "clip-vit-l14-336", "clip-vit-b16", "clip-vit-b32",
                                 "openclip-vit-b32", "openclip-vit-l14", "openclip-vit-h14"],
                        help="set the pre-trained encoder")

    parser.add_argument("--atk-target-img", type=str, default=None)
    parser.add_argument("--further-atk-img", type=str, default=None)
    parser.add_argument("--further-atk-img-num", type=int, default=1)
    parser.add_argument("--eval-imgs", type=str, default=None)

    # parser.add_argument("--jpeg-def", action="store_true")
    # parser.add_argument("--resize-atk", type=int, default=None)


def add_text2img_exp_args(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument("--encoder", type=str, default=None,
                        choices=["clip-vit-l14", "clip-vit-l14-336", "clip-vit-b16", "clip-vit-b32",
                                 "openclip-vit-b16", "openclip-vit-b32", "openclip-vit-l14", "openclip-vit-h14"])

    parser.add_argument("--text2img-model", type=str, default=None,
                        choices=["sd-v1-2", "sd-v1-4", "sd-v2-1-base", "sd-v2-1"])
    # parser.add_argument("--eval-encoder", type=str, default=None,
    #                     choices=["clip-vit-l14", "clip-vit-b16", "clip-vit-b32"])

    parser.add_argument("--atk-target-text", type=str, default=None)
    parser.add_argument("--atkdata-path", type=str, default=None)

    parser.add_argument("--def-flip-rate", type=float, default=None)
