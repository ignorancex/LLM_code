from src.metric.FINC import FINC_Evaluator
from src.datasets.ImageFilesDataset import ImageFilesDataset
from torchvision.datasets import ImageFolder
import torch
from argparse import ArgumentParser, Namespace

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--sigma', type=float, default=15, help='parameter for bandwidth')
    parser.add_argument('--fe', type=str, default='inception', help='embedding to use, expected one of [inception, dinov2, clip, resnet50, swav]')
    parser.add_argument('--eta', type=float, default=1, help='parameter in the difference of covariance matrix C_x - \eta * C_y')
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--rff', type=int, default=3000, help='dimension of random fourier feature')
    parser.add_argument('--novel_data_path', type=str, help='Folder containing images from testing distribution')
    parser.add_argument('--ref_data_path', type=str, help='Folder containing images from reference distribution')
    parser.add_argument('--save_name', type=str, help='unique name for saving and loading results')

    args = parser.parse_args()

    FINC = FINC_Evaluator(logger_path='./logs', batchsize=50, sigma=args.sigma, eta=args.eta, num_samples=args.samples, result_name=args.save_name, rff_dim=args.rff)
    FINC.set_feature_extractor(args.fe, save_path='./save')

    ref_path = args.ref_data_path
    novel_path = args.novel_data_path

    novel_dataset = ImageFolder(novel_path)
    ref_dataset = ImageFolder(ref_path)

    assert len(novel_dataset) > 0 and len(ref_dataset) > 0

    FINC.rff_differential_clustering_modes_of_dataset(novel_dataset, ref_dataset)