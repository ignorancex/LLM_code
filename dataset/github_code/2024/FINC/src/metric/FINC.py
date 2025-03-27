import torch
from argparse import ArgumentParser, Namespace
from .algorithm_utils import *
from os.path import join
from src.features.CLIPFeatureExtractor import CLIPFeatureExtractor
from src.features.DINOv2FeatureExtractor import DINOv2FeatureExtractor
from src.features.InceptionFeatureExtractor import InceptionFeatureExtractor
from src.features.ResNetFeatureExtractor import ResNet50FeatureExtractor
from src.features.SwAVFeatureExtractor import SwAVFeatureExtractor

import time
import logging
import sys

def get_logger(filepath='./logs/novelty.log'):
    '''
        Information Module:
            Save the program execution information to a log file and output to the terminal at the same time
    '''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

class FINC_Evaluator():
    def __init__(self, logger_path : str, sigma : float, eta : float, result_name: str, num_samples: int = 50000, batchsize: int = 128, rff_dim: int = 0):
        self.logger_path = logger_path
        self.sigma = sigma
        self.eta = eta
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.rff_dim = rff_dim

        self.current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        self.result_name = '{}_num_{}_sigma_{}_eta_{}'.format(result_name, num_samples, sigma, eta)
        self.save_feats_name = '{}_num_{}'.format(result_name, num_samples)


        self.feature_extractor = None
        self.name_feature_extractor = None
        self.running_logger = None

        self.init_running_logger()
        self.running_logger.info("FINC Evaluator Initialized.")
    
    def init_running_logger(self):
        self.running_logger = get_logger(join(self.logger_path, 'run_{}_{}.log'.format(self.result_name, self.current_time)))
    
    def set_feature_extractor(self, name: str, save_path=None):
        if name.lower() == 'inception':
            self.feature_extractor = InceptionFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'dinov2':
            self.feature_extractor = DINOv2FeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'clip':
            self.feature_extractor = CLIPFeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'resnet50':
            self.feature_extractor = ResNet50FeatureExtractor(save_path, logger=self.running_logger)
        elif name.lower() == 'swav':
            self.feature_extractor = SwAVFeatureExtractor(save_path, logger=self.running_logger)
        else:
            raise NotImplementedError(
                f"Cannot get feature extractor '{name}'. Expected one of ['inception', 'dinov2', 'clip', 'resnet50', 'swav']"
            )
        self.name_feature_extractor = name.lower()
        self.running_logger.info("Initialized feature-extractor network: {}".format(self.name_feature_extractor))
    
    
    def rff_clustering_modes_of_dataset(self,
                                        test_dataset: torch.utils.data.Dataset):
        
        '''
            Do a clustering task on a large dataset with Random Fourier Feature approximation
            This is different from FINC task
        '''
        assert self.rff_dim > 0
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual='./visuals/modes_rff',
                         num_visual_mode=10,
                         num_img_per_mode=25,
                         resize_img_to=224
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))

        if self.feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default Inception-V3.")
            self.set_feature_extractor(name='inception', logger=self.running_logger)
        
        with torch.no_grad():
            self.running_logger.info("Calculating test feats:")
            test_feats, test_idxs = self.feature_extractor.get_features_and_idxes(test_dataset, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=False, 
                                                                    num_samples=args.num_samples, 
                                                                    batchsize=args.batchsize)
        
        self.running_logger.info("number of test feature: {}".format(len(test_feats)))
        visualize_mode_by_eigenvectors_rff(test_feats, test_dataset, test_idxs, args)
    
    def rff_differential_clustering_modes_of_dataset(self,
                                                    test_dataset: torch.utils.data.Dataset,
                                                    ref_dataset: torch.utils.data.Dataset):
        
        '''
            Do a differential clustering task on a large dataset with Random Fourier Feature approximation
        '''

        assert self.rff_dim > 0
        
        args = Namespace(num_samples=self.num_samples, 
                         batchsize=self.batchsize, 
                         sigma=self.sigma, 
                         eta=self.eta,
                         rff_dim=self.rff_dim,
                         logger=self.running_logger,
                         backbone=self.name_feature_extractor,
                         visual_name=self.result_name,
                         current_time=self.current_time,
                         path_save_visual='./visuals/novel_modes_rff',
                         num_visual_mode=10,
                         num_img_per_mode=25,
                         resize_img_to=224
        )
        
        self.running_logger.info("Running RFF approximation with dim: {}x2".format(args.rff_dim))
        self.running_logger.info("Num_samples_per_distribution: {}, Sigma: {}".format(args.num_samples, args.sigma))
        self.running_logger.info('test dataset length: {}'.format(len(test_dataset)))
        self.running_logger.info('ref dataset length: {}'.format(len(ref_dataset)))


        if self.feature_extractor is None:
            self.running_logger.info("Feature extractor is not specified, use default Inception-V3.")
            self.set_feature_extractor(name='inception', logger=self.running_logger)
        
        with torch.no_grad():
            self.running_logger.info("Calculating test feats:")
            test_feats, test_idxs = self.feature_extractor.get_features_and_idxes(test_dataset, 
                                                                    name = 'test_' + self.save_feats_name, 
                                                                    recompute=False, 
                                                                    num_samples=args.num_samples, 
                                                                    batchsize=args.batchsize)
            
            self.running_logger.info("Calculating ref feats:")
            ref_feats, ref_idxs = self.feature_extractor.get_features_and_idxes(ref_dataset, 
                                                                    name = 'ref_' + self.save_feats_name, 
                                                                    recompute=False, 
                                                                    num_samples=args.num_samples, 
                                                                    batchsize=args.batchsize)
        
        self.running_logger.info("number of test feature: {}".format(len(test_feats)))
        self.running_logger.info("number of ref feature: {}".format(len(ref_feats)))

        visualize_novel_mode_by_eigenvectors_rff(test_feats, ref_feats, test_dataset, test_idxs, args)

        



