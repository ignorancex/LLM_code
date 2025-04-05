import torch
import copy
import glob
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.metrics import classification_report

from .dataloader import make_data_loader
from .model_loader import load_model
from .trainers import first_eval, class_report
from .analytics import fix_state_dict, df_builder
 

class PIE_analytics:
    
    def __init__(self, base_model_path: str, dataset: str, model: str, CUDA_device: str):
        #self.check_working_directory()
        
        self.base_model_path  = base_model_path
        self.dataset = dataset
        self.model = model
        self.CUDA_device = CUDA_device

        self.load_data_and_models()
        self.setup_torch_device()
        

        self.ground_labels, self.teacher_pred = class_report(self.teacher, self.dataloaders, self.dataset_sizes)
        _, student_pred = class_report(self.student, self.dataloaders, self.dataset_sizes)

        self.report = classification_report(self.ground_labels, self.teacher_pred, target_names=self.class_names, output_dict=True)

        self.dfs = df_builder(self.ground_labels, self.teacher_pred)

        self.populate_dfs()

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
        os.environ["CUDA_VISIBLE_DEVICES"]=self.CUDA_device

        

    def check_working_directory(self):
        directory = os.getcwd.split("/")
        if directory[len(directory) - 1] != "ik-pruning-bias":
            os.chdir(os.environ['HOME'] + "/ik-pruning-bias")

    def load_data_and_models(self):
        PRUNED_MODEL_PATH = 'models/' + self.dataset + '/' + self.model
        STRUCT_MAG_PATH = '/struct_mag/lr_0.0005/alpha_1.0/'
        IK_STRUCT_MAG_PATH = '/ik_struct_mag/lr_0.0005/alpha_0.01/'

        self.dataloaders, self.dataset_sizes, self.class_names = make_data_loader(8, self.dataset)
        self.student, self.teacher = load_model(self.base_model_path, self.dataset, self.model)

        self.mag_45 = glob.glob(PRUNED_MODEL_PATH + STRUCT_MAG_PATH + 'sparsity_0.45/*.pt')
        self.mag_60 = glob.glob(PRUNED_MODEL_PATH + STRUCT_MAG_PATH + 'sparsity_0.6/*.pt')
        self.mag_75 = glob.glob(PRUNED_MODEL_PATH + STRUCT_MAG_PATH + 'sparsity_0.75/*.pt')
        self.mag_90 = glob.glob(PRUNED_MODEL_PATH + STRUCT_MAG_PATH + 'sparsity_0.9/*.pt')
    
        self.mag_models = [(self.mag_45, 45), (self.mag_60, 60), (self.mag_75, 75), (self.mag_90, 90)]
        
        

        self.ik_45 = glob.glob(PRUNED_MODEL_PATH + IK_STRUCT_MAG_PATH + 'sparsity_0.45/*.pt')
        self.ik_60 = glob.glob(PRUNED_MODEL_PATH + IK_STRUCT_MAG_PATH + 'sparsity_0.6/*.pt')
        self.ik_75 = glob.glob(PRUNED_MODEL_PATH + IK_STRUCT_MAG_PATH + 'sparsity_0.75/*.pt')
        self.ik_90 = glob.glob(PRUNED_MODEL_PATH + IK_STRUCT_MAG_PATH + 'sparsity_0.9/*.pt')

        self.ik_models = [(self.ik_45, 45), (self.ik_60, 60), (self.ik_75, 75), (self.ik_90, 90)]

        
    
    def populate_dfs(self):
        for paths, sparsity in self.mag_models:
            self.dfs.add_models(paths=paths, sparsity=sparsity, method='struct_mag', student=self.student, 
                                dataloaders=self.dataloaders, dataset_sizes=self.dataset_sizes, class_names=self.class_names)

        for paths, sparsity in self.ik_models:
            self.dfs.add_models(paths=paths, sparsity=sparsity, method='ik_struct_mag', student=self.student, 
                                dataloaders=self.dataloaders, dataset_sizes=self.dataset_sizes, class_names=self.class_names)

    def setup_torch_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student = self.student.to(self.device)
        self.teacher = self.teacher.to(self.device)
    
    
    def make_catplots(self):
        ACCURACY_FIGURE_DIRECTORY = 'figures/' + self.model + '/Acc_Struct_' + self.dataset
        PIE_FIGURE_DIRECTORY = 'figures/' + self.model + '/PIE_Struct_' + self.dataset

        self.acc_plot = sns.catplot(x='sparsity', y='accuracy', hue='method', kind='point', data=self.dfs.pruning_stats)
        self.acc_plot.savefig(ACCURACY_FIGURE_DIRECTORY + '.jpg', dpi=600)
        self.acc_plot.savefig(ACCURACY_FIGURE_DIRECTORY + '.png', dpi=600)

        self.pie_fig = sns.catplot(x='sparsity', y='PIEs', hue='method', kind='point', data=self.dfs.PIEs)
        self.pie_fig.savefig(PIE_FIGURE_DIRECTORY + '.jpg', dpi=600)
        self.pie_fig.savefig(PIE_FIGURE_DIRECTORY + '.png', dpi=600)

    def display_results(self):
        self.report
