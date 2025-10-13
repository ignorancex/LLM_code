import pandas as pd
import numpy as np
from datasets import BaseMultiTaskDataset
import os

def create_global_dataset(args, dest_tasks):
    global_rgb_paths = pd.read_csv(os.path.join(args.data_path, "filtered_file_paths.csv"))
    global_dest_task_labels = {}
    for task in dest_tasks:
        if task == 'class_scene':
            top_classes = global_rgb_paths.class_idx.value_counts().index[:16]
            global_rgb_paths['class_idx'] = global_rgb_paths['class_idx'].map(lambda x: np.where(top_classes == x)[0][0])
            global_dest_task_labels[task] = global_rgb_paths.class_idx.values
        else:
            if task == 'segment_semantic':
                global_dest_task_labels[task] = global_rgb_paths['filepath'].map(lambda x: x.replace('rgb', 'segment_semantic').replace('_semantic.png', 'semantic.png')).values
            elif task == 'principal_curvature_old':
                global_dest_task_labels[task] = global_rgb_paths.filepath.map(lambda x: x.replace('rgb', 'principal_curvature')).values
            else:
                global_dest_task_labels[task] = global_rgb_paths.filepath.map(lambda x: x.replace('rgb', task)).values
    
    return BaseMultiTaskDataset(data_fps=global_rgb_paths.filepath.values, label_fps=global_dest_task_labels, src_task='rgb', dest_tasks=dest_tasks)
    