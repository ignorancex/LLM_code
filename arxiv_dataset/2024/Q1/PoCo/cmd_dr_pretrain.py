
import os



ckpt_name = 'DR_pretrain_new'
cmd = 'python main_pretrain.py ' \
      '--details use_dr-train_only_PoCaco4_256a_model_with_size_to_320_crop_to_256_iteration_8w_all_fold_64-8-1 ' \
      '--gpu_id 0 ' \
      '--batch_size 64 ' \
      '--exp_name only_PoCaco4_256a ' \
      '--model_name resnet18p ' \
      '--loss_name CELoss ' \
      '--data_name DR_pretrain ' \
      '--optim Adam ' \
      '--lr 0.0001 ' \
      '--num_classes 5 ' \
      '--max_iter 80000 ' \
      '--stepvalues 50000 ' \
      '--warmup_steps 1000 ' \
      '--val_freq 400 ' \
      '--display_freq 50 ' \
      '--k_fold -1 ' \
      '--polar 256 ' \
      '--pretrain ' \
      '--save_folder /.../result/PoCo/save_model/checkpoint_{ckpt_name}/ ' \
      '--save_log /.../result/PoCo/save_log/logs_{ckpt_name}/ '.format(ckpt_name=ckpt_name)
print(cmd)
os.system(cmd)
