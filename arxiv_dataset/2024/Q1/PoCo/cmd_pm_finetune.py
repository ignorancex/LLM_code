
import os



ckpt_name = 'pm_finetune'
cmd = 'python main_finetune.py ' \
      '--details use_dr-train_only_PoCaco4_256a_model_to_fine-tune_on_pm ' \
      '--gpu_id 4 ' \
      '--batch_size 64 ' \
      '--exp_name PoCaco4_256a_pm_finetune1_fold4 ' \
      '--model_name resnet18f ' \
      '--loss_name CELoss ' \
      '--data_name pm ' \
      '--optim Adam ' \
      '--lr 0.0001 ' \
      '--num_classes 2 ' \
      '--max_iter 10000 ' \
      '--stepvalues 6000 ' \
      '--warmup_steps 50 ' \
      '--val_freq 15 ' \
      '--display_freq 10 ' \
      '--k_fold 4 ' \
      '--polar 256 ' \
      '--pretrained /.../result/PoCo/save_model/checkpoint_DR_pretrain_new/only_PoCaco4_256a/resnet18_epoch_81.pth ' \
      '--save_folder /.../result/PoCo/save_model/checkpoint_{ckpt_name}/ ' \
      '--save_log /.../result/PoCo/save_log/logs_{ckpt_name}/ '.format(ckpt_name=ckpt_name)
print(cmd)
os.system(cmd)
