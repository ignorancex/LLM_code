import argparse
import sys

parser = argparse.ArgumentParser(description='Options for Diffusion4TGIR')

#########################
# Load Template
#########################
parser.add_argument('--config_path', type=str, default='config/fashionIQ_BLIP_config.json', help='config json path')

#########################
# Dataset / DataLoader Settings
#########################
parser.add_argument('--dataset', type=str, default='fashionIQ', help='Dataset')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
parser.add_argument('--num_workers', type=int, default=16, help='The Number of Workers')
parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle Dataset')

parser.add_argument('--pretrain_dataset', type=str, default="None", help='Pretraining dateset')
parser.add_argument('--image_transforms', type=str, default="None", help="Define the image transforms adapt pretrain model")
parser.add_argument('--text_transforms', type=str, default="None", help="Define the text transforms adapt pretrain model")

#########################
# Image Transform Settings
#########################
parser.add_argument('--img_size', type=int, default=224, help='Image Size')

#########################
# Loss Settings
parser.add_argument('--metric_loss', type=str, default="batch_based_classification_loss", help='Metric Loss Code')
#########################

#########################
# Encoder Settings
#########################
# Pretraining Model Settings, only in effect when pretrain_dataset is not None
parser.add_argument("--image_encoders", type=str, default="None", help="Image Encoder")
parser.add_argument("--text_encoders", type=str, default="None", help="Text Encoder")

# mask settings, effective in pretrainer when pretrain_dataset is not None
parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask Ratio of Image')
parser.add_argument('--img_weight', type=float, default=0.5, help='Image Weight')
# BLIP2 Settings
# ==================================================
# Architectures                  Types
# ==================================================
# blip2_opt                      pretrain_opt2.7b, caption_coco_opt2.7b, pretrain_opt6.7b, caption_coco_opt6.7b
# blip2_t5                       pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
# blip2                          pretrain, coco
parser.add_argument('--blip_model_name', type=str, default='None', help='BLIP2 Model Name')
parser.add_argument('--blip_model_type', type=str, default='pretrain', help='BLIP2 Model Type')

# Text Inversion Settings
parser.add_argument('--text_inversion_model', type=str, default='None', help='Text Inversion Model')
parser.add_argument('--text_inversion_model_type', type=str, default='pretrain_flant5xl', help='Text Inversion Model Type')

# ==================================================
# CLIP Settings
parser.add_argument('--clip_image_model', type=str, default="None", help='CLIP Image Model, None means not use')
parser.add_argument('--clip_text_model', type=str, default="None", help='CLIP Text Model, None means not use')
# Diffusion Settings
parser.add_argument('--diffusion_model_id_or_path', type=str, default='None', help='Diffusion Model Id or Path')
parser.add_argument('--strength', type=float, default=0.5, help='Diffusion Strength')
parser.add_argument('--num_inference_steps', type=int, default=2, help='Number of Inference Steps')
parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance Scale')
parser.add_argument('--prompt', type=str, default='a image', help='Prompt for target image')
parser.add_argument('--output_type', type=str, default='image', help='Output Type')
#########################
# Composition Model Settings
#########################
parser.add_argument('--compositor', type=str, default='None', help='Composition Model')
#########################
# Optimizer Settings
#########################
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'RAdam', 'AdamW'], help='Optimizer')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='l2 regularization lambda (default: 5e-5)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--warmup_iters', type=int, default=5, help='num epochs for warmup learning rate')
parser.add_argument('--decay_step', type=int, default=35, help='num epochs for first decaying learning rate')
parser.add_argument('--decay_step_second', type=int, default=45, help='num epochs for second decaying learning rate')
parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay gamma')
parser.add_argument('--lr_scheduler', type=str, default='MultiStepWithWarmup', help='learning rate scheduler')

#########################
# Logging Settings
#########################
parser.add_argument('--topk', type=str, default='1,5,10,50', help='topK recall for evaluation')
parser.add_argument('--best_metric', type=str, default='10', help='best topK recall metric for saving checkpoint')
parser.add_argument('--wandb_project_name', type=str, default='zscir', help='Weights & Biases project name')
parser.add_argument('--wandb_account_name', type=str, default='junyang', help='Weights & Biases account name')

#########################
# Resume Training
#########################
parser.add_argument('--checkpoint_path', type=str, default='', help='Path to saved checkpoint file')
# trainer
parser.add_argument('--trainer', type=str, default='diffusion_trainer', help='Trainer')
parser.add_argument('--epoch', type=int, default=80, help='epoch (default: 80)')
# only work in pretrainer
parser.add_argument('--tuning_module', type=str, default="both", help="Which pretrained encoder to fine-tune")
#########################
# Misc
#########################
parser.add_argument('--device_idx', type=str, default='0', help='Gpu idx')
parser.add_argument('--random_seed', type=int, default=42, help=
                    'Random seed value is greater than or equal to -1, which is equal to -1 getting random value from non-negative integer')
parser.add_argument('--experiment_dir', type=str, default='experiments', help='Experiment save directory')
parser.add_argument('--experiment_description', type=str, default='NO', help='Experiment description')
parser.add_argument("--evaluator_code", type=str, default="diffusion_evaluator", help="Evaluator code")


def _get_user_defined_arguments(argvs):
    prefix, conjugator = '--', '='
    return [argv.replace(prefix, '').split(conjugator)[0] for argv in argvs]


def load_config_from_command():
    user_defined_argument = _get_user_defined_arguments(sys.argv[1:])

    configs = vars(parser.parse_args())
    user_defined_configs = {k: v for k, v in configs.items() if k in user_defined_argument}
    return configs, user_defined_configs
