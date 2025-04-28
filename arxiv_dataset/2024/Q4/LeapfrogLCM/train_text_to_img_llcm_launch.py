import os
import subprocess


os.environ['MODEL_DIR']=f"path/to/weights"
os.environ['OUTPUT_DIR']=f"path/to/fine-tuned/weights"


command = (
    "accelerate launch "
    "./train_text_to_img_llcm.py "
    "--pretrained_teacher_model=$MODEL_DIR "
    "--output_dir=$OUTPUT_DIR "
    "--mixed_precision=fp16 "
    "--resolution=512 "
    "--learning_rate=8e-6 --loss_type='huber' --ema_decay=0.95 --adam_weight_decay=0.0 "
    "--max_train_samples=18117 "    
    "--dataloader_num_workers=8 "
    "--train_shards_path_or_url='path/to/training/data"
    "--validation_steps=500 "
    "--checkpointing_steps=1000 --checkpoints_total_limit=5 " 
    "--train_batch_size=128 "
    "--gradient_checkpointing --enable_xformers_memory_efficient_attention "
    "--gradient_accumulation_steps=1 "
    "--use_8bit_adam "
    "--resume_from_checkpoint=latest "
    "--seed=453645634 "
    "--max_train_steps=10000"
)

subprocess.run(command, shell=True)





