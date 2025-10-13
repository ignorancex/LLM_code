from diffusers import (
    CogVideoXTransformer3DModel,
    AutoencoderKLCogVideoX
)
import torch

dtype = torch.bfloat16
transformer_path1 = "/private/task/linyijing/CogVideoX1_5/CogVideoX1.5-5B/models--THUDM--CogVideoX1.5-5B/snapshots/5cb8a989ab3ca1223b04b42b01a5314cd3825f32"
model_ori = CogVideoXTransformer3DModel.from_pretrained(
    transformer_path1,
    subfolder="transformer_only_one_image_index",
    torch_dtype=dtype,
)
transformer_path2 = "/private/task/linyijing/CogVideoX1_5/finetune/output-sft/object365/512-last-add-noise-last-loss-lr-1e_4-bs40-26w/checkpoint-9000"
model_ft = CogVideoXTransformer3DModel.from_pretrained(
    transformer_path2,
    subfolder="transformer",
    torch_dtype=dtype,
)


params_ori = {name: param.clone() for name, param in model_ori.named_parameters()}
params_ft = {name: param.clone() for name, param in model_ft.named_parameters()}

for name in params_ori:
    # abs_diff = torch.abs(params_ori[name] - params_ft[name])
    # print(f"Layer: {name}, Absolute Difference: {torch.sum(abs_diff)}")
    rel_diff = torch.norm(params_ori[name] - params_ft[name]) / torch.norm(params_ori[name])
    print(f"Layer: {name}, Relative Difference: {rel_diff}")
    # write diff in text file
    with open('tools/diff_object_last_noise_9000.txt', 'a') as f:
        # f.write(f"Layer: {name}, Absolute Difference: {torch.sum(abs_diff)}\n")
        f.write(f"Layer: {name}, Relative Difference: {rel_diff}\n")