from transformers import AutoModelForCausalLM, AutoConfig
import torch

# 加载原始预训练模型
model_path = 'model/Llama-2-7b-hf' # you can change the path to your local model path
pretrained_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=None,
    trust_remote_code=True,
)
pretrained_state_dict = pretrained_model.state_dict()

# 配置要在其后扩展新层的层索引, id=0~31 for 32-layer LLMs
expansion_indices = [1, 3, 5, 7, 9, 11, 13, 15]# expand to a 40-layer LLM: (2-1)*8|16
# 初始化新的状态字典和新层索引
new_state_dict = {}
original_layers_count = 32
new_layer_index = 0

for original_layer_index in range(original_layers_count):
    # 第1步：直接复制当前层参数
    for key, value in pretrained_state_dict.items():
        if f".layers.{original_layer_index}." in key:
            new_key = key.replace(f".layers.{original_layer_index}.", f".layers.{new_layer_index}.")
            new_state_dict[new_key] = value
    new_layer_index += 1

    # 第2步：检查是否需要在当前层后插入新层并插值初始化
    if original_layer_index in expansion_indices:
        # 设置线性插值（取前一层和后一层的平均作为新层初始化）
        if original_layer_index < original_layers_count - 1:  # 避免处理最后一层时越界
            next_layer_index = original_layer_index + 1
            for key, value in pretrained_state_dict.items():
                if f".layers.{original_layer_index}." in key:
                    new_key = key.replace(f".layers.{original_layer_index}.", f".layers.{new_layer_index}.")
                    next_key = key.replace(f".layers.{original_layer_index}.", f".layers.{next_layer_index}.")
                    if 'down_proj' in key or 'o_proj' in key:
                        new_state_dict[new_key] = torch.zeros_like(value)
                    else:
                        if next_key in pretrained_state_dict:
                            next_value = pretrained_state_dict[next_key]
                            # 计算前后层权重的平均值
                            interpolated_value = (value + next_value) / 2
                            new_state_dict[new_key] = interpolated_value
                        else:
                            # 如果没有下一层（仅理论上），仍使用当前层权重
                            new_state_dict[new_key] = value
        new_layer_index += 1


for key in 'model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight':
    if key in pretrained_state_dict:
        new_state_dict[key] = pretrained_state_dict[key]
        print('new_state_dict[key] type',type(new_state_dict[key]))
        

new_config = AutoConfig.from_pretrained('llama2_merge_models/Llama-2-7b-col-merge-interp-init', trust_remote_code=True)
new_model = AutoModelForCausalLM.from_config(new_config, trust_remote_code=True)
# 加载新的权重字典到新模型
new_model.load_state_dict(new_state_dict)
# 保存新模型权重到文件
new_model.save_pretrained('llama2_merge_models/Llama-2-7b-col-merge-interp-init')

print('done!')