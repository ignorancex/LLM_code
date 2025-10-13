import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .base_attack import BaseAttack
import torch
import torchvision.transforms as T

from model.InternVL2_8B.conversation import get_conv_template

import logging
import time
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import json

from model.minigpt4.minigpt_utils import prompt_wrapper, generator
from torchvision.utils import save_image
import gc

def attack_loss(model, prompts, targets):
    device = model.llama_model.device

    context_embs = prompts.context_embs

    if len(context_embs) == 1:
        context_embs = context_embs * len(targets) # expand to fit the batch_size

    assert len(context_embs) == len(targets), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(targets)}"

    batch_size = len(targets)
    model.llama_tokenizer.padding_side = "right"

    to_regress_tokens = model.llama_tokenizer(
        targets,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=model.max_txt_len,
        add_special_tokens=False
    ).to(device)
    to_regress_embs = model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

    bos = torch.ones([1, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * model.llama_tokenizer.bos_token_id
    bos_embs = model.llama_model.model.embed_tokens(bos)

    pad = torch.ones([1, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * model.llama_tokenizer.pad_token_id
    pad_embs = model.llama_model.model.embed_tokens(pad)


    T = to_regress_tokens.input_ids.masked_fill(
        to_regress_tokens.input_ids == model.llama_tokenizer.pad_token_id, -100
    )


    pos_padding = torch.argmin(T, dim=1) # a simple trick to find the start position of padding

    input_embs = []
    targets_mask = []

    target_tokens_length = []
    context_tokens_length = []
    seq_tokens_length = []

    for i in range(batch_size):

        pos = int(pos_padding[i])
        if T[i][pos] == -100:
            target_length = pos
        else:
            target_length = T.shape[1]

        targets_mask.append(T[i:i+1, :target_length])
        input_embs.append(to_regress_embs[i:i+1, :target_length]) # omit the padding tokens

        context_length = context_embs[i].shape[1]
        seq_length = target_length + context_length

        target_tokens_length.append(target_length)
        context_tokens_length.append(context_length)
        seq_tokens_length.append(seq_length)

    max_length = max(seq_tokens_length)

    attention_mask = []

    for i in range(batch_size):

        # masked out the context from loss computation
        context_mask =(
            torch.ones([1, context_tokens_length[i] + 1],
                    dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )

        # padding to align the length
        num_to_pad = max_length - seq_tokens_length[i]
        padding_mask = (
            torch.ones([1, num_to_pad],
                    dtype=torch.long).to(device).fill_(-100)
        )

        targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
        input_embs[i] = torch.cat( [bos_embs, context_embs[i], input_embs[i],
                                    pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
        attention_mask.append( torch.LongTensor( [[1]* (1+seq_tokens_length[i]) + [0]*num_to_pad ] ) )

    targets = torch.cat( targets_mask, dim=0 ).to(device)
    inputs_embs = torch.cat( input_embs, dim=0 ).to(device)
    attention_mask = torch.cat(attention_mask, dim=0).to(device)


    outputs = model.llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
    loss = outputs.loss

    return loss

class Minigpt413BAttack(BaseAttack):
    def __init__(self, config, pipeline):
        self.config=config
        self.pipeline = pipeline
        
        self.attack_data_num = config['attack_config']['attack_data_num']
        self.attack_steps = config['attack_config']['attack_steps']
        self.min_loss_limit = config['attack_config']['min_loss_limit']
        self.batch_size = config['attack_config']['batch_size']
        self.attack_type = config['attack_config']['attack_type']
        self.perturbation_constraint = config['attack_config']['perturbation_constraint']
        self.learning_rate= config['attack_config']['learning_rate']
        self.generate_for_test_num = config['attack_config']['generate_for_test_num']
        self.adv_save_dir = config['attack_config']['adv_save_dir']
        self.annotation = config['attack_config']['annotation']
        self.revision_iteration_num = config['attack_config']['revision_iteration_num']

        self.image_type = config['data_loader']['image_type']

    
    def attack(self, data, pre_adv_img_path = None, jailbreak_prompt = '', iteration_num = 0):
        model = self.pipeline['model']
        tokenizer = self.pipeline['tokenizer']
        vis_processor = self.pipeline['vis_processor']

        my_generator = generator.Generator(model=model)
        my_generator.max_new_tokens = 64

        IMAGENET_MEAN = self.config['preprocessing']['image_mean']
        IMAGENET_STD = self.config['preprocessing']['image_std']
        image_size = (self.config['data_loader']['image_height'], self.config['data_loader']['image_width'])

        model = self.pipeline['model']
        tokenizer = self.pipeline['tokenizer']

        text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt

        attack_data = data[:self.attack_data_num]

        target_list = [data_dict['target'] for data_dict in attack_data]

        if self.image_type == 'blank':
            question_list = [jailbreak_prompt + data_dict['origin_question'] for data_dict in attack_data]
            if 'image_path' in self.config['data_loader']:
                image_list = [data_dict['image'].to('cpu') for data_dict in attack_data]
            else:
                image_list = [torch.zeros((1, 3, *image_size),device = model.llama_model.device, dtype=model.llama_model.dtype)] * len(attack_data)
        elif self.image_type == 'related_image':
            question_list = [jailbreak_prompt + data_dict['rephrased_question'] for data_dict in attack_data]
            image_list = [data_dict['image'].to('cpu') for data_dict in attack_data]

        # ori_images = [vis_processor(image).unsqueeze(0).to(model.llama_model.device) for image in image_list]
        ori_images = torch.cat(image_list, dim=0).detach().to('cpu')
        ori_images_one = ori_images[0].clone().detach().to(model.llama_model.device)

        num_steps = self.attack_steps

        eps = None
        if self.attack_type in ['full','grid'] and self.perturbation_constraint != 'null':
            eps = eval(self.perturbation_constraint)

        alpha=eval(self.learning_rate)
        normalized_min = [(0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
        normalized_max = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

        range_values = [max_val - min_val for min_val, max_val in zip(normalized_min, normalized_max)]
        range_tensor = torch.tensor(range_values, device = model.llama_model.device, dtype = model.llama_model.dtype).view(1, 3, 1, 1)
        alpha = range_tensor * alpha
        if eps is not None:
            eps = range_tensor * eps

        min_values = torch.tensor(normalized_min, device = model.llama_model.device, dtype = model.llama_model.dtype)
        max_values = torch.tensor(normalized_max, device = model.llama_model.device, dtype = model.llama_model.dtype)

        min_values = min_values.view(1, 3, 1, 1)
        max_values = max_values.view(1, 3, 1, 1)


        # 开始攻击
        logging.info(f'{self.attack_type} Start attacking:')
        data_size, _, img_height, img_width = ori_images.shape

        if self.attack_type in ['full','grid']:
            # 第一种攻击：从全零图像开始
            if pre_adv_img_path is None:
                adv_example = torch.zeros((1, 3, *image_size),device = model.llama_model.device, dtype=model.llama_model.dtype, requires_grad=True)
            else:
                pre_adv_img = self.load_image(pre_adv_img_path)
                adv_example = pre_adv_img - ori_images_one.unsqueeze(0)

            momentum = torch.zeros((1, 3, *image_size),device = model.llama_model.device, dtype=model.llama_model.dtype)

        elif self.attack_type == 'patch':
            patch_height = self.config['attack_config']['patch_size']['height']
            patch_width = self.config['attack_config']['patch_size']['width']
            adv_example = torch.zeros((1, 3, patch_height, patch_width),device = model.llama_model.device, dtype=model.llama_model.dtype, requires_grad=True)
        else:
            raise ValueError("无效的攻击类型")

        num_batches = (data_size + self.batch_size - 1) // self.batch_size  # 计算总批次数
        
        min_total_target_loss = float('inf')
        best_epoch_info = {}

        for i in range(num_steps):
            epoch_start_time = time.time()
            
            if self.attack_type in ['full', 'grid']:
                adv_images = ori_images.clone().to(model.llama_model.device)
                adv_images = torch.clamp(adv_images + adv_example , min=min_values, max=max_values).detach_()
            elif self.attack_type == 'patch':
                adv_images = ori_images.clone().to(model.llama_model.device)

                adv_patch_expanded = adv_example.expand(data_size, -1, -1, -1)
                adv_images[:, :, :patch_height, :patch_width] = adv_patch_expanded
                adv_images = adv_images.detach()

            accumulated_grad = torch.zeros_like(adv_example)

            total_target_loss = 0
            all_finished = True

            for batch_idx in range(num_batches):
                # 获取当前小批次数据
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, data_size)
                current_batch_size = end_idx - start_idx

                # 如果当前小批次是空的，跳过
                if current_batch_size == 0:
                    continue
                
                temp_adv_images = adv_images[start_idx:end_idx].clone().detach().requires_grad_(True)
                # 处理当前小批次
                if temp_adv_images.grad is not None:
                    temp_adv_images.grad.zero_()
                temp_adv_images.requires_grad_().retain_grad()

                text_prompts = [text_prompt_template % attack_question for attack_question in question_list[start_idx:end_idx]]
                batch_images = [[x.unsqueeze(0)] for x in temp_adv_images]

                prompt = prompt_wrapper.Prompt(model=model, text_prompts=text_prompts, img_prompts=batch_images)
                prompt.update_context_embs()

                stacked_target_loss = attack_loss(model, prompt, target_list[start_idx:end_idx])

                mask = stacked_target_loss > self.min_loss_limit

                selected_target_loss = stacked_target_loss[mask]

                target_weight=0
                if selected_target_loss.numel() > 0:
                    target_loss = selected_target_loss.mean()
                    target_weight=1
                    all_finished = False
                else:
                    target_loss = torch.mean(stacked_target_loss)

                target_loss = target_weight * target_loss
                target_grad = torch.autograd.grad(target_loss,temp_adv_images,retain_graph=True)[0]

                total_target_loss += float(torch.sum(stacked_target_loss))

                if self.attack_type in ['full','grid']:
                    accumulated_grad += target_grad.sum(dim=0)
                elif self.attack_type == 'patch':
                    grad = target_grad
                    grad_patch = grad[:, :, :patch_height, :patch_width]
                    accumulated_grad += grad_patch.sum(dim=0)



            success_num = 0

            total_target_loss /= data_size
            
            if i > num_steps - self.generate_for_test_num:
            # if i > num_steps - self.generate_for_test_num or all_finished:
                batch_size = self.config['inference_config']['batch_size']

                for index in range(0, len(question_list), batch_size):  # Add tqdm here
                    batch_texts = question_list[index:index+batch_size]
                    batch_images = adv_images[index:index+batch_size]

                    text_prompts = [text_prompt_template % attack_question for attack_question in batch_texts]
                    batch_images = [[x.unsqueeze(0)] for x in batch_images]

                    prompt = prompt_wrapper.Prompt(model=model, text_prompts=text_prompts, img_prompts=batch_images)
                    prompt.update_context_embs()

                    with torch.no_grad():
                        batch_responses, _ = my_generator.generates(prompt)

                    for id in range(len(batch_texts)):
                        if batch_responses[id][:len(target_list[index + id])] == target_list[index + id]:
                            is_success=True
                            success_num+=1
                        else:
                            is_success=False
                        success_num+=is_success
                        print("**********************")
                        print(f"Current Response{index + id}:\n{batch_responses[id]}\n")
                        print("**********************")

                
                

            epoch_end_time = time.time()
            epoch_cost_time = round(epoch_end_time - epoch_start_time, 2)
            logging.info(
                "################################\n"
                f"Current Epoch: {i}/{num_steps}\n"
                f"Passed:{success_num}/{data_size}\n"
                f"Target Loss:{total_target_loss}\n"
                # f"Total Loss:{total_all_loss}\n"
                f"Epoch Cost:{epoch_cost_time}\n"                
                "################################\n")
            
            if total_target_loss < min_total_target_loss:
                min_total_target_loss = total_target_loss
                best_epoch_info = {
                    "Current Epoch": f"{i}/{num_steps}",
                    "Passed": f"{success_num}/{data_size}",
                    "Target Loss": total_target_loss,
                    "Epoch Cost": epoch_cost_time,
                    "Adv Example": adv_example[0], # 保存adv_example[0]
                }
            
            if all_finished:
                break
            
            beta = 0.9
            avg_grad = accumulated_grad / data_size

            grad_flat = avg_grad.view(avg_grad.size(0), -1)
            grad_norm = torch.norm(grad_flat, p=2, dim=1).view(-1, *[1 for _ in range(avg_grad.dim()-1)])
            grad_norm = torch.clamp(grad_norm, min=1e-8)  # 防止除以零
            normalized_grad = avg_grad / grad_norm

            momentum = beta * momentum + (1 - beta) * normalized_grad
            avg_grad = momentum

            if self.attack_type == 'full':
                adv_example = adv_example - alpha * avg_grad.sign()
                if eps is not None:
                    adv_example = torch.clamp(adv_example, min=-eps, max=eps).detach()
                else:
                    adv_example = torch.clamp(adv_example, min=min_values, max=max_values).detach()
                
                adv_example = (torch.clamp(ori_images_one + adv_example , min=min_values, max=max_values) - ori_images_one).detach_()
            
            elif self.attack_type == 'patch':
                adv_example = adv_example[0] - alpha * avg_grad.sign()

                # 进行范围限制
                adv_example = torch.clamp(adv_example, min=min_values, max=max_values).detach()
            
            elif self.attack_type == 'grid':
                grid_adv_grad = self.average_pool_manual(avg_grad, self.config['attack_config']['grid_size'])

                adv_example = adv_example[0] - alpha * grid_adv_grad.sign()

                if eps is not None:
                    adv_example = torch.clamp(adv_example, min=-eps, max=eps).detach()
                else:
                    adv_example = torch.clamp(adv_example, min=min_values, max=max_values).detach()

                # 进行范围限制
                adv_example = torch.clamp(adv_example, min=min_values, max=max_values).detach()
            
            torch.cuda.empty_cache()
            gc.collect()
        
        logging.info(
            f"Best Epoch Information:\n"
            f"Current Epoch: {best_epoch_info['Current Epoch']}\n"
            f"Passed: {best_epoch_info['Passed']}\n"
            f"Target Loss: {best_epoch_info['Target Loss']}\n"
        )

        norm_mean = torch.tensor(IMAGENET_MEAN)
        norm_std = torch.tensor(IMAGENET_STD)
        if self.attack_type == 'full':
            result_image = (best_epoch_info['Adv Example'] + ori_images_one).cpu() * norm_std[:,None,None] + norm_mean[:,None,None]
        else:
            result_image = best_epoch_info['Adv Example'].cpu() * norm_std[:,None,None] + norm_mean[:,None,None]
        result_image = result_image * 255
        result_image = result_image.byte()
        img = Image.fromarray(result_image.permute(1, 2, 0).cpu().numpy(), 'RGB')
        adv_img_file_name = self.get_file_name(iteration_num) + '.png'
        adv_img_path = os.path.join(self.adv_save_dir, self.config['data_loader']['dataset_name'],self.config['model']['name'], adv_img_file_name)
        os.makedirs(os.path.dirname(adv_img_path), exist_ok=True)
        img.save(adv_img_path)

        if num_steps>0:
            return adv_img_path, i
        else:
            return adv_img_path, -1

    def average_pool_manual(self, avg_grad, grid_size=4):
        """
        手动将 avg_grad 分块平均并扩展回原始大小。

        参数:
        - avg_grad: 张量，形状为 [1, 3, 448, 448]
        - grid_size: 块的大小，默认为 4

        返回:
        - 更新后的 avg_grad，形状为 [1, 3, 448, 448]
        """
        # 使用内置的平均池化
        pooled = F.avg_pool2d(avg_grad, kernel_size=grid_size, stride=grid_size)
        
        # 使用插值将池化后的张量扩展回原始尺寸
        modified_avg_grad = F.interpolate(pooled, size=avg_grad.shape[2:], mode='nearest')

        return modified_avg_grad
    
    def get_file_name(self, iteration_num):
            
        filename = None
        if self.attack_type == 'full':
            filename = f"full_con{str(self.perturbation_constraint).replace('/','|')}_lr{str(self.learning_rate).replace('/','|')}_revision{self.revision_iteration_num}_iteration{iteration_num}_{self.annotation}"
        elif self.attack_type == 'patch':
            filename = f"patch_lr{str(self.learning_rate).replace('/','|')}_revision{self.revision_iteration_num}_iteration{iteration_num}_{self.annotation}"
        elif self.attack_type == 'grid':
            filename = f"grid{self.config['attack_config']['grid_size']}_con{str(self.perturbation_constraint).replace('/','|')}_lr{str(self.learning_rate).replace('/','|')}_revision{self.revision_iteration_num}_iteration{iteration_num}_{self.annotation}"
        
        return filename
    
    def load_image(self,image_file, input_size=448, max_num=12):
        image_mean, image_std = self.config['preprocessing']['image_mean'], self.config['preprocessing']['image_std']
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = image_file.convert('RGB')

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Normalize(mean=image_mean, std=image_std)
        ])

        pixel_values = transform(image).unsqueeze(0).to(self.pipeline['model'].llama_model.dtype).cuda()
        return pixel_values