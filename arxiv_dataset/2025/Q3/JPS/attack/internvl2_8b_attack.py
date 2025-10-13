import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .base_attack import BaseAttack
from utils.string_utils import autodan_SuffixManager
import torch
import torchvision.transforms as T

from model.InternVL2_8B.conversation import get_conv_template

import logging
import time
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import json

class Internvl28BAttack(BaseAttack):
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
        generation_config = dict(max_new_tokens=64, do_sample=False)
        IMAGENET_MEAN = self.config['preprocessing']['image_mean']
        IMAGENET_STD = self.config['preprocessing']['image_std']
        image_size = (self.config['data_loader']['image_height'], self.config['data_loader']['image_width'])

        model = self.pipeline['model']
        tokenizer = self.pipeline['tokenizer']

        attack_data = data[:self.attack_data_num]

        target_list = [data_dict['target'] for data_dict in attack_data]

        if self.image_type == 'blank':
            question_list = [jailbreak_prompt + data_dict['origin_question'] for data_dict in attack_data]
            if 'image_path' in self.config['data_loader']:
                image_list = [data_dict['image'] for data_dict in attack_data]
            else:
                image_list = [torch.zeros((1, 3, *image_size),device = model.device, dtype=model.dtype)] * len(attack_data)
        elif self.image_type == 'related_image':
            question_list = [jailbreak_prompt + data_dict['rephrased_question'] for data_dict in attack_data]
            image_list = [data_dict['image'] for data_dict in attack_data]

        ori_images = torch.cat(image_list, dim=0).detach().to(model.device)

        template = get_conv_template(model.template)
        template.system_message = model.system_message

        suffix_manager_list = []

        IMG_START_TOKEN='<img>'
        IMG_END_TOKEN='</img>'
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * 1 + IMG_END_TOKEN

        prompt_list = []
        for id in range(len(question_list)):
            instruction = '<image>\n' + question_list[id]

            instruction = instruction.replace('<image>', image_tokens, 1)
            suffix_manager_list.append(autodan_SuffixManager(tokenizer=tokenizer,
                                                    conv_template=template,
                                                    instruction=instruction,
                                                    target=target_list[id],
                                                    adv_string=None))
        
        for suffix_manager in suffix_manager_list:
            input_ids = suffix_manager.get_image_input_ids(adv_string=None).to(model.device)
            prompt = suffix_manager.get_prompt(adv_string=None)
            prompt_list.append(prompt)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(prompt_list, return_tensors='pt', padding=True)
        input_ids_list = model_inputs['input_ids'].to(model.device)
        attention_mask = model_inputs['attention_mask'].to(model.device)

        num_steps = self.attack_steps

        eps = None
        if self.attack_type in ['full','grid'] and self.perturbation_constraint != 'null':
            eps = eval(self.perturbation_constraint)

        alpha=eval(self.learning_rate)
        normalized_min = [(0 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
        normalized_max = [(1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]

        range_values = [max_val - min_val for min_val, max_val in zip(normalized_min, normalized_max)]
        range_tensor = torch.tensor(range_values, device = model.device, dtype = model.dtype).view(1, 3, 1, 1)
        alpha = range_tensor * alpha
        if eps is not None:
            eps = range_tensor * eps

        min_values = torch.tensor(normalized_min, device = model.device, dtype = ori_images.dtype)
        max_values = torch.tensor(normalized_max, device = model.device, dtype = ori_images.dtype)

        min_values = min_values.view(1, 3, 1, 1)
        max_values = max_values.view(1, 3, 1, 1)


        # 开始攻击
        logging.info(f'{self.attack_type} Start attacking:')
        data_size, _, img_height, img_width = ori_images.shape

        if self.attack_type in ['full','grid']:
            # 第一种攻击：从全零图像开始
            if pre_adv_img_path is None:
                adv_example = torch.zeros((1, 3, *image_size),device = model.device, dtype=ori_images.dtype, requires_grad=True)
            else:
                pre_adv_img = self.load_image(pre_adv_img_path)
                adv_example = pre_adv_img - ori_images[0:1]

            momentum = torch.zeros((1, 3, *image_size),device = model.device, dtype=ori_images.dtype)

        elif self.attack_type == 'patch':
            patch_height = self.config['attack_config']['patch_size']['height']
            patch_width = self.config['attack_config']['patch_size']['width']
            adv_example = torch.zeros((1, 3, patch_height, patch_width),device = model.device, dtype=ori_images.dtype, requires_grad=True)
        else:
            raise ValueError("无效的攻击类型")

        num_batches = (data_size + self.batch_size - 1) // self.batch_size  # 计算总批次数
        
        min_total_target_loss = float('inf')
        best_epoch_info = {}

        for i in range(num_steps):
            epoch_start_time = time.time()
            
            if self.attack_type in ['full', 'grid']:
                adv_images = ori_images.clone()
                adv_images = torch.clamp(adv_images + adv_example , min=min_values, max=max_values).detach_()
            elif self.attack_type == 'patch':
                adv_images = ori_images.clone()

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
            
                image_flags = torch.ones(current_batch_size)
                output = model(
                    pixel_values=temp_adv_images,
                    input_ids=input_ids_list[start_idx:end_idx],
                    attention_mask=attention_mask[start_idx:end_idx],
                    image_flags=image_flags,
                )

                output_logits = output['logits']

                crit = nn.CrossEntropyLoss(reduction='none')
                target_loss_list = []

                # 只使用对应子批次的suffix_manager
                for id in range(start_idx, end_idx):
                    suffix_manager = suffix_manager_list[id]
                    loss_slice = slice(suffix_manager._target_slice.start - 1, suffix_manager._target_slice.stop - 1)

                    valid_output_logits = output_logits[id - start_idx][attention_mask[id] == 1]  # 注意索引调整
                    valid_input_ids = input_ids_list[id][attention_mask[id] == 1]
                    target_loss = crit(valid_output_logits[loss_slice, :], valid_input_ids[suffix_manager._target_slice])
                    target_loss = target_loss.mean(dim=-1)
                    target_loss_list.append(target_loss)

                stacked_target_loss = torch.stack(target_loss_list)
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
                for id, suffix_manager in enumerate(suffix_manager_list):
                    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
                    generation_config['eos_token_id'] = eos_token_id
                    valid_input_ids = input_ids_list[id][attention_mask[id] == 1]
                    generation_output = model.generate(
                        pixel_values=adv_images[id].unsqueeze(0),
                        input_ids=valid_input_ids[:suffix_manager._assistant_role_slice.stop].unsqueeze(0),
                        **generation_config
                    )
                    gen_str  = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
                    if gen_str[:len(target_list[id])] == target_list[id]:
                        is_success=True
                    else:
                        is_success=False
                    success_num+=is_success
                    logging.info("**********************")
                    logging.info(f"Current Response{id}:\n{gen_str}\n")
                    logging.info("**********************")
            

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
                
                adv_example = (torch.clamp(ori_images[0] + adv_example , min=min_values, max=max_values) - ori_images[0]).detach_()
            
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
        
        logging.info(
            f"Best Epoch Information:\n"
            f"Current Epoch: {best_epoch_info['Current Epoch']}\n"
            f"Passed: {best_epoch_info['Passed']}\n"
            f"Target Loss: {best_epoch_info['Target Loss']}\n"
        )

        norm_mean = torch.tensor(IMAGENET_MEAN)
        norm_std = torch.tensor(IMAGENET_STD)
        if self.attack_type == 'full':
            result_image = (best_epoch_info['Adv Example'] + ori_images[0]).cpu() * norm_std[:,None,None] + norm_mean[:,None,None]
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

        pixel_values = transform(image).unsqueeze(0).to(self.pipeline['model'].dtype).cuda()
        return pixel_values