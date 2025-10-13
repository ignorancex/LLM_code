# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict, List, Tuple
import torch
from swift.llm import register_template, to_device
from swift.utils import is_deepspeed_enabled
from swift.llm.template import Template
from swift.llm.template.template.qwen import Qwen2_5VLTemplate, QwenTemplateMeta
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context, findall


class Qwen2_5VL_All2AllTemplate(Qwen2_5VLTemplate):
    TRAIN_HEIGHT = int(os.environ.get('IMAGE_TRAIN_SIZE', 252))
    TOKEN_LOSS_WEIGHT = float(os.environ.get('TOKEN_LOSS_WEIGHT', 3.0))
    IMG_LOSS_WEIGHT = float(os.environ.get('IMG_LOSS_WEIGHT', 3.0))
    USE_IMG_EMBED_AS_INPUT = os.environ.get('USE_IMG_EMBED_AS_INPUT', 'False').lower() == 'true'
    USE_DYNAMIC_RATIO = os.environ.get('USE_DYNAMIC_RATIO', 'False').lower() == 'true'
    CONSISTANT_EDIT_SCALE = os.environ.get('CONSISTANT_EDIT_SCALE', 'False').lower() == 'true'

    def _pre_tokenize_images(self, context_list: List[Context], loss_scale_list: List[float],
                             inputs: StdTemplateInputs) -> Tuple[List[Context], List[float]]:
        res: List[Context] = []
        res_loss_scale: List[float] = []
        inputs.image_idx = 0

        for context, loss_scale in zip(context_list, loss_scale_list):
            if context == '<image>' and inputs.is_multimodal and inputs.image_idx < len(inputs.images):
                c_list = self.replace_tag('image', inputs.image_idx, inputs)
                inputs.image_idx += 1
            else:
                c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

    def generate_image_roles(self, messages):
        label_list = []
        for msg in messages:
            content = msg.get('content', '')
            if content is None:
                continue
            image_count = content.count('<image>')
            if image_count > 0:
                label_list.extend([msg['role']] * image_count)
        return label_list

    def get_target_size(self, height, width):
        train_sizes = [(9, 9), (8, 10), (7, 11), (7, 12), (6, 12)]
        train_ratios = [max(h, w) / min(h, w) for h, w in train_sizes]
        train_patch_size = 28

        input_ratio = max(height, width) / min(height, width)
        diffs = [abs(input_ratio - r) for r in train_ratios]
        min_index = diffs.index(min(diffs))
        if height < width:
            height, width = train_sizes[min_index]
        else:
            width, height = train_sizes[min_index]
        return height * train_patch_size, width * train_patch_size

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        image_roles = self.generate_image_roles(inputs.messages)
        size = (self.TRAIN_HEIGHT, self.TRAIN_HEIGHT)
        for i in range(len(inputs.images)):
            if image_roles[i] == 'assistant' or (self.CONSISTANT_EDIT_SCALE and 'assistant' in image_roles):
                if self.USE_DYNAMIC_RATIO:
                    img_width, img_height = inputs.images[i].size
                    size = self.get_target_size(img_height, img_width)
                inputs.images[i] = inputs.images[i].resize(size)

        encoded = Template._encode(self, inputs)
        processor = self.processor
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        images = inputs.images
        videos = inputs.videos

        for media_type in ['images', 'videos']:
            if locals()[media_type]:
                if media_type == 'images':
                    media_token = self.image_token_id
                    media_inputs = processor.image_processor(
                        images=images, videos=None, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['image_grid_thw']
                else:
                    media_inputs = processor.image_processor(
                        images=None, videos=videos, return_tensors='pt', do_resize=False)
                    media_grid_thw = media_inputs['video_grid_thw']
                    media_token = self.video_token_id
                    if self.version == 'v2_5':
                        from qwen_vl_utils import vision_process
                        media_inputs['second_per_grid_ts'] = [
                            processor.image_processor.temporal_patch_size / vision_process.FPS
                        ] * len(media_grid_thw)
                idx_list = findall(input_ids, media_token)
                added_tokens_len = 0
                for i, idx in enumerate(idx_list):
                    merge_length = processor.image_processor.merge_size**2
                    token_len = (media_grid_thw[i].prod() // merge_length)
                    input_ids = input_ids[:idx
                                          + added_tokens_len] + [media_token] * token_len + input_ids[added_tokens_len
                                                                                                      + idx + 1:]
                    if labels:
                        labels = labels[:idx + added_tokens_len] + [labels[idx + added_tokens_len]
                                                                    ] * token_len + labels[added_tokens_len + idx + 1:]
                        labels[idx + added_tokens_len]
                    added_tokens_len += token_len - 1
                encoded.update(media_inputs)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        return encoded


    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        use_img_embed_as_input = self.USE_IMG_EMBED_AS_INPUT
        token_loss_weight = self.TOKEN_LOSS_WEIGHT
        img_loss_weight = self.IMG_LOSS_WEIGHT

        if not self.is_training:
            return inputs
        input_ids = inputs['input_ids']
        _model = model.model
        if not hasattr(_model, 'embed_tokens'):
            _model = _model.model  # LoRA
        pixel_values = inputs.get('pixel_values')
        pixel_values_videos = inputs.get('pixel_values_videos')
        image_grid_thw = inputs.get('image_grid_thw')
        video_grid_thw = inputs.get('video_grid_thw')
        second_per_grid_ts = inputs.get('second_per_grid_ts')

        inputs_embeds = _model.embed_tokens(input_ids)

        dtype = model.visual.get_dtype() if self.version == 'v2' else model.visual.dtype
        input_dict = {'token_loss_weight': token_loss_weight, 'image_embeddings': None, 'img_loss_weight': img_loss_weight}
        if pixel_values is None and pixel_values_videos is None:  # plain-text
            if is_deepspeed_enabled():
                from PIL import Image
                images = [Image.new('RGB', (32, 32), (0, 0, 0))]
                media_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')
                device = input_ids.device
                media_inputs = to_device(media_inputs, device)
                pixel_values = media_inputs['pixel_values'].type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
                inputs_embeds += image_embeds.mean() * 0.

                num_embeddings = model.image_prefill_embeds.num_embeddings
                image_prefill_embeds = model.image_prefill_embeds(
                    torch.arange(num_embeddings, device=inputs_embeds.device).long()
                )
                inputs_embeds = inputs_embeds + image_prefill_embeds.mean() * 0.
        else:
            if pixel_values is not None:
                pixel_values = pixel_values.type(dtype)
                image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                if not use_img_embed_as_input:
                    # for image to predict, do not use image embeddings as input
                    label_image_mask = (inputs['labels'] == model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    # reshape image embeddings to match mask
                    expanded_image_embeds = torch.zeros_like(inputs_embeds)
                    expanded_image_embeds.masked_scatter_(image_mask, image_embeds)
                    # prepare mask
                    excluded_mask = torch.logical_and(image_mask, label_image_mask)
                    image_mask = torch.logical_and(image_mask, ~label_image_mask).to(inputs_embeds.dtype)
                    # exclude image embeddings from input embeddings
                    image_embeds_labels = expanded_image_embeds.masked_select(excluded_mask).view(-1, image_embeds.size(-1))
                    input_dict.update({'image_embeddings': image_embeds_labels.clone().detach()}) # TODO: may not need detach
                    # get prefill image embeddings
                    num_embeddings = model.image_prefill_embeds.num_embeddings
                    image_prefill_embeds = model.image_prefill_embeds(
                        torch.arange(num_embeddings, device=inputs_embeds.device).long()
                    )
                    num_repeats = image_embeds_labels.size(0) // num_embeddings
                    image_prefill_embeds = image_prefill_embeds.repeat(num_repeats, 1)
                    assert image_prefill_embeds.size(0) == image_embeds_labels.size(0)
                    # replace output image embeddings with prefill image embeddings
                    inputs_embeds.masked_scatter_(excluded_mask, image_prefill_embeds)
                    # replace input image embeddings with image embeddings
                    inputs_embeds = inputs_embeds * (1 - image_mask) + expanded_image_embeds * image_mask
                else:
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(dtype)
                video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        kwargs = {}
        if self.version == 'v2_5':
            kwargs = {'second_per_grid_ts': second_per_grid_ts}
        position_ids, _ = model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask=inputs['attention_mask'], **kwargs)
        input_dict.update({'inputs_embeds': inputs_embeds, 'position_ids': position_ids.contiguous()})
        return input_dict


register_template(
    QwenTemplateMeta(
        template_type='qwenall2all_custom',
        template_cls=Qwen2_5VL_All2AllTemplate,))
