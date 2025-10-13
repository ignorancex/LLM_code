# Copyright (c) Alibaba, Inc. and its affiliates.
from qwen_vl_utils import vision_process
from swift.utils import get_env_args
from swift.llm import Model, ModelGroup, ModelArch, ModelMeta, get_model_tokenizer_multimodal, register_model
from swift.llm.model.patcher import patch_output_clone, patch_output_to_input_device


def patch_qwen_vl_utils(vision_process):
    if hasattr(vision_process, '_patch'):
        return
    for key in [
            'image_factor', 'min_pixels', 'max_pixels', 'max_ratio', 'video_min_pixels', 'video_max_pixels',
            'video_total_pixels', 'frame_factor', 'fps', 'fps_min_frames', 'fps_max_frames'
    ]:
        type_func = float if key == 'fps' else int
        setattr(vision_process, key.upper(), get_env_args(key, type_func, getattr(vision_process, key.upper())))
    _read_video_decord = vision_process._read_video_decord

    def _new_read_video_decord(ele: dict):
        from swift.llm import load_file
        ele['video'] = load_file(ele['video'])
        return _read_video_decord(ele)

    vision_process.VIDEO_READER_BACKENDS['decord'] = _new_read_video_decord
    vision_process._patch = True


def get_model_tokenizer_qwen2_5_all2all(*args, **kwargs):
    import os
    from modeling.ar.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    height = int(os.environ.get('IMAGE_TRAIN_SIZE', 252))
    print(f'IMAGE_TRAIN_SIZE {height}')
    USE_DYNAMIC_RATIO = os.environ.get('USE_DYNAMIC_RATIO', 'False').lower() == 'true'
    print(f'USE_DYNAMIC_RATIO {USE_DYNAMIC_RATIO}')
    CONSISTANT_EDIT_SCALE = os.environ.get('CONSISTANT_EDIT_SCALE', 'False').lower() == 'true'
    print(f'CONSISTANT_EDIT_SCALE {CONSISTANT_EDIT_SCALE}')
    TOKEN_LOSS_WEIGHT = float(os.environ.get('TOKEN_LOSS_WEIGHT', 3.0))
    print(f'TOKEN_LOSS_WEIGHT {TOKEN_LOSS_WEIGHT}')
    IMG_LOSS_WEIGHT = float(os.environ.get('IMG_LOSS_WEIGHT', 3.0))
    print(f'IMG_LOSS_WEIGHT {IMG_LOSS_WEIGHT}')

    kwargs['automodel_class'] = Qwen2_5_VLForConditionalGeneration
    model, tokenizer = get_model_tokenizer_multimodal(*args, **kwargs)
    if model is not None and hasattr(model.model, 'embed_tokens'):
        patch_output_clone(model.model.embed_tokens)
        patch_output_to_input_device(model.model.embed_tokens)

    patch_qwen_vl_utils(vision_process)
    return model, tokenizer


register_model(
    ModelMeta(
        model_type='qwenall2all_custom', 
        model_groups=[
            ModelGroup([
                Model(None, None, 'QwenAll2All'),
            ]),
        ],
        is_multimodal=True,
        model_arch=ModelArch.qwen2_vl,
        template='qwenall2all_custom',
        get_function=get_model_tokenizer_qwen2_5_all2all,
        architectures=['Qwen2_5_VLForConditionalGeneration'],
        requires=['transformers==4.49', 'qwen_vl_utils>=0.0.6', 'decord'],
        tags=['vision']))
