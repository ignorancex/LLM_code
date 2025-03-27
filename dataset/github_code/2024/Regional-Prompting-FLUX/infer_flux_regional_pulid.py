import torch
from pipeline_flux_regional_pulid import RegionalFluxPipeline_PULID, RegionalFluxAttnProcessor2_0

if __name__ == "__main__":
    
    model_path = "black-forest-labs/FLUX.1-dev"
    
    pipeline = RegionalFluxPipeline_PULID.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    
    attn_procs = {}
    for name in pipeline.transformer.attn_processors.keys():
        if 'transformer_blocks' in name and name.endswith("attn.processor"):
            attn_procs[name] = RegionalFluxAttnProcessor2_0()
        else:
            attn_procs[name] = pipeline.transformer.attn_processors[name]
    pipeline.transformer.set_attn_processor(attn_procs)

    # load pulid
    pipeline.load_pulid_models()
    pipeline.load_pretrain()

    # single-person example

    # generation settings
    image_width = 1280
    image_height = 1280
    num_samples = 1
    num_inference_steps = 24
    guidance_scale = 3.5
    seed = 124

    # regional prompting settings
    mask_inject_steps = 10
    double_inject_blocks_interval = 1
    single_inject_blocks_interval = 1
    base_ratio = 0.2

    # regional prompting settings
    base_prompt = "In a classroom during the afternoon, a man is practicing guitar by himself, with sunlight beautifully illuminating the room"
    background_prompt = "empty classroom"
    regional_prompt_mask_pairs = {
        "0": {
            "description": "A man in a blue shirt and jeans, playing guitar",
            "mask": [64, 320, 448, 1280]
        }
    }

    # pulid input 
    id_image_paths = ["./assets/lecun.jpeg"]
    id_weights = [1.0] # scale for pulid embedding injection


    # multi-person example

    # generation settings
    # image_width = 1280
    # image_height = 968
    # num_samples = 1
    # num_inference_steps = 24
    # guidance_scale = 3.5
    # seed = 124

    # regional prompting settings

    # mask_inject_steps = 8
    # double_inject_blocks_interval = 1
    # single_inject_blocks_interval = 2
    # base_ratio = 0.1

    # base_prompt = "In an elegant dining room, two men are having dinner at opposite ends of a long formal table, with warm lighting creating an atmospheric ambiance"
    # background_prompt = "a dining room"
    # regional_prompt_mask_pairs = {
    #     "0": {
    #         "description": "A man in a suit sitting at the table, with a plate of food and wine glass in front of him",
    #         "mask": [64, 128, 320, 968]
    #     },
    #     "1": {
    #         "description": "A man in a suit sitting at the table, with a plate of food and wine glass in front of him",
    #         "mask": [960, 128, 1216, 968]
    #     }
    # }
    # # pulid input 
    # id_image_paths = ["./assets/trump.jpg", "./assets/musk.jpg"]
    # id_weights = [0.8, 0.8] # scale for pulid embedding injection

    # prepare regional prompts and masks
    # ensure image width and height are divisible by the vae scale factor
    image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
    image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor

    regional_prompts = []
    regional_masks = []
    background_mask = torch.ones((image_height, image_width))

    for region_idx, region in regional_prompt_mask_pairs.items():
        description = region['description']
        mask = region['mask']
        x1, y1, x2, y2 = mask

        mask = torch.zeros((image_height, image_width))
        mask[y1:y2, x1:x2] = 1.0

        background_mask -= mask

        regional_prompts.append(description)
        regional_masks.append(mask)
            
    # if regional masks don't cover the whole image, append background prompt and mask
    if background_mask.sum() > 0:
        regional_prompts.append(background_prompt)
        regional_masks.append(background_mask)

    # setup regional kwargs that pass to the pipeline
    joint_attention_kwargs = {
        'regional_prompts': regional_prompts,
        'regional_masks': regional_masks,
        'double_inject_blocks_interval': double_inject_blocks_interval,
        'single_inject_blocks_interval': single_inject_blocks_interval,
        'base_ratio': base_ratio,
        'id_image_paths': id_image_paths,
        'id_weights': id_weights,
        'id_masks': regional_masks[:len(id_image_paths)], # use foreground mask as id mask
    }
   
    images = pipeline(
        prompt=base_prompt,
        num_samples=num_samples,
        width=image_width, height=image_height,
        mask_inject_steps=mask_inject_steps,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        joint_attention_kwargs=joint_attention_kwargs,
    ).images

    for idx, image in enumerate(images):
        image.save(f"output_{idx}.jpg")
