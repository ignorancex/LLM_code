import torch
from diffusers import DDIMScheduler

from rgbx.rgb2x.pipeline_rgb2x import StableDiffusionAOVMatEstPipeline


def predict(image):
    pipe = StableDiffusionAOVMatEstPipeline.from_pretrained("zheng95z/rgb-to-x")
    pipe.to("cuda")

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing"
    )

    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")

    prompts = {
        "normal": "Camera-space Normal",
        "irradiance": "Irradiance (diffuse lighting)",
    }

    generator = torch.Generator(device="cuda").manual_seed(0)

    out = []
    for aov_name, prompt in prompts.items():
        aov = pipe(
            prompt=prompt,
            photo=image,
            num_inference_steps=50,
            generator=generator,
            required_aovs=[aov_name],
            output_type='pt',
        ).images[0][0]
        out.append(aov)
    return out