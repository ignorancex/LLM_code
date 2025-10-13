from bg_randomized_loce.synthetic_bg_generation import UniformNoiseBackgroundGenerator, WuerstchenBackgroundGenerator, \
    BG_DIFFUSION
from bg_randomized_loce.utils.logging import init_logger

if __name__ == "__main__":
    init_logger()

    # Uniform noise
    generator = UniformNoiseBackgroundGenerator()
    generator.generate_noise_images(n=100)


    # Wuerstchen

    prompts = [
        f"A vast sky filled with dramatic clouds in varying shapes and layers, emphasizing realistic lighting and textures, devoid of objects or creatures."
        if background == "cloudscape" else
        f"A vast space skybox with stars, galaxies, and nebulae, creating a serene cosmic atmosphere, devoid of objects or creatures."
        if background == "space" else
        f"A tranquil underwater ocean scene with soft light beams filtering through the water, focusing on the textures of sand, water ripples, and vibrant coral formations."
        if background == "ocean" else
        f"A volcanic scene with a dark sky, an erupting volcano, glowing red lava flows, and ash clouds in the atmosphere."
        if background == "volcanic" else
        f"A {background} background scene with a realistic and immersive environment, devoid of any objects, creatures, or vehicles."
        if "abstract" not in background else
        f"An {background.replace('_', ' ')} background with smooth, surreal elements and vibrant colors."
        for background in BG_DIFFUSION
    ]

    generator = WuerstchenBackgroundGenerator()
    generator.generate_backgrounds(prompts, BG_DIFFUSION, n=100)


