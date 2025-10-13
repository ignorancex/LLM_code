"""
    Helper functions mainly for multilingual text image customization.
    Acknowledgement: Codes here are heavily borrowed from TextFLUX: https://github.com/yyyyyxie/textflux.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_prompt(words):
    words_str = ', '.join(f"'{word}'" for word in words)
    prompt_template = (
        "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
        "[IMAGE1] is a template image rendering the text, with the words {words}; "
        "[IMAGE2] shows the text content {words} naturally and correspondingly integrated into the image."
    )
    return prompt_template.format(words=words_str)


prompt_template2 = (
    "The pair of images highlights some white words on a black background, as well as their style on a real-world scene image. "
    "[IMAGE1] is a template image rendering the text, with the words; "
    "[IMAGE2] shows the text content naturally and correspondingly integrated into the image."
)


def run_multilingual_inference(model, image_input, mask_input, reference_input, texts,
                               num_steps=30, guidance_scale=30, seed=42, num_images=1):
    # Resize.
    width, height = image_input.size
    new_width = (width // 32) * 32
    new_height = (height // 32) * 32
    image_input = image_input.convert("RGB").resize((new_width, new_height))
    mask_input = mask_input.convert("RGB").resize((new_width, new_height))

    texts = [i.strip() for i in texts.split('\n')]
    rendered_text = render_glyph_multi(image_input, mask_input, texts)
    combined_image = Image.fromarray(np.hstack((np.array(rendered_text), np.array(image_input))))
    combined_mask = Image.fromarray(
        np.hstack((np.array(Image.new("RGB", image_input.size, (0, 0, 0))), np.array(mask_input))))

    prompt = generate_prompt(texts)
    print("Final prompt:", prompt)

    all_generated_images = []
    for i in range(num_images):
        res = model.generate(
            image=combined_image,
            mask_image=combined_mask,
            ref_image=reference_input,
            prompt=prompt_template2,
            prompt_2=prompt,
            scale=1.0,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            width=combined_image.width,
            height=combined_image.height,
            seed=seed + i,
        )[0]
        all_generated_images.append(res)
    return all_generated_images


def insert_spaces(text, num_spaces):
    """
    Insert a specified number of spaces between each character to adjust spacing during text rendering.
    """
    if len(text) <= 1:
        return text
    return (' ' * num_spaces).join(list(text))


def draw_glyph2(
        font,
        text,
        polygon,
        vertAng=10,
        scale=1,
        width=512,
        height=512,
        add_space=True,
        scale_factor=2,
        rotate_resample=Image.BICUBIC,
        downsample_resample=Image.Resampling.LANCZOS
):
    """
    Render tilted/curved text within a specified region (defined by polygon):
        - First upscale (supersample), then rotate, then downsample to ensure high quality;
        - Dynamically adjust font size and whether to insert spaces between characters based on the region's shape.
    Return the final downsampled RGBA numpy array to the target dimensions (height, width).
    """
    big_w = width * scale_factor
    big_h = height * scale_factor

    # Upscale polygon coordinates
    big_polygon = polygon * scale_factor * scale
    rect = cv2.minAreaRect(big_polygon.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if (abs(angle) % 90 < vertAng or abs(90 - abs(angle) % 90) % 90 < vertAng):
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    # Create large image and temporary white background image
    big_img = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
    tmp = Image.new("RGB", big_img.size, "white")
    tmp_draw = ImageDraw.Draw(tmp)

    _, _, _tw, _th = tmp_draw.textbbox((0, 0), text, font=font)
    if _th == 0:
        text_w = 0
    else:
        w_f, h_f = float(w), float(h)
        text_w = min(w_f, h_f) * (_tw / _th)

    if text_w <= max(w, h):
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_sp = insert_spaces(text, i)
                _, _, tw2, th2 = tmp_draw.textbbox((0, 0), text_sp, font=font)
                if th2 != 0:
                    if min(w, h) * (tw2 / th2) > max(w, h):
                        break
            text = insert_spaces(text, i - 1)
        font_size = min(w, h) * 0.80
    else:
        shrink = 0.75 if vert else 0.85
        if text_w != 0:
            font_size = min(w, h) / (text_w / max(w, h)) * shrink
        else:
            font_size = min(w, h) * 0.80

    new_font = font.font_variant(size=int(font_size))
    left, top, right, bottom = new_font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    # Create transparent text rendering layer
    layer = Image.new("RGBA", big_img.size, (0, 0, 0, 0))
    draw_layer = ImageDraw.Draw(layer)
    cx, cy = rect[0]
    if not vert:
        draw_layer.text(
            (cx - text_width // 2, cy - text_height // 2 - top),
            text,
            font=new_font,
            fill=(255, 255, 255, 255)
        )
    else:
        _w_ = max(box[:, 0]) - min(box[:, 0])
        x_s = min(box[:, 0]) + _w_ // 2 - text_height // 2
        y_s = min(box[:, 1])
        for c in text:
            draw_layer.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(
        angle,
        expand=True,
        center=(cx, cy),
        resample=rotate_resample
    )

    xo = int((big_img.width - rotated_layer.width) // 2)
    yo = int((big_img.height - rotated_layer.height) // 2)
    big_img.paste(rotated_layer, (xo, yo), rotated_layer)

    final_img = big_img.resize((width, height), downsample_resample)
    final_np = np.array(final_img)
    return final_np


def render_glyph_multi(original, computed_mask, texts):
    """
    For each independent region in computed_mask:
        - Extract region positions using contours and sort them from top to bottom, left to right;
        - Call draw_glyph2 to render corresponding text in each region (supports tilt/curve);
        - Overlay the rendering results of each region onto a transparent black background image, and output the final rendered image.
    """
    mask_np = np.array(computed_mask.convert("L"))
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 50:
            continue
        regions.append((x, y, w, h, cnt))
    regions = sorted(regions, key=lambda r: (r[1], r[0]))

    render_img = Image.new("RGBA", original.size, (0, 0, 0, 0))
    try:
        base_font = ImageFont.truetype("resources/Arial-Unicode-Regular.ttf", 40)
    except:
        base_font = ImageFont.load_default()

    for i, region in enumerate(regions):
        if i >= len(texts):
            break
        text = texts[i].strip()
        if not text:
            continue
        cnt = region[4]
        polygon = cnt.reshape(-1, 2)
        rendered_np = draw_glyph2(
            font=base_font,
            text=text,
            polygon=polygon,
            vertAng=10,
            scale=1,
            width=original.size[0],
            height=original.size[1],
            add_space=True,
            scale_factor=1,
            rotate_resample=Image.BICUBIC,
            downsample_resample=Image.Resampling.LANCZOS
        )
        rendered_img = Image.fromarray(rendered_np, mode="RGBA")
        render_img = Image.alpha_composite(render_img, rendered_img)
    return render_img.convert("RGB")
