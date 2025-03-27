import os
import numpy as np
from PIL import Image


def visualize(out_dir, image=None, **kwargs):
    """Visualize maps and masks to out_dir.

    Args:
        out_dir
        image (image_path or PIL Image)
        map/map_masked (torch.bool): (15, *, *)
        map (torch.bool): (14, *, *)
        mask (torch.bool): (1, *, *)
    """

    colormap = {
        "drivable_area": (166, 206, 227),  # '#a6cee3'
        "ped_crossing": (251, 154, 153),  # '#fb9a99'
        "walkway": (227, 26, 28),  # '#e31a1c'
        "carpark_area": (255, 127, 0),  # '#ff7f00'
        "car": (255, 158, 0),  # Orange
        "truck": (255, 99, 71),  # Tomato
        "trailer": (255, 140, 0),  # Darkorange
        "bus": (255, 127, 80),  # Coral
        "construction_vehicle": (233, 150, 70),  # Darksalmon
        "bicycle": (220, 20, 60),  # Crimson
        "motorcycle": (255, 61, 99),  # Red
        "pedestrian": (0, 0, 230),  # Blue
        "traffic_cone": (47, 79, 79),  # Darkslategrey
        "barrier": (112, 128, 144),  # Slategrey
        "mask": (63, 63, 63),  # Grey
    }  # RGB

    colormap = {
        "drivable_area": (0, 48, 117),
        "ped_crossing": (0, 72, 172),
        "walkway": (0, 61, 147),
        "carpark_area": (0, 39, 91),
        "car": (179, 0, 27),
        "truck": (108, 13, 27),
        "trailer": (69, 8, 17),
        "bus": (141, 2, 136),
        "construction_vehicle": (179, 85, 0),
        "bicycle": (0, 133, 41),
        "motorcycle": (38, 125, 128),
        "pedestrian": (204, 201, 41),
        "traffic_cone": (204, 133, 41),
        "barrier": (128, 128, 128),
        "mask": (0, 0, 0),
    }

    os.makedirs(out_dir, exist_ok=True)
    if image is not None:
        if isinstance(image, str):
            image = Image.open(image)  # .resize((704, 396)).crop((0, 140, 704, 396))
            image.save(os.path.join(out_dir, "image.png"))
        elif isinstance(image, Image.Image):
            # image = image.resize((704, 396)).crop((0, 140, 704, 396))
            image.save(os.path.join(out_dir, "image.png"))
    for key, value in kwargs.items():
        value = np.flip(value.cpu().numpy(), axis=1)
        canvas = np.zeros((*value.shape[-2:], 3), dtype=np.uint8)
        if value.shape[0] == 1:
            canvas[value[0], :] = colormap["mask"]
            Image.fromarray(canvas).save(os.path.join(out_dir, key + ".png"))
        else:
            canvas[:, :] = (0, 29, 68)
            for i, (_, color) in enumerate(colormap.items()):
                if np.any(value[i]):
                    canvas[value[i], :] = color
                if i == 13:
                    Image.fromarray(canvas).save(os.path.join(out_dir, key + ".png"))
                    if value.shape[0] == 14:
                        break
                elif i == 14:
                    Image.fromarray(canvas).save(os.path.join(out_dir, key + "_masked.png"))
