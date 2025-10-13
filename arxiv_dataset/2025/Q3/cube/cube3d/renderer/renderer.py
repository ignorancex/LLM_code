import argparse
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image


def render_asset(
    asset_path,
    output_dir,
    nviews=24,
    img_resolution=512,
):
    """
    Render given asset into output_dir and return the saved image paths.
    Assumes that blender is installed and is in your path.

        nviews : number of views to render
        img_resolution : resolution of each rendered view in pixels
    """

    curr_file_path = __file__
    curr_dir = os.path.dirname(curr_file_path)

    command = [
        "blender",
        "--background",
        "-noaudio",
        "--python",
        f"{curr_dir}/blender_script.py",
        "--",
        "--object_path",
        asset_path,
        "--num_renders",
        str(nviews),
        "--output_dir",
        output_dir,
        "--render_resolution",
        str(img_resolution),
        "--transparent_background",
        "--engine",
        "CYCLES",
    ]

    subprocess.run(command, check=True)

    # return the saved images paths
    images = []

    for i in range(nviews):
        fp = os.path.abspath(os.path.join(output_dir, f"{i:03d}_textured.png"))
        images.append(fp)

    return images


def save_gif(image_paths, outfile):
    images = [Image.open(img) for img in image_paths]
    if len(images) > 1:
        background = Image.new("RGBA", images[0].size, (255, 255, 255))
        images = [
            Image.alpha_composite(background, png).convert("RGB") for png in images
        ]
        images[0].save(
            outfile, save_all=True, append_images=images[1:], duration=100, loop=0
        )


def render_turntable(obj_path, output_dir, output_name="turntable"):
    """
    Render a turntable gif of the mesh. Assumes that blender is installed and is in your path.
        obj_path : path to the obj file
        output_dir : directory to save the gif. Final image will be saved as `turntable.gif`
    """
    image_paths = render_asset(obj_path, output_dir)
    gif_turntable_outfile = Path(output_dir) / f"{output_name}.gif"
    save_gif(image_paths, gif_turntable_outfile)
    return gif_turntable_outfile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args(sys.argv[1:])
    render_turntable(args.input, args.output_dir)
