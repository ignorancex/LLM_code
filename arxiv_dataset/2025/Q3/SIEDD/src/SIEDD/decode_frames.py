import re
from pathlib import Path

import click
from pydantic_yaml import parse_yaml_file_as

from SIEDD.utils.helpers import save_tensor_img
from SIEDD.configs import RunConfig
from SIEDD.data_processing import DataProcess
from SIEDD.decode import decode_image
from SIEDD.strainer import Strainer


def parse_frames(frames_str: str):
    frames: list[int] = []
    regex = re.compile("[^0-9,-]")
    frames_str = regex.sub("", frames_str)
    frames_str += ","
    split_frames = [x for x in frames_str.split(",") if x != ""]
    for frame_i in split_frames:
        if "-" in frame_i:
            start, end = frame_i.split("-")
            frames.extend(range(int(start), int(end) + 1))
        else:
            frames.append(int(frame_i))
    return sorted(set(frames))


@click.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("frames")
@click.option(
    "--output", required=True, help="Output path pattern, e.g., output_dir/f%05d.png"
)
@click.option("--resolution", help="Output resolution in WxH format (e.g. 1920x1080)")
def main(path: Path, frames: str, output: str, resolution: str | None):
    """
    PATH is the output path from SIEDD training, which should contain
    the model.bin files, dataset_state.json, shared_encoder.bin, and run_cfg.yaml.

    FRAMES is the frame indices to decode. The first frame is 0.
    You can define multiple frames delimited by commas and
    you can define ranges using hyphens. Ex: 0,1,2-4,200
    will decode frames 0, 1, 2, 3, 4, and 200
    """
    frame_numbers = parse_frames(frames)
    frame_outputs = []
    click.echo(f"Decoding frames: {frame_numbers} from {path}")
    for frame in frame_numbers:
        try:
            out_path = output % frame
        except Exception:
            if len(frame_numbers) > 1:
                raise RuntimeError(f"All frames writing to the same file: {output}")
            out_path = output
        frame_outputs.append(Path(out_path).absolute())

    cfg = parse_yaml_file_as(RunConfig, path / "run_cfg.yaml")
    pipeline = DataProcess(cfg, path)
    strainer = Strainer(
        cfg=cfg,
        data_pipeline=pipeline,
        save_path=path,
        resume=False,
        setup_wandb=False,
    )
    strainer.data_pipeline.data_set.load_state(strainer.save_path)
    num_frames = pipeline.num_frames
    gs = cfg.trainer_cfg.group_size
    frame_groups_dict = {}
    frame_group_ranges = [
        range(start, start + gs) for start in range(0, num_frames, gs)
    ]
    for group in frame_group_ranges:
        frames_in_group = []
        group_out_paths = []
        for frame, pth in zip(frame_numbers, frame_outputs):
            if frame in group:
                frames_in_group.append(frame)
                group_out_paths.append(pth)
        if len(frames_in_group) > 0:
            frame_groups_dict[group.start] = frames_in_group, group_out_paths

    if resolution is None:
        C, H, W = pipeline.data_set.data_shape
        res = W, H
    else:
        W, H = resolution.split("x")
        res = int(W), int(H)
    for frame_group_start, (frame_group, group_out_paths) in frame_groups_dict.items():
        idx_within_group = [idx % gs for idx in frame_group]
        pred, fps = decode_image(strainer, frame_group_start, res)
        pred = pred[idx_within_group]
        for out_path, pred_img in zip(group_out_paths, pred):
            save_tensor_img(pred_img, out_path)


if __name__ == "__main__":
    main()
