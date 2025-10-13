from SIEDD.exp import stable_config_4, PROJECT_ROOT
from SIEDD.main import DataProcess
from SIEDD.configs import DenoisingType
from SIEDD.utils.metric import MSEPSNR
from itertools import batched
import cv2
import einops
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd


def main():
    cfg = stable_config_4()
    cfg.encoder_cfg.image_transform.denoising = True

    dp = DataProcess(cfg, PROJECT_ROOT / "data" / "UVG", True)
    it = tqdm(
        list(
            map(
                list,
                batched(range(dp.num_frames), 20),
            )
        )
    )
    msepsnr = MSEPSNR()

    denoising_types: list[DenoisingType] = [
        "all_white",
        "all_black",
        "salt_pepper",
        "random",
        "gaussian",
    ]
    baselines = {x: [] for x in denoising_types}
    gausses = {x: [] for x in denoising_types}
    medians = {x: [] for x in denoising_types}

    for frames in it:
        original_frames = [dp.data_set.load_sample(i) for i in frames]

        for denoising_type in denoising_types:
            cfg.encoder_cfg.image_transform.denoising_type = denoising_type
            dp.data_set.transform.args.denoising_type = denoising_type
            results = [
                dp.data_set.preprocess_image(x, i)
                for i, x in zip(frames, original_frames)
            ]
            processed = [x[0] for x in results]
            orig = [x[1] for x in results]
            processed = torch.stack(processed).cuda()
            orig = torch.stack(orig).cuda()
            orig = einops.rearrange(orig, "n (h w) c -> n h w c", h=1080)
            processed = einops.rearrange(processed, "n (h w) c -> n h w c", h=1080)
            outs = []
            for i, idx in enumerate(frames):
                outs.append(dp.data_set.transform.inverse(processed[i], idx))
            processed = torch.stack(outs)
            # from PIL import Image
            # Image.fromarray((processed[0].cpu().numpy()*255).astype("uint8")).save("processed.png")
            # exit()
            baseline_psnr = msepsnr(orig, processed)[0]

            gauss = torch.from_numpy(
                np.array(
                    [cv2.GaussianBlur(x.cpu().numpy(), (7, 7), 1) for x in processed]
                )
            ).cuda()
            gauss_psnr = msepsnr(orig, gauss)[0]

            median = torch.from_numpy(
                np.array([cv2.medianBlur(x.cpu().numpy(), 5) for x in processed])
            ).cuda()
            median_psnr = msepsnr(orig, median)[0]

            baselines[denoising_type].append(baseline_psnr.mean().item())
            gausses[denoising_type].append(gauss_psnr.mean().item())
            medians[denoising_type].append(median_psnr.mean().item())
            dp.data_set.uncache_frames()
    data = [
        {x: np.mean(b) for x, b in baselines.items()},
        {x: np.mean(g) for x, g in gausses.items()},
        {x: np.mean(m) for x, m in medians.items()},
    ]

    df = pd.DataFrame(data, index=["baseline", "gaussian", "median"])  # type: ignore
    print(df)
    df.to_csv("denoising_baseline.csv")


if __name__ == "__main__":
    main()
