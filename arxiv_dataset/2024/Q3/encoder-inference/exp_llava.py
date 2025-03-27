import yaml, argparse, os, torch, numpy as np, re
from PIL import Image
from io import BytesIO

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers import CLIPImageProcessor, CLIPVisionModel

import utils, enatk


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)
    utils.add_img2text_exp_args(parser)

    return parser.parse_args()


def exp_atk(args, cfg: dict, logger):
    encoder = utils.model.get_img_encoder(args.encoder)
    device = encoder.device

    atker = enatk.ImgAtkZeroth(device=device, logger=logger, **cfg["atker_pms"])

    if args.atk_target_img is not None:
        target_x = np.array(Image.open(args.atk_target_img))
        target_embed = encoder(target_x)

    else:
        testset = utils.data.get_img_dataset(args.dataset, train=False, trans=None)
        target_idx = torch.randint(0, len(testset), (1,)).item()
        target_x, target_y = testset[target_idx]
        target_embed = encoder(target_x)

    gen_data = atker.attack(encoder=encoder, target=target_embed,
                            gen_num=cfg["gen_num"], query_bs=cfg["query_bs"])

    np.savez(os.path.join(args.save_dir, "{}_gen-data.npz".format(args.save_name)),
             target_x=target_x, gen_data=gen_data)


@torch.inference_mode()
def exp_eval_llava(args, cfg: dict, logger):
    device = "cpu" if args.cpu_eval else "cuda:0"
    dtype = torch.float16
    model_id = "llava-hf/llava-1.5-13b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=BitsAndBytesConfig())
    processor = AutoProcessor.from_pretrained(model_id)

    prompt = "USER: <image>\n<image>\nScore the similarity (in the range [0,1], higher score means more similar) of the given two images\nASSISTANT:"

    fin_sim, fin_cnt = 0, 0

    for idx, path in enumerate(cfg["atkdata"]):
        data = np.load(path)
        target_x, atkdata = data["target_x"], data["gen_data"]

        sim, cnt = 0, 0
        for gen_x in data["gen_data"]:
            inputs = processor(prompt, (target_x, gen_x), return_tensors='pt').to(device, torch.float16)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            out1 = re.findall("\d+\.\d+", processor.decode(output[0], skip_special_tokens=True))
            if len(out1) != 1: continue

            inputs = processor(prompt, (gen_x, target_x), return_tensors='pt').to(device, torch.float16)
            output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            out2 = re.findall("\d+\.\d+", processor.decode(output[0], skip_special_tokens=True))
            if len(out2) != 1: continue

            sim += float(out1[0]) + float(out2[0])
            cnt += 1

        if cnt > 0:
            sim /= (cnt << 1)
            fin_sim += sim
            fin_cnt += 1

        logger.info("Case {}: sim = {:.3e}, cnt = {}".format(idx+1, sim, cnt))

    if fin_cnt > 0:
        fin_sim /= fin_cnt

    logger.info("fin_sim = {:.3e}, fin_cnt = {}".format(fin_sim, fin_cnt))
    logger.info("PEI Score: {:.3e}".format(fin_sim))


def exp_furatk(args, cfg: dict, logger):
    device = "cpu" if args.cpu_eval else "cuda:0"
    # dtype = torch.float16
    dtype = torch.float32
    model_id = "openai/clip-vit-large-patch14-336"

    encoder = CLIPVisionModel.from_pretrained(model_id).to(device)
    for pp in encoder.parameters():
        pp.requires_grad = False

    trans = CLIPImageProcessor.from_pretrained(model_id)

    H = trans.crop_size["height"]
    W = trans.crop_size["width"]
    C = 3
    img_mean = torch.tensor(trans.image_mean, dtype=dtype, device=device).view(1, C, 1, 1)
    img_std = torch.tensor(trans.image_std, dtype=dtype, device=device).view(1, C, 1, 1)

    clip_min = (0.0 - img_mean) / img_std
    clip_max = (1.0 - img_mean) / img_std

    target_x = np.array(Image.open(args.atk_target_img))
    with torch.no_grad():
        target_embed = encoder(
            trans(target_x, return_tensors="pt").to(device)["pixel_values"]
        ).pooler_output

    opt_x = torch.rand([args.further_atk_img_num, C, H, W], dtype=dtype, device=device)
    steps = cfg["steps"]
    save_freq = cfg["save_freq"]
    for sp in range(steps):
        lr = cfg["lr"]

        inp_x = (opt_x - img_mean) / img_std
        inp_x.requires_grad = True
        losses = ((encoder(inp_x).pooler_output - target_embed) ** 2).view(len(inp_x), -1).mean(1) #.sum()
        loss = losses.sum()
        mi_loss = losses.min().item()
        mx_loss = losses.max().item()
        grad = torch.autograd.grad(loss, [inp_x,])[0]
        grad = grad / img_std

        opt_x += (-lr) * torch.sign(grad)
        opt_x.clip_(0,1)

        # logger.info("Step [{}/{}]: loss = {:.3f}".format(sp+1, steps, loss.item()))
        logger.info("Step [{}/{}]: loss = {:.3f}, mi_loss = {:.3f}, mx_loss = {:.3f}".format(sp+1, steps, loss.item(), mi_loss, mx_loss))

        if (sp+1) % save_freq == 0:
            img_x = (opt_x * 255).round().clip(0,255).permute([0,2,3,1]).to("cpu", torch.uint8).numpy()
            np.save("{}-iter{}.npy".format(args.further_atk_img, sp+1), img_x)

    img_x = (opt_x * 255).round().clip(0,255).permute([0,2,3,1]).to("cpu", torch.uint8).numpy()
    # img_x = img_x.squeeze(0)

    base_dirs = os.path.dirname(args.further_atk_img)
    if not os.path.exists(base_dirs):
        os.makedirs(base_dirs)

    np.save("{}-fin.npy".format(args.further_atk_img), img_x)

    # if args.further_atk_img_num == 1:
    #     Image.fromarray(img_x).save(args.further_atk_img)
    # else:
    #     np.save(args.further_atk_img, img_x)


@torch.inference_mode()
def exp_eval_furatk(args, cfg: dict, logger):
    device = "cpu" if args.cpu_eval else "cuda:0"
    dtype = torch.float16
    model_id = "llava-hf/llava-1.5-13b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=BitsAndBytesConfig())
    processor = AutoProcessor.from_pretrained(model_id)

    prompt = "USER: <image>\nPlease read out the text in this image.\nASSISTANT:"

    if os.path.splitext(args.eval_imgs)[-1] == ".npy":
        imgs = np.load(args.eval_imgs)
    else:
        imgs = [np.asarray(Image.open(args.eval_imgs)),]
    for i, im in enumerate(imgs):
        inputs = processor(prompt, im, return_tensors='pt').to(0, torch.float16)
        output = model.generate(**inputs, max_new_tokens=100)
        print("[idx {}]: {}".format(i, processor.decode(output[0], skip_special_tokens=True)))
        print("")


def main(args, logger):
    cfg = None
    if args.exp_config is not None:
        with open(args.exp_config, "r") as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
        with open(os.path.join(args.save_dir, "{}_cfg.yaml".format(args.save_name)), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    if args.exp_type == "atk":
        exp_atk(args, cfg, logger)
    elif args.exp_type == "eval":
        exp_eval_llava(args, cfg, logger)
    elif args.exp_type == "furatk":
        exp_furatk(args, cfg, logger)
    elif args.exp_type == "eval-furatk":
        exp_eval_furatk(args, cfg, logger)


if __name__ == "__main__":
    args = get_args()
    logger = utils.generic_init(args)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    try:
        main(args, logger)

    except Exception as e:
        logger.exception('Unexpected exception! %s', e)
