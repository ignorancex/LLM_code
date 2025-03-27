import yaml, argparse, os, torch, pickle
import numpy as np

import utils, enatk


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)
    utils.add_text2img_exp_args(parser)

    return parser.parse_args()


def exp_atk(args, cfg: dict, logger):
    encoder = utils.model.get_text_encoder(args.encoder, pooling=False)
    device = encoder.device

    atker = enatk.TextAtkZerothV2(device=device, logger=logger, **cfg["atker_pms"])

    target_x = args.atk_target_text
    target_embed = encoder(target_x)

    atk_res = atker.attack(encoder=encoder, target=target_embed,
                           gen_num=cfg["gen_num"], query_bs=cfg["query_bs"])

    save_res = {
        "target_x": target_x,
        "gen_data": atk_res["texts"],
    }

    with open(os.path.join(args.save_dir, "{}_gen-data.pkl".format(args.save_name)), "wb") as f:
        pickle.dump(save_res, f)

    logger.info("")
    logger.info("target text : '{}'".format(target_x))
    logger.info("")

    logger.info("==== Below are the generated {} texts ====".format(len(atk_res["texts"])))
    for txt in atk_res["texts"]:
        logger.info("'{}'".format(txt))
    logger.info("")


@torch.inference_mode()
def exp_eval(args, cfg: dict, logger):
    device = "cpu" if args.cpu_eval else "cuda:0"
    model= utils.model.get_text2img_model(args.text2img_model, device=device, num_img=cfg["eval_num_img"])
    # encoder = utils.model.get_img_encoder(args.eval_encoder, device=device)
    encoder = utils.model.get_img_encoder("clip-vit-b16", device=device)
    cosine_loss_fn = torch.nn.CosineSimilarity().to(device)

    def run_text2img(prompt):
        size = len(prompt)
        res = []
        for i in range(0, size, cfg["eval_bs"]):
            k = min(i+cfg["eval_bs"], size)
            res.extend(model(prompt[i:k]))
            torch.cuda.empty_cache()
        return res

    def flip_func(x):
        if args.def_flip_rate is None:
            return x

        charset = "abcdefghijklmnopqrstuvwxyz"
        flip_num = int(args.def_flip_rate * len(x))

        if flip_num == 0:
            return x

        flip_idx = np.random.permutation(np.arange(len(x)))[:flip_num]

        flip = [False for i in range(len(x))]
        for idx in flip_idx:
            flip[idx] = True

        res = ""
        for i in range(len(x)):
            res += charset[np.random.randint(len(charset))] if flip[i] else x[i]

        return res

    losses = []
    for path in cfg["atkdata"]:
        with open(path, "rb") as f:
            data = pickle.load(f)

        target_x, atkdata = data["target_x"], data["gen_data"]
        target_x = [target_x,] * len(atkdata)

        target_x = [flip_func(x) for x in target_x]
        atkdata = [flip_func(x) for x in atkdata]

        target_gendata = run_text2img(target_x)
        atkdata_gendata = run_text2img(atkdata)

        target_embeds = encoder(target_gendata).view(len(target_gendata), -1)
        atkdata_embeds = encoder(atkdata_gendata).view(len(atkdata_gendata), -1)

        lo = cosine_loss_fn(target_embeds, atkdata_embeds)
        losses.append(lo.mean().item())

        logger.info("target_x = '{}'".format(target_x[0]))
        logger.info("[Current] cosine-similarity: avg = {:.3e}, std = {:.3e}".format(lo.mean().item(), lo.std().item()))

    losses = np.array(losses)

    logger.info("[text-to-image-generation]")
    logger.info("Evaluation Results:")
    # logger.info("target_cls prediction: {}".format(target_pred))
    # logger.info("embed_losses: mean = {:.3e}, min = {:.3e}, max = {:.3e}"
    #             .format(embed_losses.mean().item(), embed_losses.min().item(), embed_losses.max().item()))
    # logger.info("atkdata agree_rate: {:.3%}".format(agree_rate))
    logger.info("[Overall] cosine-similarity: avg = {:.3e}, std = {:.3e}".format(losses.mean(), losses.std()))
    logger.info("PEI Score: {:.3e}".format(losses.mean()))


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
        exp_eval(args, cfg, logger)


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
