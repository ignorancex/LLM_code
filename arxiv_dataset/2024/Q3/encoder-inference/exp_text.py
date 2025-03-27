import yaml, argparse, os, torch, pickle
import numpy as np

import utils, enatk


def get_args():
    parser = argparse.ArgumentParser()

    utils.add_shared_args(parser)
    utils.add_text_exp_args(parser)
    utils.add_cls_task_args(parser)

    return parser.parse_args()


def exp_embed(args, cfg: dict, logger):
    encoder = utils.model.get_text_encoder(args.encoder)

    for train in [True, False]:
        ds = utils.data.get_text_dataset(args.dataset, train=train)
        loader = utils.data.Loader(ds, batch_size=args.embed_bs, train=False)

        logger.info("embedding {} of {} samples".format("trainset" if train else "testset", len(ds)))
        embeds, labels = [], []
        for x, y in loader:
            embeds.append(encoder(x).cpu().numpy())
            labels.append(y.cpu().numpy())

        embeds = np.concatenate(embeds)
        labels = np.concatenate(labels)

        save_path = os.path.join(args.save_dir, args.save_name + "_" + ("trainset.npz" if train else "testset.npz"))
        np.savez(save_path, embeds=embeds, labels=labels)
        logger.info("saved embeded set to {}".format(save_path))


def exp_train(args, cfg: dict, logger):
    trainset = utils.data.get_pre_embed_dataset(args.trainset_path)
    trainloader = utils.data.Loader(trainset, batch_size=cfg["batch_size"], train=True)

    testset = utils.data.get_pre_embed_dataset(args.testset_path)
    testloader = utils.data.Loader(testset, batch_size=cfg["batch_size"], train=False)

    device = "cuda:0"
    classifier_pms = cfg.get("classifier_pms", {})
    classifier_pms.update({"in_dims": trainset.embed_dims, "out_dims": args.cls_num})
    classifier = utils.model.get_classifier(cfg["classifier"], **classifier_pms)
    classifier = classifier.to(device)

    optim = utils.get_optim(cfg["optim"], classifier.parameters(), **cfg["optim_pms"])
    scheduler = utils.get_scheduler(cfg["scheduler"], optim, **cfg["scheduler_pms"])

    criterion = torch.nn.CrossEntropyLoss()

    for step in range(cfg["steps"]):
        x, y = next(trainloader)
        x, y = x.to(device), y.to(device)
        classifier.train()

        _y = classifier(x)
        loss = criterion(_y, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        train_acc = (_y.argmax(dim=1) == y).sum().item() / len(y)
        train_loss = loss.item()

        if (step+1) % cfg["eval_freq"] == 0:
            res = utils.evaluate(classifier, criterion, testloader, device)
            test_acc, test_loss = res["acc"], res["loss"]

            logger.info("Step [{}/{}]".format(step+1, cfg["steps"]))
            logger.info("train_acc = {:.2%}, train_loss = {:.4e}".format(train_acc, train_loss))
            logger.info("test_acc  = {:.2%}, test_loss  = {:.4e}".format(test_acc, test_loss))
            logger.info("")

    torch.save(classifier, os.path.join(args.save_dir, "{}_classifier.pt".format(args.save_name)))


def exp_atk(args, cfg: dict, logger):
    encoder = utils.model.get_text_encoder(args.encoder)
    device = encoder.device

    atker = enatk.TextAtkZerothV2(device=device, logger=logger, **cfg["atker_pms"])

    if args.atk_target_text is not None:
        target_x = args.atk_target_text
        target_embed = encoder(target_x)

    else:
        testset = utils.data.get_text_dataset(args.dataset, train=False)
        target_idx = torch.randint(0, len(testset), (1,)).item()
        target_x, target_y = testset[target_idx]
        target_embed = encoder(target_x)

    logger.info("target text : '{}'".format(target_x))

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
    # logger.info("target label: '{}'".format(target_y))
    logger.info("")

    logger.info("==== Below are the generated {} texts ====".format(len(atk_res["texts"])))
    for txt in atk_res["texts"]:
        logger.info("'{}'".format(txt))
    logger.info("")


@torch.inference_mode()
def exp_eval(args, cfg: dict, logger):
    device = "cpu" if args.cpu_eval else "cuda:0"
    encoder = utils.model.get_text_encoder(args.encoder, device=device)
    classifier = torch.load(args.classifier_path, map_location=device)

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

    agrees = []
    losses = []
    for path in cfg["atkdata"]:
        with open(path, "rb") as f:
            data = pickle.load(f)

        target_x, atkdata = data["target_x"], data["gen_data"]
        target_x = [target_x,] * len(atkdata)

        target_x = [flip_func(x) for x in target_x]
        atkdata = [flip_func(x) for x in atkdata]

        target_embed = encoder(target_x).view(len(atkdata), -1)
        atkdata_embeds = encoder(atkdata).view(len(atkdata), -1)

        n, dim = atkdata_embeds.shape
        embed_losses = ((atkdata_embeds - target_embed) ** 2).mean(dim=-1)

        target_pred = classifier(target_embed).argmax(dim=-1)
        temp = classifier(atkdata_embeds)
        agree_rate = (classifier(atkdata_embeds).argmax(dim=-1) == target_pred).sum().item() / len(atkdata_embeds)

        agrees.append(agree_rate)
        losses.append(embed_losses.mean().item())

    agrees = np.array(agrees)
    losses = np.array(losses)

    logger.info("[text-classification]")
    logger.info("Evaluation Results:")
    logger.info("n: {}, embed_dim: {}, embed_num: {}".format(n, dim, len(cfg["atkdata"])))
    # logger.info("target_cls: prediction: {}".format(target_pred))
    # logger.info("embed_losses: mean = {:.3e}, min = {:.3e}, max = {:.3e}"
    #             .format(embed_losses.mean().item(), embed_losses.min().item(), embed_losses.max().item()))
    # logger.info("atkdata agree_rate: {:.3%}".format(agree_rate))
    logger.info("losses: avg = {:.3e}, std = {:.3e}".format(losses.mean(), losses.std()))
    logger.info("agrees: avg = {:.2%}, std = {:.2%}".format(agrees.mean(), agrees.std()))
    logger.info("PEI Score: {:.2%}".format(agrees.mean()))


def main(args, logger):
    cfg = None
    if args.exp_config is not None:
        with open(args.exp_config, "r") as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
        with open(os.path.join(args.save_dir, "{}_cfg.yaml".format(args.save_name)), "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    if args.exp_type == "embed":
        exp_embed(args, cfg, logger)
    elif args.exp_type == "train":
        exp_train(args, cfg, logger)
    elif args.exp_type == "eval":
        exp_eval(args, cfg, logger)
    elif args.exp_type == "atk":
        exp_atk(args, cfg, logger)


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
