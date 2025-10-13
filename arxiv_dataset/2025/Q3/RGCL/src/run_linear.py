
from tqdm import tqdm
import wandb
import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import numpy as np
from utils.metrics import eval_and_save_epoch_end, count_parameters
from data_loader.rac_dataloader import CLIP2Dataloader
from utils.visualise_embed_space import plot_embedding_pca, plot_embedding_tsne
from data_loader.dataset import load_feats_from_CLIP

from model.classifier import classifier, classifier_hateClipper
from transformers import AdamW, get_cosine_schedule_with_warmup

# device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args_sys(args_list=None):

    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    '''
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')'''
    arg_parser.add_argument(
        "--path", type=str, default="./data")
    arg_parser.add_argument(
        "--output_path", type=str, default="./logging"
    )
    arg_parser.add_argument(
        '--model', type=str, default="ViT-L_14@336px",
        help="The clip model to use, should match file name in args.path")
    # eg openai_clip-vit-large-patch14-336_HF for pooler feature
    # , openai_clip-vit-large-patch14-336_HF_All for all features, not availble now
    arg_parser.add_argument(
        '--fusion_mode', type=str, default='concat', help='concat, align or cross')
    arg_parser.add_argument(
        '--epochs', type=int, default=5, help='number of epochs')
    arg_parser.add_argument(
        '--dataset', type=str, default='FB', help='FB or MMHS')
    arg_parser.add_argument(
        '--data_split', type=str, default='dev_seentest_seen',
        help='Evaluate on which dataset split')
    arg_parser.add_argument(
        '--exp_name', type=str, default='exp')
    arg_parser.add_argument("--group_name", type=str, default="concat",
                            help=" Name for the wandb group")
    # This train batch size is only for fine tunning
    # As in other modes, training CLIP is not possible
    arg_parser.add_argument('--batch_size', type=int, default=32)

    arg_parser.add_argument("--lr", type=float, default=0.001)
    arg_parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=False,
        help="Use lr scheduler or not",
    )
    arg_parser.add_argument('--num_layers', type=int, default=3)

    # MLP dimension for general
    arg_parser.add_argument('--proj_dim', type=int, default=1024)

    # For hateclipper
    # the pre-modality fusion feature projection dimension
    arg_parser.add_argument('--map_dim', type=int, default=1024)

    arg_parser.add_argument('--dropout', type=float, nargs=3,
                            default=[0.1, 0.4, 0.2],
                            help="Set drop probabilities for map, fusion, pre_output")

    arg_parser.add_argument('--weight_decay', type=float, default=0.01)
    arg_parser.add_argument("--grad_clip", type=float,
                            default=0.1, help="Gradient clipping")
    arg_parser.add_argument("--seed", type=int, default=0)
    arg_parser.add_argument("--device", type=str, default="cuda")
    arg_parser.add_argument("--log_interval", type=int, default=10)
    arg_parser.add_argument("--visualise_embed", type=bool, default=False)
    arg_parser.add_argument(
        "--final_eval",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Doing the final eval or not",
    )
    arg_parser.add_argument(
        "--save_embed",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Save the embedding or not",
    )
    arg_parser.add_argument("--pos_weight_value", type=float, default=None,
                            help="The weight for the positive samples in the cross entropy loss")
    arg_parser.add_argument("--clip_probe", type=lambda x: (str(x).lower() == "true"),
                            default=False, help="Whether to probe CLIP or not")
    
    args = arg_parser.parse_args()
    return args


def main(args):
    exp_name = args.model+"_" + args.fusion_mode+"_"+args.dataset+"_"+args.exp_name
    config = {
        "model": args.model,
        "fusion mode": args.fusion_mode,
        "dataset": args.dataset,
        "data_split": args.data_split,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "no. classifier layers": args.num_layers,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    run = wandb.init(entity="jingbiao",
                     project="CLIP-MMHS-LinearProbe",
                     name=exp_name,
                     group=args.group_name,
                     config=config
                     )
    args.output_path = os.path.join(
        args.output_path, "linear", args.dataset, args.group_name, exp_name, "")
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        os.makedirs(args.output_path + "/ckpt/")
    print(args)
    artifact = run.use_artifact(
        "byrne-lab/MMHS-Retrieval/Facebook_Hateful_Memes:latest", type="dataset")
    # artifact.download("./data/artifacts/")

    # <----------------- Load the data ----------------->
    if args.dataset == "FB":
        train, dev, test_seen, test_unseen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), "FB", args.model
        )
    else:
        train, dev, test_seen = load_feats_from_CLIP(
            os.path.join(args.path, "CLIP_Embedding"), args.dataset, args.model
        )

    (train_dl, dev_dl, test_seen_dl) = CLIP2Dataloader(
        train,
        dev,
        test_seen,
        batch_size=args.batch_size,
        return_dataset=False,
        normalize=False,
    )
    list(enumerate(train_dl))
    image_feat_dim = list(enumerate(train_dl))[0][1]["image_feats"].shape[1]
    
    text_feat_dim = list(enumerate(train_dl))[0][1]["text_feats"].shape[1]
    print("Image feature dimension: ", image_feat_dim)
    print("Text feature dimension: ", text_feat_dim)
    model = classifier_hateClipper(image_feat_dim, text_feat_dim, args.num_layers, args.proj_dim,
                                   args.map_dim, args.fusion_mode, dropout=args.dropout, args=args)
    model.cuda()
    print(model)
    print("Total trainable parameters: {}".format(count_parameters(model)))
    if args.dataset != "MMHS-FineGrained":
        if args.pos_weight_value != None:
            lossFn_classifier = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([args.pos_weight_value], device=args.device))
        else:
            lossFn_classifier = nn.BCEWithLogitsLoss()
    else:
        lossFn_classifier = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(
    #    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_training_steps = args.epochs * len(train)
    if args.lr_scheduler:
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=100,
                                                       num_training_steps=num_training_steps,
                                                       )

    global_step = -1
    # Best model criterion
    best_metric = 0.0
    best_epoch_path = None
    # progress_bar = tqdm(range(num_training_steps))
    for epoch in tqdm(range(args.epochs)):

        for step, batch in enumerate(train_dl):
            global_step += 1
            optimizer.zero_grad()
            # Need classifier in train mode but CLIP in eval mode to frozen LN
            model.train()
            image_feats = batch["image_feats"].to(args.device)
            text_feats = batch["text_feats"].to(args.device)

            labels = batch["labels"].to(args.device)

            output = model(image_feats, text_feats)
            # If labels is 1d, we need to unsqueeze it to 2d
            if len(labels.shape) == 1:
     
                labels = labels.unsqueeze(1)
                
                
            loss_classifier = lossFn_classifier(
                output, labels.float())

            total_loss = loss_classifier

            total_loss.backward()

            if args.lr_scheduler:
                lr_scheduler.step()
            torch.nn.utils.clip_grad_norm(
                model.parameters(), args.grad_clip)
            optimizer.step()

            # progress_bar.update(1)
            if step % 50 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        step,
                        len(train_dl),
                        100.0 * step / len(train_dl),
                        total_loss.item(),
                    )
                )
                metrics = {"train/loss": total_loss.item(),
                           "train/learning_rate": optimizer.param_groups[0]["lr"]}
                wandb.log({**metrics}, global_step)
        if epoch % args.log_interval == 0:
            (acc, roc, pre, recall, f1, eval_loss), _ = eval_and_save_epoch_end(
                args, artifact, train_dl, dev_dl, test_seen_dl, model, epoch)
            metrics = {"dev/acc": acc,
                       "dev/roc": roc,
                       "dev/precision": pre,
                       "dev/recall": recall,
                       "dev/f1": f1,
                       "dev/loss": eval_loss}
            wandb.log({**metrics}, global_step)
        # Save the model if the val criterion is the best so far
        if roc > best_metric:
            print("Current Epoch roc: ", roc, "Best model so far, saving...")
            best_metric = roc

            # Delete the previous best model
            if best_epoch_path is not None:
                if os.path.exists(best_epoch_path):
                    os.remove(best_epoch_path)

            best_epoch_path = args.output_path + \
                "/ckpt/best_model_{}_{}.pt".format(epoch, roc)

            torch.save(
                model.state_dict(),
                best_epoch_path
            )

        # If last epoch or early stop, save the model
        if epoch == args.epochs - 1:
            print("Last Epoch, saving...")
            torch.save(
                model.state_dict(),
                args.output_path +
                "/ckpt/last_model_{}_{}.pt".format(epoch, roc)
            )
            eval_and_save_epoch_end(
                args, artifact, train_dl, dev_dl, test_seen_dl, model, epoch, last_epoch=True)
    if args.visualise_embed:
        plot_embedding_pca(data_path=args.output_path,
                           epoch=args.epochs, name=exp_name, log2wandb=True)
        plot_embedding_tsne(data_path=args.output_path,
                            epoch=args.epochs, name=exp_name, log2wandb=True)


if __name__ == '__main__':

    args = parse_args_sys()
    # set the seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
    wandb.finish()
