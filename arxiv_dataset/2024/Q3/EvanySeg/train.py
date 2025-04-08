from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import MSELoss
from dataset import MyDataset
import random
from tqdm import tqdm
from tools import *
import warnings
import pandas as pd
from utils.logger import Logger
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train(args,model,train_dataset,train_dataloader,optimizer,criterion1,criterion2,scheduler,logger):
    Path(args.result_path).mkdir(exist_ok=True, parents=True)
    for epoch in range(1, args.epochs+1):
        epoch_losses = 0
        epoch_dice_loss = 0
        epoch_hd_loss = 0
        for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='train'):
            crop_images = batch['crop_image_trans'].cuda()
            output = model(crop_images)
            if args.flag == 1:
                true_dices = batch['dice'].cuda()
                true_dices = true_dices.float()
                output_dice = output["logits_dice"].squeeze(1).float()
                loss_dice = criterion1(output_dice, true_dices)
                losses = loss_dice
                epoch_losses += losses.item()
                epoch_dice_loss += loss_dice.item()
            else:
                true_dices = batch['dice'].cuda()
                true_dices = true_dices.float()
                true_hd = batch['hd'].cuda()
                true_hd = true_hd.float()

                output_dice = output["logits_dice"].squeeze(1).float()
                output_hd = output["logits_hd"].squeeze(1).float()

                loss_dice = criterion1(output_dice, true_dices)
                loss_hd = criterion2(output_hd, true_hd)
                losses = loss_dice + 0.0001 * loss_hd
                epoch_losses += losses.item()
                epoch_dice_loss += loss_dice.item()
                epoch_hd_loss += loss_hd.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        if args.flag == 1:
            logger.info(
                f"epoch:{epoch}  --epoch_sum_loss:{epoch_losses}  --epoch_dice_loss:{epoch_dice_loss}  --lr:{optimizer.param_groups[0]['lr']}")
        else:
            logger.info(
                f"epoch:{epoch}  --epoch_sum_loss:{epoch_losses}  --epoch_dice_loss:{epoch_dice_loss}  --epoch_hd_loss:{epoch_hd_loss} --lr:{optimizer.param_groups[0]['lr']}")
        scheduler.step()

        if epoch % args.save_interval == 0:
            torch.save(model.state_dict(), f"{args.result_path}/checkpoints/epoch_{epoch}_{args.classification_model_name}_{args.flag}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=32, help='number of workers')
    parser.add_argument('--img_size', type=int, default=224, help='image size')
    parser.add_argument('--flag', type=int, default=1, help="1 for dice; 2 for dice and hd")
    parser.add_argument("--root_path", type=str, default='./datasets/preprocess/train_sam_Polyp', help='path to dataset')
    parser.add_argument("--weight_decay", type=float, default=1e-5, help='weight decay')
    parser.add_argument("--save_interval", type=int, default=1, help='save interval')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--result_path", type=str, default="./result")
    parser.add_argument("--classification_model_name", type=str, default="vit_base_224", help='[ resnet50 | resnet101 | vit_base_224 | vit_large_224 ]')
    args=parser.parse_args()

    args.result_path = create_paths(args.result_path)
    logger = Logger(f"{args.result_path}/log/train_log.txt").get_logger()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    train_dataset = MyDataset(root_dir=args.root_path, transform=transform, flag=args.flag)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    criterion1 = MSELoss()
    criterion2 = MSELoss()

    model = load_classification_model(args)
    model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    train(args, model, train_dataset, train_dataloader, optimizer, criterion1, criterion2, scheduler, logger)


