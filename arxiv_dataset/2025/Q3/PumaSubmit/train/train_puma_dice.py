import logging
from transformers import SegformerImageProcessor
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from utils import Mine_resize
from utils import KorniaAugmentation
from torch import optim
from torch.utils.data import DataLoader, random_split
from LoadPumaData import load_data_tissue,PumaTissueDataset
from utils import compute_puma_dice_micro_dice
from utils import FocalLoss
from torch.cuda.amp import autocast, GradScaler
from class_augs import tissue_aug_gpu
import torch.nn.utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import puma_dice_loss, circular_augmentation,random_sub_image_sampling
import numpy as np



def upsample_difficult_cases(dice_epoch = None, inddss = None):
    diff_inds = inddss[np.where(dice_epoch > np.median(dice_epoch))[0]]
    return diff_inds




def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale = 0,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        target_siz = (128,128),
        n_class = 6,
        image_data1 = None,
        mask_data1 = None,
        val_images = None,# None for training on whole dataset1
        val_masks = None,
        class_weights = torch.ones(6),
        augmentation = True,
        val_batch = 1,
    early_stopping = 8,
        ful_size = (1024,1024),
        grad_wait = 1, # for small batch size, we manually save and update gradients
        logg = False,# logger file
        logg_selected = False,
        val_augmentation = None,
        train_indexes=None,# with this parameter we found difficult frames to learn during training and upsampled these frames in training
        input_folder='',
        output_folder='',
        ground_truth_folder='',
        phase_mode = ['train', 'val'],# there training modes are possible ['train'], ['train','val] or ['train', 'val', 'test']
        test_images=None,# None for a train-val approach and not train-val-test
        test_masks=None,
        test_folder='',
        test_output_folder='',
        test_ground_truth_folder='',
        folds = None,
        dir_checkpoint = Path('E:/PumaDataset/checkpoints/'),#model checkpoint save directory
        er_di = False, # this was written for an experiment for erosion, dilation effect analysis
        progressive = False, # progressive training approach
        necros_im = None,
        model_name = '',# there are two models available: ['segformer'] or, ['unet']
        val_sleep_time = 0, # starts validation after val_sleep_time epochs
        nuclei = False
):
    # 1. Create dataset
    train_set = PumaTissueDataset(image_data1,
                                      mask_data1,
                                      n_class1=n_class,
                                      size1=target_siz,
                                    device1=device,
                                      transform = augmentation,
                                  target_size=ful_size,
                                  train_indexes = train_indexes,
                                  mode='train',
                                  er_di = er_di,)
    if val_images is not None:
        val_set = PumaTissueDataset(val_images,
                                          val_masks,
                                          n_class1=n_class,
                                          size1=target_siz,
                                        device1=device,
                                          transform = None,
                                    target_size=ful_size,
                                    mode='valid',
                                    er_di = er_di)
        n_val = len(val_set)

    if test_images is not None:
        test_set = PumaTissueDataset(test_images,
                                      test_masks,
                                      n_class1=n_class,
                                      size1=target_siz,
                                      device1=device,
                                      transform=None,
                                      target_size=ful_size,
                                      mode='valid',
                                     er_di = er_di)
    # check if everything is correct
    # test_im = val_set.__getitem__(0)

    n_train = len(train_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0)#os.cpu_count(), pin_memory=True)
    val_loader_args = dict(batch_size=val_batch, num_workers=0)#os.cpu_count(), pin_memory=True)
    aug_pipeline = KorniaAugmentation(
        mode="train", num_classes=n_class, seed=42, size=target_siz
    )

    if test_images is not None:
        dataloaders = {
            'train': DataLoader(train_set, shuffle=True, **loader_args),
            'val': DataLoader(val_set, shuffle=False, drop_last=False, **val_loader_args),
            'test': DataLoader(val_set, shuffle=False, drop_last=False, **val_loader_args)
        }
    elif val_images is not None:
        dataloaders = {
            'train': DataLoader(train_set, shuffle=True, **loader_args),
            'val': DataLoader(val_set, shuffle=False, drop_last=False, **val_loader_args),
        }
    else :
        dataloaders = {
            'train': DataLoader(train_set, shuffle=True, **loader_args),
        }


    # (Initialize logging)
    if logg:
        # Create a logger instance
        logging1 = logging.getLogger(__name__)

        # Set up a specific handler for this logger
        if logg_selected:
            handler = logging.FileHandler('logs/' + str(target_siz[0]) + '/' + 'selected/' + str(class_weights) + '.log',
                                      encoding='utf-8')
        else:
            handler = logging.FileHandler('logs/' + str(target_siz[0]) + '/' + str(class_weights) + '.log',
                                      encoding='utf-8')
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # Add the handler to the logger
        logging1.addHandler(handler)
        logging1.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

        logging1.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Images scaling:  {img_scale}
            Weights: {class_weights}
        ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if progressive:
        optimizer = optim.Adam([
            {'params': model.encoder.parameters(), 'lr': 1e-4},  # Low LR for encoder
            {'params': model.decoder.parameters(), 'lr': 1e-4},  # Moderate LR for decoder
            {'params': model.segmentation_head.parameters(), 'lr': 1e-3},  # High LR for head
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)#,weight_decay=0.0001)
        # optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
        #                         weight_decay=1e-4)  # Adjust weight_decay if needed
#    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    focal_loss_fn = FocalLoss(gamma=2, alpha=class_weights.float())

    if model_name == 'segformer':
        # since segformer learns faster
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5,factor=weight_decay,min_lr=1e-7,cooldown=2)  # goal: maximize Dice score
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50,factor=weight_decay,min_lr=1e-7,cooldown=10)  # goal: maximize Dice score

    optimizer.zero_grad()
    best_val_score = 0
    # 5. Begin training
    counter = 0
    random.seed(42)
    # images, true_masks = dataloaders['train']
    nec_tis = False
    all_epochs = epochs
    processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512", do_resize=False, do_rescale=False)
    dice_cof = 0.5
    m_dice_cof = 0.5
    if progressive: # set gradient update for all parameters False
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False

    if grad_wait<=0:
        grad_wait=1
    for pro_phase in ["segmentation_head", "decoder", "encoder"] if progressive else ["all"]:
        print(f"\nStarting Phase: {pro_phase.upper()}")

        if pro_phase == "segmentation_head":
            epochs = int(all_epochs/20)
            sub_size = 1024

            for param in model.segmentation_head.parameters():
                param.requires_grad = True
        elif pro_phase == "decoder":
            sub_size = 1024

            epochs = int(all_epochs/20)

            for param in model.decoder.parameters():
                param.requires_grad = True
        elif pro_phase == "encoder":
            sub_size = 1024

            epochs = int(all_epochs - 2*int(all_epochs/20))

            for param in model.encoder.parameters():
                param.requires_grad = True
        for epoch in range(1, epochs + 1):
            # gw = grad_wait
            if counter > early_stopping:
                break
            scaler = GradScaler(enabled=amp)
            gradient_clipping = 1.0  # Gradient clipping value

            for phase in phase_mode:
                if phase == 'train':
                    model.train()
                    epoch_loss = 0
                    epoch_dice = torch.zeros(n_class, device=device)
                    with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                        for images, true_masks, inds in dataloaders[phase]:

                            images = images.to(device=device, dtype=torch.float32)
                            true_masks = true_masks.to(device=device, dtype=torch.long)



                            # this is a specific calss stick augmentation specially for blood vessel tissue
                            # if model_name != 'segformer':
                            # class_num = [2]
                            # class_num = random.choice(class_num)
                            # aug_num = random.choice([0,1,2,3,5,6,7,8,9,10])
                            # # print('random_staff = ', aug_num, class_num)
                            # # if (true_masks == 5).sum()>0:
                            # #     nec_tis = True
                            # #     class_num = 5
                            #
                            # if aug_num > 2:
                            #     images, true_masks = tissue_aug_gpu(train_masks=true_masks, train_images=images,
                            #                                     num_classes=n_class, class_num1=class_num)


                            true_masks = true_masks.long()


                            # Augmentations
                            aug_num = random.choice([0,1,2,3,5,6,7,8,9,10])
                            # print('random_staff = ', aug_num, class_num)
                            if aug_num > 2:
                                if augmentation:
                                    images, true_masks = aug_pipeline(image=images, mask=true_masks)


                            torch.clamp(images, 0, 1)

                            true_masks = true_masks.long()


                            images, true_masks = Mine_resize(image=images, mask=true_masks, final_size=target_siz)

                            optimizer.zero_grad()

                            # Mixed Precision Training
                            with autocast(enabled=amp):
                                if model_name == 'segformer':
                                    batch_numpy = [img.permute(1, 2, 0).cpu().numpy() for img in
                                                   images[:,0:3]]  # (H, W, C) format
                                    # Process input images
                                    images1 = processor(images=batch_numpy,
                                                       return_tensors="pt")  # Now it's ready for SegFormer
                                    images1 = {key: value.to(device) for key, value in images1.items()}
                                    if images.shape[1]>3:
                                        images1['pixel_values'] = torch.concatenate((images1['pixel_values'],images[:,3].unsqueeze(1)),dim=1)
                                    masks_pred = model(**images1)
                                    masks_pred = F.interpolate(masks_pred.logits, size=true_masks.size()[1:],
                                                                    mode='bilinear', align_corners=False)
                                else:
                                    masks_pred = model(images)
                                # loss = 0.5 * criterion(masks_pred, true_masks)
                                loss1 = 0.5 * focal_loss_fn(masks_pred, true_masks)
                                loss2 = 0.5 * puma_dice_loss(masks_pred, true_masks)
                                # if loss1<0 or loss2<0:
                                #     print(f'Focal Loss: {loss1.item()}, Dice Loss: {loss2.item()}')

                                loss = loss1 + loss2



                            # Backward pass with gradient scaling
                            scaler.scale(loss).backward()

                            # Gradient clipping
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                            # Optimizer step
                            scaler.step(optimizer)
                            scaler.update()

                            # Update progress
                            epoch_loss += loss.item() * true_masks.size(0)  # Accumulate loss weighted by batch size
                            pbar.update(true_masks.shape[0])

                            masks_pred = F.softmax(masks_pred, dim=1)
                            # plt.imshow((masks_pred.argmax(dim=1)[0]).detach().cpu())
                            # plt.show()
                            # Calculate dice scores
                            with torch.no_grad():
                                dice_scores = torch.zeros(n_class, device=device)
                                for c in range(n_class):
                                    preds_class = (masks_pred.argmax(dim=1) == c).float()
                                    true_class = (true_masks == c).float()
                                    intersection = (preds_class * true_class).sum()
                                    union = preds_class.sum() + true_class.sum()
                                    dice_scores[c] = (2 * intersection + 1e-7) / (union + 1e-7)
                                epoch_dice += dice_scores * true_masks.size(0)  # Accumulate weighted Dice
                                # plt.imshow(masks_pred.argmax(dim=1)[0].cpu())
                                # plt.show()
                if phase == 'train':
                    epoch_loss /= n_train  # Divide by total training samples
                    epoch_dice /= n_train
                    print(epoch_dice.mean(), epoch_loss)
                elif phase == 'val' and epoch> val_sleep_time:  # Validation phase
                    if (model_name != 'segformer') or (epoch%1 == 0):
                        model.eval()
                        val_loss = 0
                        val_dice = torch.zeros(n_class, device=device)
                        total_val_images = 0  # Track total validation images
                        frame_type = []
                        with torch.no_grad():
                            for images, true_masks, inds in dataloaders[phase]:
                                images = images.to(device=device, dtype=torch.float32)
                                true_masks = true_masks.to(device=device, dtype=torch.long)
                                images, true_masks = Mine_resize(image=images, mask=true_masks, final_size=target_siz)

                                with autocast(enabled=amp):
                                    if model_name == 'segformer':
                                        batch_numpy = [img.permute(1, 2, 0).cpu().numpy() for img in
                                                       images]  # (H, W, C) format
                                        # Process input images
                                        images = processor(images=batch_numpy,
                                                           return_tensors="pt")  # Now it's ready for SegFormer
                                        images = {key: value.to(device) for key, value in images.items()}
                                        masks_pred = model(**images)
                                        masks_pred = F.interpolate(masks_pred.logits, size=true_masks.size()[1:],
                                                                        mode='bilinear', align_corners=False)
                                    else:
                                        masks_pred = model(images)
                                    masks_pred = F.softmax(masks_pred, dim=1)
                                    loss = puma_dice_loss(masks_pred, true_masks)

                                val_loss += loss.item() * true_masks.size(0)  # Accumulate loss weighted by batch size

                                # thresholds = [0.5, 0.5, 0.5, 0.7, 0.5, 0.8]
                                num_classes = masks_pred.shape[1]
                                #
                                # # Step 2: Apply thresholds to create binary masks
                                # binary_masks = torch.zeros_like(masks_pred, dtype=torch.float32)
                                # for i, thresh in enumerate(thresholds):
                                #     binary_masks[:, i, :, :] = i*(masks_pred[:, i, :, :] >= thresh).float()
                                #
                                # masks_pred1 = binary_masks[0].sum(dim=0)
                                # Calculate dice scores
                                dice_scores = torch.zeros(n_class, device=device)
                                for c in range(n_class):
                                    preds_class = (masks_pred.argmax(dim=1) == c).float()
                                    true_class = (true_masks == c).float()
                                    intersection = (preds_class * true_class).sum()
                                    union = preds_class.sum() + true_class.sum()
                                    dice_scores[c] = (2 * intersection + 1e-7) / (union + 1e-7)
                                val_dice += dice_scores * true_masks.size(0)
                                total_val_images += true_masks.size(0)

                    # Average metrics

                        if nuclei:
                            in_channels = 4
                        else:
                            in_channels = 3
                        mean_puma_dice, micro_dices, mean_micro_dice = compute_puma_dice_micro_dice(model=model,
                                                                                                    target_siz=(1024,1024),
                                                                                                    epoch=epoch,
                                                                                                    input_folder=input_folder,
                                                                                                    output_folder=output_folder,
                                                                                                    ground_truth_folder=ground_truth_folder,
                                                                                                    device=device,
                                                                                                    er_di = er_di,
                                                                                                    in_channels=in_channels,
                                                                                                    save_jpg=True)
                        if folds is None:
                            print("model micro val_score = ", micro_dices, mean_micro_dice)
                        mean_micro_dice[4] = 0.1*mean_micro_dice[4]# decrease model sensitivity to necrosis
                        if model_name == 'segformer':
                            mean_micro_dice[1] = 1 * mean_micro_dice[1]
                        micro_dices = np.mean(mean_micro_dice[0:num_classes-1])


                        total_dice = dice_cof * mean_puma_dice + m_dice_cof * micro_dices
                        scheduler.step(total_dice)
                        current_lr = optimizer.param_groups[0]['lr']
                        # print(f"Current Learning Rate: {current_lr:.7f}\n")
                        print("model val_score = ", micro_dices, mean_micro_dice)
                        # print("Dice Scores:", mean_puma_dice)
                        # print("Mean_Micro_Dice Scores:", mean_micro_dice, f"Current Learning Rate: {current_lr:.7f}\n")
                        if total_dice > best_val_score:
                            # print("saving best model val_score = ", micro_dices,mean_micro_dice)
                            # print("saving best model val_mean = ", val_mean)

                            best_val_score = total_dice
                            best_model_wts = model.state_dict()
                            mean_micro_dice[4] = 10 * mean_micro_dice[4]
                            if model_name == 'segformer':
                                mean_micro_dice[1] = 1 * mean_micro_dice[1]
                            best_dice_class = mean_micro_dice
                            # mean_puma_dice, micro_dices, mean_micro_dice = compute_puma_dice_micro_dice(model=model,
                            #                                                                             target_siz=(1024,1024),
                            #                                                                             epoch=epoch,
                            #                                                                             input_folder=input_folder,
                            #                                                                             output_folder=output_folder,
                            #                                                                             ground_truth_folder=ground_truth_folder,
                            #                                                                             device=device,
                            #                                                                             er_di=er_di,
                            #                                                                             save_jpg = True)
                            if logg:
                                logging1.info(f'''saving best model val_score =:
                                micro_dices:{micro_dices}
                                ''')
                            if save_checkpoint:
                                if folds is not None:
                                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                                    state_dict = model.state_dict()
                                    # state_dict['mask_values'] = dataset.mask_values
                                    torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1)))
                                else:
                                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                                    # state_dict['mask_values'] = dataset.mask_values
                                    torch.save(best_model_wts,
                                               str(dir_checkpoint / 'checkpoint_epoch{}'.format(1)) + str(
                                                   class_weights) + '.pth')
                        if test_images is not None:
                            test_puma_dice, test_micro_dices, test_mean_micro_dice = compute_puma_dice_micro_dice(model=model,
                                                                                                    target_siz=target_siz,
                                                                                                    epoch=epoch,
                                                                                                    input_folder=test_folder,
                                                                                                    output_folder=test_output_folder,
                                                                                                    ground_truth_folder=test_ground_truth_folder,
                                                                                                    device=device,
                                                                                                    er_di = er_di)
                            print('test puma dice: ', test_puma_dice, 'test_micro_dices = ', test_micro_dices, 'test_mean_micro_dice = ', test_mean_micro_dice)

                        try:
                            print('best_dice_class = ', best_dice_class)
                        except:
                            adfr = 0

                        # print(f"Validation Loss: {val_loss / len(dataloaders[phase])}")
                elif phase == 'folder_val':  # Validation phase from folders
                    model.eval()
                    with torch.no_grad():
                        mean_puma_dice, micro_dices, mean_micro_dice = compute_puma_dice_micro_dice(model=model,
                                                                                                    target_siz=target_siz,
                                                                                                    epoch=epoch,
                                                                                                    input_folder=input_folder,
                                                                                                    output_folder=output_folder,
                                                                                                    ground_truth_folder=ground_truth_folder,
                                                                                                    device=device,
                                                                                                    er_di = er_di,)
                        if folds is None:
                            print("model micro val_score = ", micro_dices, mean_micro_dice)
                        # mean_micro_dice[4] = 0.5*mean_micro_dice[4]
                        micro_dices = np.mean(mean_micro_dice[0:n_class-1])
                        # scheduler.step(val_dice.mean())

                        scheduler.step(0.5 * mean_puma_dice + 0.5 * micro_dices)
                        current_lr = optimizer.param_groups[0]['lr']
                        # print(f"Current Learning Rate: {current_lr:.7f}\n")

                        # print("Dice Scores:", mean_puma_dice)
                        # print("Mean_Micro_Dice Scores:", mean_micro_dice, f"Current Learning Rate: {current_lr:.7f}\n")
                        if  0.5 * mean_puma_dice + 0.5 * micro_dices > best_val_score:
                            # print("saving best model val_score = ", micro_dices,mean_micro_dice)
                            # print("saving best model val_mean = ", val_mean)

                            best_val_score = 0.5 * mean_puma_dice + 0.5 * micro_dices
                            best_model_wts = model.state_dict()
                            # mean_micro_dice[4] = 2 * mean_micro_dice[4]
                            best_dice_class = mean_micro_dice
                            mean_puma_dice, micro_dices, mean_micro_dice = compute_puma_dice_micro_dice(model=model,
                                                                                                        target_siz=target_siz,
                                                                                                        epoch=epoch,
                                                                                                        input_folder=input_folder,
                                                                                                        output_folder=output_folder,
                                                                                                        ground_truth_folder=ground_truth_folder,
                                                                                                        device=device,
                                                                                                        er_di=er_di,
                                                                                                        save_jpg = True)
                            if logg:
                                logging1.info(f'''saving best model val_score =:
                                micro_dices:{micro_dices}
                                ''')
                            if save_checkpoint:
                                if folds is not None:
                                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                                    state_dict = model.state_dict()
                                    # state_dict['mask_values'] = dataset.mask_values
                                    torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1)))
                                else:
                                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                                    # state_dict['mask_values'] = dataset.mask_values
                                    torch.save(best_model_wts,
                                               str(dir_checkpoint / 'checkpoint_epoch{}'.format(1)) + str(
                                                   class_weights) + '.pth')





                if phase == 'train':
                    epoch_loss /= n_train  # Divide by total training samples
                    epoch_dice /= n_train
                    # print(epoch_dice)
                else:
                    try:
                        val_loss /= total_val_images  # Divide by total validation samples
                        val_dice /= total_val_images

                        print(f"{phase.capitalize()} Loss: {epoch_loss if phase == 'train' else val_loss:.4f}")
                        print(
                            f"{phase.capitalize()} Dice: {epoch_dice.mean() if phase == 'train' else val_dice.mean():.4f}. Per-Class = {epoch_dice if phase == 'train' else val_dice}")
                    except:
                        print('not validated')

    try:
        model.load_state_dict(best_model_wts)
    except:
        print('no model loaded')
    if logg:
        for handler in logging1.handlers:
            handler.close()  # Close the handler
            logging1.removeHandler(handler)  # Remove the handler
        aaa = 10


    return model

