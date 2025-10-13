from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils import adjust_lr, structure_loss
from utils.misc import compute_metrix


def train_model(net, optimizer, train_loader, epoch, writer):
    net.train()
    ave_loss = 0
    num_img = 0
    
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='pack') as pbar:
        for pack in train_loader:
            torch.cuda.empty_cache()
            images, masks, vid = pack['images'].cuda(), pack['masks'].cuda(), pack['video_id']
            t, c, h, w = masks.size()
            target_1 = F.interpolate(masks, size=h // 4, mode='nearest').cuda()
            target_2 = F.interpolate(masks, size=h // 8, mode='nearest').cuda()
            target_3 = F.interpolate(masks, size=h // 16, mode='nearest').cuda()
            target_4 = F.interpolate(masks, size=h // 32, mode='nearest').cuda()
            target_5 = F.interpolate(masks, size=h // 64, mode='nearest').cuda()

            for i in range(len(images)):
                optimizer.zero_grad()
                image = images[i].unsqueeze(0)
                mask = masks[i].unsqueeze(0)
                sideout5, sideout4, sideout3, sideout2, sideout1, final, glb5, glb4, glb3, glb2, glb1, tokenattmap4, tokenattmap3, tokenattmap2, tokenattmap1, var_loss = net.forward(image, vid[i])
                loss1 = structure_loss(sideout5, target_4[i].unsqueeze(0))
                loss2 = structure_loss(sideout4, target_3[i].unsqueeze(0))
                loss3 = structure_loss(sideout3, target_2[i].unsqueeze(0))
                loss4 = structure_loss(sideout2, target_1[i].unsqueeze(0))
                loss5 = structure_loss(sideout1, target_1[i].unsqueeze(0))
                loss6 = structure_loss(final, mask)
                loss7 = structure_loss(glb5, target_5[i].unsqueeze(0))
                loss8 = structure_loss(glb4, target_4[i].unsqueeze(0))
                loss9 = structure_loss(glb3, target_3[i].unsqueeze(0))
                loss10 = structure_loss(glb2, target_2[i].unsqueeze(0))
                loss11 = structure_loss(glb1, target_2[i].unsqueeze(0))
                loss12 = structure_loss(tokenattmap4, target_3[i].unsqueeze(0))
                loss13 = structure_loss(tokenattmap3, target_2[i].unsqueeze(0))
                loss14 = structure_loss(tokenattmap2, target_1[i].unsqueeze(0))
                loss15 = structure_loss(tokenattmap1, target_1[i].unsqueeze(0))
                loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + 0.3*(loss7 + loss8 + loss9 + loss10 + loss11)+ 0.3*(loss12 + loss13 + loss14 + loss15) + var_loss
                Loss_loc = loss1 + loss2 + loss3 + loss4 + loss5 + loss6
                Loss_glb = loss7 + loss8 + loss9 + loss10 + loss11
                Loss_map = loss12 + loss13 + loss14 + loss15
                ave_loss += loss.item()
                num_img += 1
                
                loss.backward()
                optimizer.step()

            pbar.set_postfix({'loss': '{:.3f}'.format(loss)})
            pbar.update()
        
        writer.add_scalar('loss', ave_loss / num_img, epoch)
        adjust_lr(optimizer, epoch, decay_rate=0.9, decay_epoch=60)


def test_model(net, test_loader, epoch, writer):
    net.eval()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Val', unit='pack', total=len(test_loader))
        m_iou, m_dice, m_s_score, m_f_score, m_e_score, m_mae = 0, 0, 0, 0, 0, 0
        num_img= 0
        
        for pack in pbar:
            torch.cuda.empty_cache()
            images, masks, vid = pack['images'].cuda(), pack['masks'].cuda(), pack['video_id']
            for i in range(len(images)):
                image = images[i].unsqueeze(0)
                mask = masks[i].unsqueeze(0)
                pred = net.forward(image, vid[i])
                iou, dice, s_score, f_score, e_score, mae = compute_metrix(pred, mask, activation=torch.sigmoid)
                num_img += 1
                [m_iou, m_dice, m_s_score, m_f_score, m_e_score, m_mae] = [m + n for m, n in zip([m_iou, m_dice, m_s_score, m_f_score, m_e_score, m_mae], [iou, dice, s_score, f_score, e_score, mae])]
                
                if dice > 0.8:
                    writer.add_image('prediction', pred.squeeze(0), global_step=num_img)
                    writer.add_image('mask', mask.squeeze(0), global_step=num_img)
                    writer.add_image('image', image.squeeze(0), global_step=num_img)
            
            pbar.set_postfix({'dice': '{:.3f}'.format(dice)})
            pbar.update()
        
    m_iou /= num_img
    m_dice /= num_img
    m_s_score /= num_img
    m_f_score /= num_img
    m_e_score /= num_img
    m_mae /= num_img
    writer.add_scalar('IoU', m_iou, epoch)
    writer.add_scalar('Dice', m_dice, epoch)
    writer.add_scalar('S-measure', m_s_score, epoch)
    writer.add_scalar('F-measure', m_f_score, epoch)
    writer.add_scalar('E-measure', m_e_score, epoch)
    writer.add_scalar('MAE', m_mae, epoch)
    print(f'IoU: {m_iou}, Dice: {m_dice}, S-measure: {m_s_score}, F-measure: {m_f_score}, E-measure: {m_e_score}, MAE: {m_mae}')

    return m_dice