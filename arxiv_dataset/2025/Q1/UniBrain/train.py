from model import *
from utils import *
import time

def train_UniBrain(img_size,
          abn_stage,
          seg_stage,
          train_set_name,
          val_set_name,
          test_set_name,
          batch_size,
          num_epochs,
          learning_rate,
          model_name,
          reg_loss_name,
          save_every_epoch,
          weight_decay,
          if_train_aug,
          if_pred_aal,
          save_start_epoch,
          round_num,lambda_ext,lambda_reg,lambda_seg,lambda_cls):
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    model = model_name.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if reg_loss_name=="NCC":
        reg_loss_func=NCC().loss   
    elif reg_loss_name=="GCC":
        reg_loss_func=GCC()
    elif reg_loss_name=="MSE":
        reg_loss_func=nn.MSELoss().to(device)
    
    ext_loss_func=torch.nn.BCELoss()
    seg_loss_func=nn.CrossEntropyLoss()
    ssim_loss_func=mutual_information
    classifier_loss_func=torch.nn.CrossEntropyLoss()
    cc_loss=normalized_cross_correlation
    mi_loss=mutual_information
        
    cur_path = os.getcwd()
    result_path=cur_path+'/result'
    
    loss_log_path=result_path+'/loss_log'
    create_folder(result_path,'loss_log')
    
    sample_img_path=result_path+'/sample_img'
    create_folder(result_path,'sample_img')
    
    model_save_path=result_path+'/model'
    create_folder(result_path,'model')
    
    model_str=str(model)[0:str(model).find("(")]
    #smooth_str=str(smooth)[0:str(smooth).find("(")]
    
    lr_str=str(learning_rate)
    dataset_str=train_set_name[0:str(train_set_name).find(".")]
    
    seg_stage_str=str(seg_stage)
    abn_stage_str=str(abn_stage)
    
    weight_decay_str=str(weight_decay)
    round_num_str=str(round_num)
    if_train_aug_str=str(if_train_aug)
    if_pred_aal_str=str(if_pred_aal)
    
    lambda_ext_str=str(lambda_ext)
    lambda_reg_str=str(lambda_reg)
    lambda_seg_str=str(lambda_seg)
    lambda_cls_str=str(lambda_cls)
    
    
    modal_name=model_str+"_"+seg_stage_str+"_"+abn_stage_str+"_"+lr_str+"_"+weight_decay_str+"_"+dataset_str+"_"+if_train_aug_str+"_"+if_pred_aal_str+"_"+round_num_str+"_"+lambda_ext_str+"_"+lambda_reg_str+"_"+lambda_seg_str+"_"+lambda_cls_str
    
    modal_path=sample_img_path+"/"+modal_name
    create_folder(sample_img_path,modal_name)
    
    sample_o_path=modal_path+"/"+"o"
    sample_t_path=modal_path+"/"+"t"
    create_folder(modal_path,"o")
    create_folder(modal_path,"t")
    
    sample_reginv_mask_path=modal_path+"/"+"reginv_am"
    sample_segpred_mask_path=modal_path+"/"+"segpred_am"
    create_folder(modal_path,"reginv_am")
    create_folder(modal_path,"segpred_am")
    
    sample_ajc_m_path=modal_path+"/"+"ajc_m"
    create_folder(modal_path,"ajc_m")
    
    
    for i in range(int(seg_stage)):
        idx=i+1
        s_name="s_"+str(idx)
        s_mask_name="s_"+str(idx)+"_mask"
        sample_s_path=modal_path+"/"+s_name
        sample_s_mask_path=modal_path+"/"+s_mask_name
        
        create_folder(modal_path,s_name)
        create_folder(modal_path,s_mask_name)
    
    for q in range(int(abn_stage)):
        qdx=q+1
        r_name="r_"+str(qdx)
        r_grid_name="r_"+str(qdx)+"_grid"
        sample_r_path=modal_path+"/"+r_name
        sample_r_grid_path=modal_path+"/"+r_grid_name
        
        create_folder(modal_path,r_name)
        create_folder(modal_path,r_grid_name)
        
    modal_info="Model: {}    seg_stage: {}    abn_stage: {}          lr: {}    weight decay: {}    dataset: {}    if_train_aug:{}       pred_aal: {}      round: {}      lambda_ext: {}      lambda_reg: {}      lambda_seg: {}      lambda_cls: {}".format(model_str,seg_stage_str,abn_stage_str,lr_str,weight_decay_str,dataset_str,if_train_aug_str,if_pred_aal_str,round_num_str,lambda_ext_str,lambda_reg_str,lambda_seg_str,lambda_cls_str)
    
    
    #val
    modal_name_val=modal_name+"_val"
    create_log(modal_info,loss_log_path,modal_name_val)
    
    print (modal_info)

    train_loader=load_data_no_fix_v2(train_set_name,batch_size)
    val_loader=load_data_no_fix_v2(val_set_name,batch_size)
    test_loader=load_data_no_fix_v2(test_set_name,batch_size)
    
    
    for epoch in range(num_epochs):
        
        total_loss_train=[]
        total_seg_loss_train=[]
        total_sim_loss_train=[]
        total_classifier_loss_train=[]
        
        start=time.time()
        
        fixed_data=torch.from_numpy(np.load('./dataset/'+"template_img_orig_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        fixed_label_am=torch.from_numpy(np.load('./dataset/'+"template_img_gm_mask_orig_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        fixed_label_aal=torch.from_numpy(np.load('./dataset/'+"template_img_aal_mask_orig_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        fixed_data_val=torch.from_numpy(np.load('./dataset/'+"template_img_orig_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        fixed_label_am_val=torch.from_numpy(np.load('./dataset/'+"template_img_gm_mask_orig_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()
        fixed_label_aal_val=torch.from_numpy(np.load('./dataset/'+"template_img_aal_mask_orig_96.npy")).to(device).view(-1,1,img_size,img_size,img_size).float()

            
        for i, x in enumerate(train_loader):
            id,label,raw_img_np,raw_img_brain_mask_np,raw_img_gm_mask_np,raw_img_aal_mask_np=x
            moving_data,moving_label_bm=raw_img_np,raw_img_brain_mask_np
            
            #fixed_data=fixed_data.to(device).view(-1,1,img_size,img_size,img_size).float()
            moving_data=moving_data.to(device).view(-1,1,img_size,img_size,img_size).float()
            moving_label_bm=moving_label_bm.to(device).view(-1,1,img_size,img_size,img_size).float()
            label = label.to(device).long()
            
            
            if if_train_aug == True:
                       
                degree=2
                voxel=2
                scale_min=0.98   #0.98
                scale_max=1.02   #1.02
                
                moving_data,moving_label_bm=train_aug(moving_data,moving_label_bm,img_size,degree,voxel,scale_min,scale_max)

            optimizer.zero_grad()
            
            striped_list, mask_list, warped_list, theta_list, theta_list_inv, am_mov_pred, am_fix_2_moving, aal_mov_pred, aal_fix_2_moving, ajc_m, y_predicted = model(fixed_data,moving_data,fixed_label_am,fixed_label_aal,if_train=True)
            
            #ext_net
            ext_loss=ext_loss_func(mask_list[-1],moving_label_bm)
            
            #seg_net
            am_fix_2_moving_Nd=am_1d_2_Nd_torch(am_fix_2_moving)
            
            am_moving_pred_Nd=am_mov_pred
            
            seg_loss_train=seg_loss_func(am_moving_pred_Nd,am_fix_2_moving_Nd)  
            
            #reg_net
            sim_loss_train=reg_loss_func(warped_list[-1],fixed_data)
            
            #GNN_pred
            classifier_loss_train=classifier_loss_func(y_predicted,label)
            
            loss_train = lambda_reg*sim_loss_train + lambda_ext*ext_loss + lambda_seg*seg_loss_train + lambda_cls*classifier_loss_train
            
            loss_train.backward()
            optimizer.step()
            
            total_loss_train.append(loss_train.item())
            total_seg_loss_train.append(seg_loss_train.item())
            total_sim_loss_train.append(sim_loss_train.item())
            total_classifier_loss_train.append(classifier_loss_train.item())
            
        ave_loss_train,std_loss_train=get_ave_std(total_loss_train)
        ave_seg_loss_train,std_seg_loss_train=get_ave_std(total_seg_loss_train)
        ave_sim_loss_train,std_sim_loss_train=get_ave_std(total_sim_loss_train)
        ave_classifier_loss_train,std_classifier_loss_train=get_ave_std(total_classifier_loss_train)

        if epoch % save_every_epoch ==0:
            model.eval()
            with torch.no_grad():
                
                #ext
                total_ext_dice=[]
                total_ext_jaccard=[]

                #reg
                total_reg_mi=[]
                total_reg_cc=[]
                total_reg_dice=[]
                total_reg_jaccard=[]

                #seg
                total_seg_fix_inv_dice=[]
                total_seg_fix_inv_jaccard=[]
                total_seg_pred_dice=[]
                total_seg_pred_jaccard=[]

                #par
                total_par_dice=[]
                total_par_jaccard=[]

                #cls
                all_labels_val = []
                all_predictions_val = []
                all_probabilities_val = []
                
                inference_times = []
                
                for j, y in enumerate(val_loader):
                    id_val,label_val,raw_img_np_val,raw_img_brain_mask_np_val,raw_img_gm_mask_np_val,raw_img_aal_mask_np_val=y
                    moving_data_val,moving_label_bm_val,moving_label_am_val,moving_label_aal_val=raw_img_np_val,raw_img_brain_mask_np_val,raw_img_gm_mask_np_val,raw_img_aal_mask_np_val

                    moving_data_val=moving_data_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    moving_label_bm_val=moving_label_bm_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    moving_label_am_val=moving_label_am_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    moving_label_aal_val=moving_label_aal_val.to(device).view(-1,1,img_size,img_size,img_size).float()
                    label_val=label_val.to(device).long()

                    #am label 2 onehot 1d
                    #fixed_label_am_val=one_hot_1D_label(fixed_label_am_val).view(-1,1,img_size,img_size,img_size).float()
                    #moving_label_am_val=one_hot_1D_label(moving_label_am_val).view(-1,1,img_size,img_size,img_size).float()
                    start_time = time.time()
                    striped_list_val,mask_list_val,warped_list_val,theta_list_val,theta_list_inv_val,am_mov_pred_val,am_fix_2_moving_val, aal_mov_pred_val, aal_fix_2_moving_val, ajc_m_val, y_predicted_val= model(fixed_data_val,moving_data_val,fixed_label_am_val,fixed_label_aal_val,if_train=False)
                    end_time = time.time()
                    total_inference_time = end_time - start_time
                    inference_times.append(total_inference_time)
                    
                    #ext_eval
                    merged_mask_list_val=torch.prod(torch.cat(mask_list_val),0).unsqueeze(0)
                    dice_ext_val=f1_loss(merged_mask_list_val,moving_label_bm_val)
                    jaccard_ext_val=jaccard_loss(merged_mask_list_val,moving_label_bm_val)

                    total_ext_dice.append(dice_ext_val)
                    total_ext_jaccard.append(jaccard_ext_val)

                    #reg_eval
                    mi_reg_val=mi_loss(warped_list_val[-1],fixed_data_val)
                    cc_reg_val=cc_loss(warped_list_val[-1],fixed_data_val)

                    reg_grid=F.affine_grid(theta_list_val[-1], moving_label_am_val.size(),align_corners=True)
                    am_mov_2_fix_val=F.grid_sample(moving_label_am_val, reg_grid,
                                                              mode="nearest",align_corners=True,padding_mode="zeros")
                    dice_reg_val=compute_label_dice_v2(fixed_label_am_val,am_mov_2_fix_val,label="gm")
                    jaccard_reg_val=compute_label_jaccard_v2(fixed_label_am_val,am_mov_2_fix_val,label="gm")

                    total_reg_mi.append(mi_reg_val)
                    total_reg_cc.append(cc_reg_val)
                    total_reg_dice.append(dice_reg_val)
                    total_reg_jaccard.append(jaccard_reg_val)       

                    #seg_eval
                    am_moving_pred_Nd_val=am_mov_pred_val
                    am_moving_pred_1d_val=torch.argmax(am_moving_pred_Nd_val,axis=1)

                    dice_seg_fix_inv_val=compute_label_dice_v2(am_fix_2_moving_val,moving_label_am_val,label="gm")
                    jaccard_seg_fix_inv_val=compute_label_jaccard_v2(am_fix_2_moving_val,moving_label_am_val,label="gm")

                    dice_seg_pred_val=compute_label_dice_v2(am_moving_pred_1d_val,moving_label_am_val,label="gm")
                    jaccard_seg_pred_val=compute_label_jaccard_v2(am_moving_pred_1d_val,moving_label_am_val,label="gm")

                    total_seg_fix_inv_dice.append(dice_seg_fix_inv_val)
                    total_seg_fix_inv_jaccard.append(jaccard_seg_fix_inv_val)
                    total_seg_pred_dice.append(dice_seg_pred_val)
                    total_seg_pred_jaccard.append(jaccard_seg_pred_val)

                    #par_eval
                    dice_par_pred_val=compute_label_dice_v2(aal_fix_2_moving_val,moving_label_aal_val,label="aal")
                    jaccard_par_pred_val=compute_label_jaccard_v2(aal_fix_2_moving_val,moving_label_aal_val,label="aal")                

                    total_par_dice.append(dice_par_pred_val)
                    total_par_jaccard.append(jaccard_par_pred_val)

                    #cls
                    _, predicted = torch.max(y_predicted_val.data, 1)
                    probabilities = torch.nn.functional.softmax(y_predicted_val, dim=1)[:, 1].cpu().numpy()

                    all_labels_val.extend(label_val.cpu().numpy())
                    all_predictions_val.extend(predicted.cpu().numpy())
                    all_probabilities_val.extend(probabilities)                    

                #ext
                ave_ext_dice,std_ext_dice=get_ave_std(total_ext_dice)
                ave_ext_jaccard,std_ext_jaccard=get_ave_std(total_ext_jaccard)

                #reg
                ave_reg_mi,std_reg_mi=get_ave_std(total_reg_mi)
                ave_reg_cc,std_reg_cc=get_ave_std(total_reg_cc)
                ave_reg_dice,std_reg_dice=get_ave_std(total_reg_dice)
                ave_reg_jaccard,std_reg_jaccard=get_ave_std(total_reg_jaccard)

                #seg
                ave_seg_fix_inv_dice,std_seg_fix_inv_dice=get_ave_std(total_seg_fix_inv_dice)
                ave_seg_fix_inv_jaccard,std_seg_fix_inv_jaccard=get_ave_std(total_seg_fix_inv_jaccard)
                ave_seg_pred_dice,std_seg_pred_dice=get_ave_std(total_seg_pred_dice)
                ave_seg_pred_jaccard,std_seg_pred_jaccard=get_ave_std(total_seg_pred_jaccard)

                #par
                ave_par_dice,std_par_dice=get_ave_std(total_par_dice)
                ave_par_jaccard,std_par_jaccard=get_ave_std(total_par_jaccard)

                #acc,auc,f1
                acc_val = accuracy_score(all_labels_val, all_predictions_val)
                auc_val = roc_auc_score(all_labels_val, all_probabilities_val)
                macro_f1_val = f1_score(all_labels_val, all_predictions_val, average='macro')

                average_time_per_sample = sum(inference_times) / len(inference_times)
                
                loss_info="Epoch[{}/{}], Ext Dice: {:.4f}/{:.4f} , Ext Jaccard: {:.4f}/{:.4f} , Reg MI: {:.4f}/{:.4f} , Reg CC: {:.4f}/{:.4f} , Reg Dice: {:.4f}/{:.4f} , Reg Jaccard: {:.4f}/{:.4f}  ,  Seg fix_inv Dice: {:.4f}/{:.4f}   ,   Seg fix_inv Jaccard: {:.4f}/{:.4f}  ,   Seg pred Dice: {:.4f}/{:.4f}  ,  Seg pred Jaccard: {:.4f}/{:.4f}  ,  Par Dice: {:.4f}/{:.4f}  ,  Par Jaccard: {:.4f}/{:.4f}  ,  acc_val: {:.4f}  ,  auc_val: {:.4f}  ,  macro_f1_val: {:.4f}   ,  time: {:.6f}".format(epoch,num_epochs,
                ave_ext_dice,std_ext_dice,
                ave_ext_jaccard,std_ext_jaccard,
                ave_reg_mi,std_reg_mi,
                ave_reg_cc,std_reg_cc,
                ave_reg_dice,std_reg_dice,
                ave_reg_jaccard,std_reg_jaccard,
                ave_seg_fix_inv_dice,std_seg_fix_inv_dice,
                ave_seg_fix_inv_jaccard,std_seg_fix_inv_jaccard,
                ave_seg_pred_dice,std_seg_pred_dice,
                ave_seg_pred_jaccard,std_seg_pred_jaccard,
                ave_par_dice,std_par_dice,
                ave_par_jaccard,std_par_jaccard,
                acc_val,
                auc_val,
                macro_f1_val,average_time_per_sample
                )


                print (loss_info)
                append_log(loss_info,loss_log_path,modal_name_val)
                
                if epoch>save_start_epoch:
                    
                    save_sample_any(epoch,"o",fixed_data_val,sample_o_path)
                    #save_nii_any(epoch,"o",fixed_data_val,sample_o_path)

                    save_sample_any(epoch,"t",moving_data_val,sample_t_path)
                    #save_nii_any(epoch,"t",moving_data_val,sample_t_path)
                    
                    save_sample_any(epoch,"o_am",fixed_label_am_val,sample_o_path)
                    #save_nii_any(epoch,"o_am",fixed_label_am_val,sample_o_path)

                    save_sample_any(epoch,"t_am",moving_label_am_val,sample_t_path)
                    #save_nii_any(epoch,"t_am",moving_label_am_val,sample_t_path)                    
                    
                    save_sample_any(epoch,"reginv_am",am_fix_2_moving_val,sample_reginv_mask_path)
                    #save_nii_any(epoch,"reginv_am",am_fix_2_moving_val,sample_reginv_mask_path)  
                    
                    save_sample_any(epoch,"segpred_am",am_moving_pred_1d_val.unsqueeze(0),sample_segpred_mask_path)
                    #save_nii_any(epoch,"segpred_am",am_moving_pred_1d_val.unsqueeze(0),sample_segpred_mask_path) 
                    
                    
                    for t in range(int(seg_stage)):
                        tdx=t+1
                        s_name="s_"+str(tdx)
                        s_mask_name="s_"+str(tdx)+"_mask"

                        sample_s_path=modal_path+"/"+s_name
                        sample_s_mask_path=modal_path+"/"+s_mask_name

                        save_sample_any(epoch,s_name,striped_list_val[t],sample_s_path)
                        #save_nii_any(epoch,s_name,striped_list_val[t],sample_s_path)

                        save_sample_any(epoch,s_mask_name,mask_list_val[t],sample_s_mask_path)

                    for y in range(int(abn_stage)):
                        ydx=y+1
                        r_name="r_"+str(ydx)
                        sample_r_path=modal_path+"/"+r_name

                        save_sample_any(epoch,r_name,warped_list_val[y],sample_r_path)
                        #save_nii_any(epoch,r_name,warped_list_val[y],sample_r_path)
                    
                    save_adjacency_matrix(epoch, "ajc_m", ajc_m_val, sample_ajc_m_path)
                    torch.save(model.state_dict(), os.path.join(model_save_path,modal_name+"_"+str(epoch)+".pth"))
        
    return    
