
from torch import nn
import torch
from skimage.measure import regionprops, label
from torchvision.transforms import ToTensor, ToPILImage
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.metrics import  confusion_matrix, roc_curve, accuracy_score, precision_recall_fscore_support, auc,precision_recall_curve, average_precision_score

from utils.utils import create_dir
from torchvision.utils import save_image, make_grid
from torch.nn import functional as F
from PIL import Image
import nibabel as nib
import cv2 as cv

def save_predicted_volume(self,pred_volume, data_orig,patient_disease_id,smax,epoch):


        create_dir(self.cfg.save_sample_path + f"/validation/epoch_{epoch}/original/")
        create_dir(self.cfg.save_sample_path + f"/validation/epoch_{epoch}/reconstructed/")
        create_dir(self.cfg.save_sample_path + f"/validation/epoch_{epoch}/residual/")

        pred_volume = pred_volume.squeeze()
        
        #Select only the first MRI of the batch
        data_orig = data_orig[0,:,:,:,:]
        pred_volume = pred_volume[:1,:,:,:]
        # calculate the residual image
        diff_volume = torch.abs((data_orig-pred_volume))
                
        data_orig = np.squeeze( np.array(data_orig.cpu()* 255,dtype=np.int16))
        pred_volume =  np.squeeze(np.array(pred_volume.cpu()* 255,dtype=np.int16))
        diff_volume = np.squeeze(np.array(diff_volume.cpu()* 255, dtype=np.int16))
        
            
        
        img = nib.Nifti1Image(diff_volume , np.eye(4))
        nib.save(img, self.cfg.save_sample_path + f"/validation/epoch_{epoch}/residual/" + f"{patient_disease_id}_{smax}.nii.gz")
        img = nib.Nifti1Image(data_orig , np.eye(4))
        nib.save(img, self.cfg.save_sample_path + f"/validation/epoch_{epoch}/original/" + f"{patient_disease_id}_{smax}.nii.gz")
        img = nib.Nifti1Image(pred_volume , np.eye(4))
        nib.save(img, self.cfg.save_sample_path + f"/validation/epoch_{epoch}/reconstructed/" + f"{patient_disease_id}_{smax}.nii.gz")

import torchio as tio

def _save_predicted_volume(self, save_folder, final_volume,data_orig,image_path,image_visible=None) :
    
    """

    Used to save the residual volumes from Autoencoders after performing median filtering with different kernel sizes
    
    """

    # Define rescaling transform
    rescale_transform = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(1, 99)) 
    visible_image = None
    if image_visible is not None:
        
        image_visible = torch.squeeze(image_visible)
        image_visible = torch.unsqueeze(image_visible,dim=0)

        final_volume = torch.squeeze(final_volume)
        final_volume = torch.unsqueeze(final_volume,dim=0)

        data_orig = torch.squeeze(data_orig)
        data_orig = torch.unsqueeze(data_orig,dim=0)


        image_visible   = tio.ScalarImage(tensor=image_visible.cpu())
        image_inpainted = tio.ScalarImage(tensor=final_volume.cpu())
        image_original = tio.ScalarImage(tensor=data_orig.cpu())

        #rescaled_image_visible   = rescale_transform(image_visible)
        #rescaled_image_inpainted = rescale_transform(image_inpainted)
        #rescaled_image_original  = rescale_transform(image_original)

        final_volume = image_visible.data + image_inpainted.data
        #rescaled_image_inpainted = rescale_transform(final_volume)
        #visible_image = rescaled_image_visible.data


        diff_volume = torch.abs((image_original.data-final_volume.data))
        rescaled_image = torch.squeeze(diff_volume).numpy()

    
    else: 

        diff_volume = torch.abs((data_orig-final_volume))
        diff_volume = torch.squeeze(diff_volume)
        diff_volume = torch.unsqueeze(diff_volume,dim=0)

        # Define the RandomNoise augmentation
        image = tio.ScalarImage(tensor=diff_volume.cpu())

        rescaled_image = rescale_transform(image)
        rescaled_image   = rescaled_image.data.unsqueeze(0)
        rescaled_image = torch.squeeze(rescaled_image).numpy()

    

    x_mf_5  = apply_3d_median_filter(rescaled_image)
    x_mf_3  = apply_3d_median_filter(rescaled_image,kernelsize=3)
    x_mf_7  = apply_3d_median_filter(rescaled_image,kernelsize=7)

    x_mf_0 = torch.squeeze(torch.tensor(rescaled_image))
    x_mf_0 = x_mf_0.unsqueeze(0)


    x_mf_3 = torch.squeeze(torch.tensor(x_mf_3))
    x_mf_3 = x_mf_3.unsqueeze(0)

    x_mf_5 = torch.squeeze(torch.tensor(x_mf_5))
    x_mf_5 = x_mf_5.unsqueeze(0)

    x_mf_7 = torch.squeeze(torch.tensor(x_mf_7))
    x_mf_7 = x_mf_7.unsqueeze(0)


    final_volume = torch.squeeze(torch.tensor(final_volume.cpu()))
    final_volume = final_volume.unsqueeze(0)
    if visible_image is not None:
        
        visible_image = torch.squeeze(torch.tensor(visible_image.cpu()))
        visible_image = visible_image.unsqueeze(0)



    

    image_array = image_path[0].rsplit("/",1)
    image_root = image_array[0]
    image_name = image_array[1]


    save_path = image_root + f"/residuals/{save_folder}" #Hard coded model name... Need to change it but too lazy :(
    
    create_dir(save_path + "/MF_0")
    create_dir(save_path + "/MF_3")
    create_dir(save_path + "/MF_5")
    create_dir(save_path + "/MF_7")
    create_dir(save_path + "/MF_7")
    create_dir(save_path + "/Prediction_with_visible")
    create_dir(save_path + "/Visible_volume")


    rescaled_image   = tio.ScalarImage(tensor=x_mf_0)
    rescaled_image.save(save_path + "/MF_0/" + image_name)

    rescaled_image   = tio.ScalarImage(tensor=x_mf_5)
    rescaled_image.save(save_path + "/MF_5/" + image_name)

    rescaled_image   = tio.ScalarImage(tensor=x_mf_3)
    rescaled_image.save(save_path + "/MF_3/" + image_name)

    rescaled_image   = tio.ScalarImage(tensor=x_mf_7)
    rescaled_image.save(save_path + "/MF_7/" + image_name)
    """
    rescaled_image   = tio.ScalarImage(tensor=final_volume)
    rescaled_image.save(save_path + "/Prediction_with_visible/" + image_name)
    if visible_image is not None:
        rescaled_image   = tio.ScalarImage(tensor=visible_image)
        rescaled_image.save(save_path + "/Visible_volume/" + image_name)
    
    """
    # Save the new image
    #path = '/home/debayan/Desktop/MRI_HCHS/JAMA-labelled-1000-corrected/augmentation_samples/sub-2f67216f_polyps-r_0_smax_r_cc_65_mf_5.nii.gz'
    #



import time

    
def _test_step_multiclass(self,final_volume, latent_vector, data_orig,label_vol,disease_label_vol, patient_disease_id, crop_size,batch_idx, residual_volume=None, smax=None,final_volume_KLGrad=None,noisy_img=None,target=None,index=None) :
        

    #check if binary or multiclass
    
    #print("index",index,label_vol.item(),disease_label_vol.item(),self.eval_dict[index]['labelPerVol'])

    #slow execution by waiting for 10 ms
    
    
    eval_d = self.eval_dict[index].copy()
    eval_d['patient_disease_id'].append(patient_disease_id)
    eval_d['labelPerVol'].append(label_vol.item())
    eval_d['diseaseLabelPerVol'].append(disease_label_vol.item())
    eval_d['crop_size'].append(crop_size.item())
    self.eval_dict[index] = eval_d 
    
    

    #print('eval dict 0',self.eval_dict[0]['labelPerVol'])
    #print('eval dict 1',self.eval_dict[1]['labelPerVol'])
    #print('eval dict 2',self.eval_dict[2]['labelPerVol'])


        


        

def _test_step(self, final_volume, latent_vector, data_orig,label_vol,disease_label_vol, patient_disease_id, crop_size,batch_idx, residual_volume=None, smax=None,final_volume_KLGrad=None,noisy_img=None,target=None) :
        

        

        self.eval_dict['patient_disease_id'].append(patient_disease_id)
        self.eval_dict['labelPerVol'].append(label_vol.item())
        self.eval_dict['diseaseLabelPerVol'].append(disease_label_vol.item())
        self.eval_dict['crop_size'].append(crop_size.item())

        

        if final_volume is not None: 

            create_dir(self.cfg.save_sample_path + "/original/")
            create_dir(self.cfg.save_sample_path + "/reconstructed/")
            create_dir(self.cfg.save_sample_path + "/noisy_img/")
            create_dir(self.cfg.save_sample_path + "/residual/")
            create_dir(self.cfg.save_sample_path + "/target/")
            create_dir(self.cfg.save_sample_path + "/residual_after_mf_3/")
            create_dir(self.cfg.save_sample_path + "/residual_after_mf_5/")
            create_dir(self.cfg.save_sample_path + "/residual_after_mf_7/")

            create_dir(self.cfg.save_sample_path + "/residual_3d_connected/")

            create_dir(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_0_org/")
            create_dir(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_1_org/")
            create_dir(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_2_org/")

            create_dir(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_0_mf5/")
            create_dir(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_1_mf5/")
            create_dir(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_2_mf5/")
            final_volume = final_volume.squeeze()

            
            #Select only the first MRI of the batch
            data_orig = data_orig[0,:,:,:,:]

            if noisy_img is not None:
                noisy_img = noisy_img[0,:,:,:,:]

            if target is not None:
                target = target[0,:,:,:,:]

                
            
            if len(final_volume.shape) > 3:
                final_volume = final_volume[:1,:,:,:]
            
            # calculate the residual image
            diff_volume = torch.abs((data_orig-final_volume))
            
            
            if residual_volume is not None:
                diff_volume = torch.abs((diff_volume-residual_volume))

            # Calculate Reconstruction errors with respect to anomal/normal regions
            l1err = nn.functional.l1_loss(final_volume.squeeze(),data_orig.squeeze())
            l2err = nn.functional.mse_loss(final_volume.squeeze(),data_orig.squeeze())
            
            # store in eval dict
            try:
                self.eval_dict['l1recoErrorAll'].append(l1err.item())
                self.eval_dict['l2recoErrorAll'].append(l2err.item())

                if latent_vector is not None: self.eval_dict['latentSpaceAll'].append(latent_vector.cpu().numpy()) 
                if label_vol.item() == 0: #Healthy 
                    self.eval_dict['l1recoErrorHealthy'].append(l1err.item()) 
                    self.eval_dict['l2recoErrorHealthy'].append(l2err.item()) 
                    
                    if latent_vector is not None: self.eval_dict['latentSpaceHealthy'].append(latent_vector)
                elif  label_vol.item() == 1: #Unhealthy 
                    self.eval_dict['l1recoErrorUnhealthy'].append(l1err.item()) 
                    self.eval_dict['l2recoErrorUnhealthy'].append(l2err.item()) 
                    
                    if latent_vector is not None: self.eval_dict['latentSpaceUnhealthy'].append(latent_vector)
            except:
                print("Look at utils eval, something not working")

            # move data to CPU and save the original, reconstructed and residual volumes coming from the test dataset
            
            diff_volume = np.squeeze(np.array(diff_volume.cpu()* 255, dtype=np.int16))
            
            #axis_0_images,axis_1_images,axis_2_images = apply_colormap_3D(self,diff_volume)

            data_orig = np.squeeze( np.array(data_orig.cpu()* 255,dtype=np.int16))
            if noisy_img is not None: noisy_img = np.squeeze( np.array(noisy_img.cpu()* 255,dtype=np.int16))
            if target is not None: target = np.squeeze( np.array(target.cpu()))
            final_volume =  np.squeeze(np.array(final_volume.cpu()* 255,dtype=np.int16))
            
            #Median filtering the residual volume 
            #diff_volume_medfilter_3    = apply_3d_median_filter(np.copy(diff_volume),kernelsize=3)
            #diff_volume_medfilter_5    = apply_3d_median_filter(np.copy(diff_volume),kernelsize=5)
            #diff_volume_medfilter_7    = apply_3d_median_filter(np.copy(diff_volume),kernelsize=7)
            
            
            #diff_volume_connected_c    = filter_3d_connected_components(np.copy(diff_volume))
            """
            
            axis_0_images_mf,axis_1_images_mf,axis_2_images_mf = apply_colormap_3D(self,diff_volume_medfilter_5)

            for i in range(0,axis_0_images.shape[0]): 

                img_0 = axis_0_images[i,:,:,:]
                img_1 = axis_1_images[i,:,:,:]
                img_2 = axis_2_images[i,:,:,:]

                cv.imwrite(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_0_org/{i}.png",img_0)
                cv.imwrite(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_1_org/{i}.png",img_1)
                cv.imwrite(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_2_org/{i}.png",img_2)

                img_0 = axis_0_images_mf[i,:,:,:]
                img_1 = axis_1_images_mf[i,:,:,:]
                img_2 = axis_2_images_mf[i,:,:,:]

                cv.imwrite(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_0_mf5/{i}.png",img_0)
                cv.imwrite(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_1_mf5/{i}.png",img_1)
                cv.imwrite(self.cfg.save_sample_path + f"/color_map/{patient_disease_id}/axis_2_mf5/{i}.png",img_2)
            """
            #img = nib.Nifti1Image(diff_volume_medfilter_3 , np.eye(4))
            #nib.save(img, self.cfg.save_sample_path + "/residual_after_mf_3/" + f"{patient_disease_id}.nii.gz")

            #img = nib.Nifti1Image(diff_volume_medfilter_5 , np.eye(4))
            #nib.save(img, self.cfg.save_sample_path + "/residual_after_mf_5/" + f"{patient_disease_id}.nii.gz")

            #img = nib.Nifti1Image(diff_volume_medfilter_7 , np.eye(4))
            #nib.save(img, self.cfg.save_sample_path + "/residual_after_mf_7/" + f"{patient_disease_id}.nii.gz")

            #img = nib.Nifti1Image(diff_volume_connected_c , np.eye(4))
            #nib.save(img, self.cfg.save_sample_path + "/residual_3d_connected/" + f"{patient_disease_id}.nii.gz")

            img = nib.Nifti1Image(diff_volume , np.eye(4))
            nib.save(img, self.cfg.save_sample_path + "/residual/" + f"{patient_disease_id}_{smax}.nii.gz")


            if noisy_img is not None: 
                img = nib.Nifti1Image(noisy_img , np.eye(4))
                nib.save(img, self.cfg.save_sample_path + "/noisy_img/" + f"{patient_disease_id}_{smax}.nii.gz")

            if target is not None: 
                img = nib.Nifti1Image(target , np.eye(4))
                nib.save(img, self.cfg.save_sample_path + "/target/" + f"{patient_disease_id}_{smax}.nii.gz")

            
            img = nib.Nifti1Image(data_orig , np.eye(4))
            nib.save(img, self.cfg.save_sample_path + "/original/" + f"{patient_disease_id}_{smax}.nii.gz")

            img = nib.Nifti1Image(final_volume , np.eye(4))
            nib.save(img, self.cfg.save_sample_path + "/reconstructed/" + f"{patient_disease_id}_{smax}.nii.gz")

        

def _test_end(self) :
    # average over all test samples
   
    self.eval_dict['l1recoErrorAllMean'] = np.mean(self.eval_dict['l1recoErrorAll'])
    self.eval_dict['l1recoErrorAllStd'] = np.std(self.eval_dict['l1recoErrorAll'])
    self.eval_dict['l2recoErrorAllMean'] = np.mean(self.eval_dict['l2recoErrorAll'])
    self.eval_dict['l2recoErrorAllStd'] = np.std(self.eval_dict['l2recoErrorAll'])

    self.eval_dict['l1recoErrorHealthyMean'] = np.mean(self.eval_dict['l1recoErrorHealthy'])
    self.eval_dict['l1recoErrorHealthyStd'] = np.std(self.eval_dict['l1recoErrorHealthy'])
    self.eval_dict['l1recoErrorUnhealthyMean'] = np.mean(self.eval_dict['l1recoErrorUnhealthy'])
    self.eval_dict['l1recoErrorUnhealthyStd'] = np.std(self.eval_dict['l1recoErrorUnhealthy'])

    self.eval_dict['l2recoErrorHealthyMean'] = np.mean(self.eval_dict['l2recoErrorHealthy'])
    self.eval_dict['l2recoErrorHealthyStd'] = np.std(self.eval_dict['l2recoErrorHealthy'])
    self.eval_dict['l2recoErrorUnhealthyMean'] = np.mean(self.eval_dict['l2recoErrorUnhealthy'])
    self.eval_dict['l2recoErrorUnhealthyStd'] = np.std(self.eval_dict['l2recoErrorUnhealthy'])

def get_eval_dictionary():
    _eval = {
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'diffs_volume': [],
        'latentSpaceAll': [],
        'latentSpaceHealthy': [],
        'latentSpaceUnhealthy': [],
        'l1recoErrorAll': [],
        'l1recoErrorUnhealthy': [],
        'l1recoErrorHealthy': [],
        'l2recoErrorAll': [],
        'l2recoErrorUnhealthy': [],
        'l2recoErrorHealthy': [],

        'KLD_to_learned_prior' : [],
        'patient_disease_id' : [],
        
        'AnomalyScoreRegPerVol' : [],
        'AnomalyScoreCombiPerVol' : [],
        'AnomalyScoreCombiPerVolL2' : [],
        'AnomalyScoreCombPriorPerVol' : [],
        'AnomalyScoreCombPriorPerVolL2' : [],
        'AnomalyScoreRecoPerVol' : [],
        'AnomalyScoreRecoPerVolL2' : [],


        'smax' : [],
        'crop_size' : [],
        'AnomalyScorePerVol' : [],
        'AnomalyScorePerVol_std' : [],
        'AnomalyScorePerVol_one_instance' : [],
        'AnomalyScorePerVol_voting' : [],
        'labelPerVol':[],
        'diseaseLabelPerVol':[],
        'patient_disease_id':[],

        'l1recoErrorAllMean':0.0,
        'l1recoErrorAllSftd':0.0,
        'l1recoErrorHealthyMean':0.0,
        'l1recoErrorHealthyStd':0.0,
        'l1recoErrorUnhealthyMean':0.0,
        'l1recoErrorUnhealthyStd':0.0,

        'l2recoErrorAllMean':0.0,
        'l2recoErrorAllStd':0.0,
        'l2recoErrorHealthyMean':0.0,
        'l2recoErrorHealthyStd':0.0,
        'l2recoErrorUnhealthyMean':0.0,
        'l2recoErrorUnhealthyStd':0.0,
    
        'labelPerVol':[]

    }
    return _eval


def get_eval_dictionary_classification():
    _eval = {
        'x': [],
        'reconstructions': [],
        'diffs': [],
        'diffs_volume': [],
        
        'smax' : [],
        'crop_size' : [],
        'AnomalyScorePerVol' : [],
        'AnomalyScorePerVol_std' : [],
        'AnomalyScorePerVol_one_instance' : [],
        'AnomalyScorePerVol_voting' : [],
        'labelPerVol':[],
        'diseaseLabelPerVol':[],
        'patient_disease_id':[]

    }
    return _eval

def calc_thresh(data,plot_save_path):

    create_dir(plot_save_path)
    
    
    _, fpr_healthy_comb, _, threshs_healthy_comb = compute_roc(np.array(data['AnomalyScoreCombiPerVol']),np.array(data['labelPerVol'])) 
    _, fpr_healthy_combPrior, _, threshs_healthy_combPrior = compute_roc(np.array(data['AnomalyScoreCombPriorPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_reg, _, threshs_healthy_reg = compute_roc(np.array(data['AnomalyScoreRegPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_reco, _, threshs_healthy_reco = compute_roc(np.array(data['AnomalyScoreRecoPerVol']),np.array(data['labelPerVol']))
    _, fpr_healthy_prior_kld, _, threshs_healthy_prior_kld = compute_roc(np.array(data['KLD_to_learned_prior']),np.array(data['labelPerVol']))
    #L1
    _, precision_comb, recall_comb, threshs_healthy_comb_prc = compute_prc(np.array(data['AnomalyScoreCombiPerVol']),np.array(data['labelPerVol'])) 
    f1_comb = (2 * precision_comb * recall_comb) / (precision_comb + recall_comb)
    #L2
    _, precision_combL2, recall_combL2, threshs_healthy_comb_prcL2 = compute_prc(np.array(data['AnomalyScoreCombiPerVolL2']),np.array(data['labelPerVol'])) 
    f1_combL2 = (2 * precision_combL2 * recall_combL2) / (precision_combL2 + recall_combL2)
    #L1
    _, precision_comb_prior, recall_comb_prior, threshs_healthy_combPrior_prc = compute_prc(np.array(data['AnomalyScoreCombPriorPerVol']),np.array(data['labelPerVol']))
    f1_comb_prior = (2 * precision_comb_prior * recall_comb_prior) / (precision_comb_prior + recall_comb_prior)
    #L2
    _, precision_comb_priorL2, recall_comb_priorL2, threshs_healthy_combPrior_prcL2 = compute_prc(np.array(data['AnomalyScoreCombPriorPerVolL2']),np.array(data['labelPerVol']))
    f1_comb_priorL2 = (2 * precision_comb_priorL2 * recall_comb_priorL2) / (precision_comb_priorL2 + recall_comb_priorL2)
    
    _, precision_comb_reg, recall_comb_reg, threshs_healthy_reg_prc = compute_prc(np.array(data['AnomalyScoreRegPerVol']),np.array(data['labelPerVol']))
    f1_comb_reg = (2 * precision_comb_reg * recall_comb_reg) / (precision_comb_reg + recall_comb_reg)
    #L1
    _, precision_comb_reco, recall_comb_reco, threshs_healthy_reco_prc = compute_prc(np.array(data['AnomalyScoreRecoPerVol']),np.array(data['labelPerVol']))
    f1_comb_reco = (2 * precision_comb_reco * recall_comb_reco) / (precision_comb_reco + recall_comb_reco)
    #L2
    _, precision_comb_recoL2, recall_comb_recoL2, threshs_healthy_reco_prcL2 = compute_prc(np.array(data['AnomalyScoreRecoPerVolL2']),np.array(data['labelPerVol']))
    f1_comb_recoL2 = (2 * precision_comb_recoL2 * recall_comb_recoL2) / (precision_comb_recoL2 + recall_comb_recoL2)

    
    _, precision_prior_kld, recall_prior_kld, threshs_healthy_prior_kld_prc = compute_prc(np.array(data['KLD_to_learned_prior']),np.array(data['labelPerVol']))
    f1_prior_kld = (2 * precision_prior_kld * recall_prior_kld) / (precision_prior_kld + recall_prior_kld)

    
    f1_comb = np.nan_to_num(f1_comb) 
    f1_combL2 = np.nan_to_num(f1_combL2) 
    f1_comb_prior = np.nan_to_num(f1_comb_prior) 
    f1_comb_priorL2 = np.nan_to_num(f1_comb_priorL2) 
    f1_comb_reg = np.nan_to_num(f1_comb_reg) 
    f1_comb_reco = np.nan_to_num(f1_comb_reco)
    f1_comb_recoL2 = np.nan_to_num(f1_comb_recoL2) 
    f1_prior_kld = np.nan_to_num(f1_prior_kld) 

    fig, ax = plt.subplots()
    ax.plot(recall_comb_reco, precision_comb_reco, color='green')
    

    best_prec = precision_comb_reco[ np.argmax(f1_comb_reco)]
    best_rec  = recall_comb_reco[ np.argmax(f1_comb_reco)]

    ax.plot(best_rec,best_prec, marker="x", markersize=5, markeredgecolor="red", markerfacecolor="red")

    

     

    best_precl2 = precision_comb_recoL2[ np.argmax(f1_comb_recoL2)]
    best_recl2  = recall_comb_recoL2[ np.argmax(f1_comb_recoL2)]

    ax.plot(recall_comb_recoL2, precision_comb_recoL2, color='yellow')
    ax.plot(best_recl2,best_precl2, marker="x", markersize=5, markeredgecolor="red", markerfacecolor="red")

    fig.savefig(plot_save_path + "prc_curve_reco.png", dpi=500)
    plt.clf() 

    


    plt.close()
    

    threshholds_healthy= {
                'thresh_1p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.01)], 
                'thresh_1p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.01)], 
                'thresh_1p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.01)], 
                'thresh_1p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.01)], 
                'thresh_1p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.01)], 
                'thresh_5p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.05)], 
                'thresh_5p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.05)], 
                'thresh_5p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.05)], 
                'thresh_5p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.05)], 
                'thresh_5p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.05)], 
                'thresh_10p_comb' : threshs_healthy_comb[np.argmax(fpr_healthy_comb>0.1)], 
                'thresh_10p_combPrior' : threshs_healthy_combPrior[np.argmax(fpr_healthy_combPrior>0.1)],
                'thresh_10p_reg' : threshs_healthy_reg[np.argmax(fpr_healthy_reg>0.1)], 
                'thresh_10p_reco' : threshs_healthy_reco[np.argmax(fpr_healthy_reco>0.1)],
                'thresh_10p_prior_kld' : threshs_healthy_prior_kld[np.argmax(fpr_healthy_prior_kld>0.1)],
                'thresh_prc_comb' : threshs_healthy_comb_prc[np.argmax(f1_comb)],
                'thresh_L2_cmb_' : threshs_healthy_comb_prcL2[np.argmax(f1_combL2)],
                'thresh_prc_combPrior' : threshs_healthy_combPrior_prc[np.argmax(f1_comb_prior)],
                'thresh_prc_L2_cmbPrior' : threshs_healthy_combPrior_prcL2[np.argmax(f1_comb_priorL2)],
                'thresh_prc_reg' : threshs_healthy_reg_prc[np.argmax(f1_comb_reg)],
                'thresh_prc_reco' : threshs_healthy_reco_prc[np.argmax(f1_comb_reco)],
                'thresh_L2_rcns' : threshs_healthy_reco_prcL2[np.argmax(f1_comb_recoL2)],
                'thresh_prc_prior_kld' : threshs_healthy_prior_kld_prc[np.argmax(f1_prior_kld)],
                
                } 

    return threshholds_healthy



def calc_thresh_classification_multiclass(data,plot_save_path=None):

    #Calculate ROC and PRC curves for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    thresh_roc = dict()
    thresh_prc = dict()
    thresholds = dict()
    f1 = dict()

    for i in range(0,3):
        
        #create binary labels for each class 
        labels = np.where(np.array(data['labelPerVol']) == i, 1, 0) 

        print(np.array(data['AnomalyScorePerVol']))
        fpr[i], tpr[i], _, thresh_roc[i] = compute_roc(np.array(data['AnomalyScorePerVol']),np.array(data['labelPerVol'])) 
        precision[i], recall[i], thresh_prc[i] = compute_prc(np.array(data['AnomalyScorePerVol']),np.array(data['labelPerVol'])) 
        f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i])    
        f1[i] = np.nan_to_num(f1[i])

        best_prec = precision[i][ np.argmax(f1[i])] 
        best_rec  = recall[i][ np.argmax(f1[i])]

        if plot_save_path is not None:

            create_dir(plot_save_path)
            fig, ax = plt.subplots()
            ax.plot(recall[i], precision[i], color='green')
            ax.plot(best_rec,best_prec, marker="x", markersize=5, markeredgecolor="red", markerfacecolor="red")
            fig.savefig(plot_save_path + '/prc_class_' + str(i) + '.png')
            plt.close()
            plt.clf()

        threshholds_class = { 
                'thresh_1p_class_' + str(i) : thresh_roc[i][np.argmax(fpr[i]>0.01)], 
                'thresh_5p_class_' + str(i) : thresh_roc[i][np.argmax(fpr[i]>0.05)], 
                'thresh_10p_class_' + str(i) : thresh_roc[i][np.argmax(fpr[i]>0.1)], 
                'thresh_prc_class_' + str(i) : thresh_prc[i][np.argmax(f1[i])],
                }
    
        thresholds[i] = threshholds_class
        
    



def calc_thresh_classification(data,plot_save_path=None):

    
    
    _, fpr, _, thresh_roc = compute_roc(np.array(data['AnomalyScorePerVol']),np.array(data['labelPerVol'])) 
   
    _, precision, recall, thresh_prc = compute_prc(np.array(data['AnomalyScorePerVol']),np.array(data['labelPerVol'])) 
    f1 = (2 * precision * recall) / (precision + recall)    
    f1 = np.nan_to_num(f1) 
    
    
    best_prec = precision[ np.argmax(f1)]
    best_rec  = recall[ np.argmax(f1)]

    

    if plot_save_path is not None:
        create_dir(plot_save_path)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='green')
        ax.plot(best_rec,best_prec, marker="x", markersize=5, markeredgecolor="red", markerfacecolor="red")
        fig.savefig(plot_save_path + "prc_curve.png", dpi=500)
        plt.clf() 

        plt.close()
        

    threshholds_anomaly= {
                'thresh_1p_roc' : thresh_roc[np.argmax(fpr>0.01)], 
                'thresh_1p_prc' : thresh_prc[np.argmax(f1)], 

                'thresh_5p_roc' : thresh_roc[np.argmax(fpr>0.05)], 
                'thresh_5p_prc' : thresh_prc[np.argmax(f1)], 

                'thresh_10p_roc' : thresh_roc[np.argmax(fpr>0.1)], 
                'thresh_10p_prc' : thresh_prc[np.argmax(f1)], 
                
                } 

    return threshholds_anomaly



def redFlagEvaluation_einscanner(Set,thresh):

    
    eval_dict_redflag = {}
    if thresh is not None:
        threshholds_healthy = thresh
        eval_dict_redflag['redflag_thresholds'] = threshholds_healthy
##
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreCombiPerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreCombiPerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolComb'] = AUC
    eval_dict_redflag['AUPRCperVolComb'] = AUPRC

    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreCombiPerVolL2']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreCombiPerVolL2']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolCombL2'] = AUC
    eval_dict_redflag['AUPRCperVolCombL2'] = AUPRC
    
    # KL Term For each volume 
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreRegPerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreRegPerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolReg'] = AUC
    eval_dict_redflag['AUPRCperVolReg'] = AUPRC
    # Reconstruction Term for each volume L1
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreRecoPerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreRecoPerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolReco'] = AUC
    eval_dict_redflag['AUPRCperVolReco'] = AUPRC

    # Reconstruction Term for each volume L2
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreRecoPerVolL2']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreRecoPerVolL2']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolRecoL2'] = AUC
    eval_dict_redflag['AUPRCperVolRecoL2'] = AUPRC
    # KLD to Prior Term for each volume 
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['KLD_to_learned_prior']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['KLD_to_learned_prior']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolKLDPrior'] = AUC
    eval_dict_redflag['AUPRCperVolKLDPrior'] = AUPRC
    # Reconstruction plus KLD to Prior Term for each volume L1
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreCombPriorPerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreCombPriorPerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolCombPrior'] = AUC
    eval_dict_redflag['AUPRCperVolCombPrior'] = AUPRC

    # Reconstruction plus KLD to Prior Term for each volume L2
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreCombPriorPerVolL2']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreCombPriorPerVolL2']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolCombPriorL2'] = AUC
    eval_dict_redflag['AUPRCperVolCombPriorL2'] = AUPRC
    eval_dict_redflag['disease_to_patient_id'] = Set['patient_disease_id']
    eval_dict_redflag['labels'] = Set['labelPerVol']
    eval_dict_redflag['disease_labels'] = Set['diseaseLabelPerVol']
    # Threshold-based metrics
    if thresh is not None:
        for t in threshholds_healthy:

            if 'comb' in t:
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVol']) > threshholds_healthy[t] ,pos_label=1, average='binary')
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVol']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreCombiPerVol'] > threshholds_healthy[t]
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVol']) > threshholds_healthy[t])
            elif 'reg' in t:
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRegPerVol']) > threshholds_healthy[t] ,pos_label=1, average='binary')
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRegPerVol']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreRegPerVol'] > threshholds_healthy[t]
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRegPerVol']) > threshholds_healthy[t])
            elif 'reco' in t: 
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVol']) > threshholds_healthy[t] ,pos_label=1, average='binary')

                
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVol']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreRecoPerVol'] > threshholds_healthy[t]

                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVol']) > threshholds_healthy[t])
            
            elif 'L2_cmb' in t:
                print(t,"1")
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVolL2']) > threshholds_healthy[t] ,pos_label=1, average='binary')
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVolL2']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreCombiPerVolL2'] > threshholds_healthy[t]
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVolL2']) > threshholds_healthy[t])


            elif 'L2_rcns' in t:
                print(t,"2")
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVolL2']) > threshholds_healthy[t] ,pos_label=1, average='binary')
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVolL2']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreRecoPerVolL2'] > threshholds_healthy[t]
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVolL2']) > threshholds_healthy[t])
           


            

          
    return eval_dict_redflag




def redFlagEvaluation_einscanner(Set,thresh):

    
    eval_dict_redflag = {}
    if thresh is not None:
        threshholds_healthy = thresh
        eval_dict_redflag['redflag_thresholds'] = threshholds_healthy
##
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreCombiPerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreCombiPerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolComb'] = AUC
    eval_dict_redflag['AUPRCperVolComb'] = AUPRC

    eval_dict_redflag['ReconError_Comb'] = list(Set['AnomalyScoreCombiPerVol'])
    eval_dict_redflag['FPR_Comb'] = list(_fpr)
    eval_dict_redflag['TPR_Comb'] = list(_tpr)
    eval_dict_redflag['Prec_PRCurve_Comb'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_Comb'] =   list(_recalls)

    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreCombiPerVolL2']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreCombiPerVolL2']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolCombL2'] = AUC
    eval_dict_redflag['AUPRCperVolCombL2'] = AUPRC

    eval_dict_redflag['FPR_L_2Comb'] = list(_fpr)
    eval_dict_redflag['TPR_L_2Comb'] = list(_tpr)
    eval_dict_redflag['Prec_PRCurve_L_2Comb'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_L_2Comb'] =   list(_recalls)
    
    # KL Term For each volume 
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreRegPerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreRegPerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolReg'] = AUC
    eval_dict_redflag['AUPRCperVolReg'] = AUPRC

    eval_dict_redflag['FPR_Reg'] = list(_fpr)
    eval_dict_redflag['TPR_Reg'] = list(_tpr)
    eval_dict_redflag['Prec_PRCurve_Reg'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_Reg'] =   list(_recalls)

    # Reconstruction Term for each volume L1
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreRecoPerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreRecoPerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolReco'] = AUC
    eval_dict_redflag['AUPRCperVolReco'] = AUPRC
    eval_dict_redflag['ReconError_L1'] = list(Set['AnomalyScoreRecoPerVol'])
    eval_dict_redflag['FPR_L1'] = list(_fpr)
    eval_dict_redflag['TPR_L1'] = list(_tpr)
    eval_dict_redflag['Prec_PRCurve_L1'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_L1'] =   list(_recalls)

    # Reconstruction Term for each volume L2
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreRecoPerVolL2']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreRecoPerVolL2']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolRecoL2'] = AUC
    eval_dict_redflag['AUPRCperVolRecoL2'] = AUPRC
    eval_dict_redflag['ReconError_L2'] = list(Set['AnomalyScoreRecoPerVolL2'])
    eval_dict_redflag['FPR_L2'] = list(_fpr)
    eval_dict_redflag['TPR_L2'] = list(_tpr)
    eval_dict_redflag['Prec_PRCurve_L2'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_L2'] =   list(_recalls)

    # KLD to Prior Term for each volume 
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['KLD_to_learned_prior']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['KLD_to_learned_prior']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolKLDPrior'] = AUC
    eval_dict_redflag['AUPRCperVolKLDPrior'] = AUPRC
    
    eval_dict_redflag['FPR_KLDPrior'] = list(_fpr)
    eval_dict_redflag['TPR_KLDPrior'] = list(_tpr)
    eval_dict_redflag['Prec_PRCurve_KLDPrior'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_KLDPrior'] =   list(_recalls)


    # Reconstruction plus KLD to Prior Term for each volume L1
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreCombPriorPerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreCombPriorPerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUCperVolCombPrior'] = AUC
    eval_dict_redflag['AUPRCperVolCombPrior'] = AUPRC

    eval_dict_redflag['FPR_PriorComb'] = list(_fpr)
    eval_dict_redflag['TPR_PriorComb'] = list(_tpr)
    eval_dict_redflag['Prec_PRCurve_PriorComb'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_PriorComb'] =   list(_recalls)


    # Reconstruction plus KLD to Prior Term for each volume L2
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScoreCombPriorPerVolL2']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScoreCombPriorPerVolL2']),np.array(Set['labelPerVol']))

    eval_dict_redflag['AUCperVolCombPriorL2'] = AUC
    eval_dict_redflag['AUPRCperVolCombPriorL2'] = AUPRC
    
    eval_dict_redflag['FPR_PriorL2'] = list(_fpr)
    eval_dict_redflag['TPR_PriorL2'] = list(_tpr)
    eval_dict_redflag['Prec_PRCurve_PriorL2'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_PriorL2'] =   list(_recalls)

    eval_dict_redflag['disease_to_patient_id'] = Set['patient_disease_id']
    eval_dict_redflag['labels'] = Set['labelPerVol']
    eval_dict_redflag['disease_labels'] = Set['diseaseLabelPerVol']
    eval_dict_redflag['smax'] = [x for x in list(Set['smax'])]
    eval_dict_redflag['crop_size'] = [x for x in list(Set['crop_size'])]

    # Threshold-based metrics
    if thresh is not None:
        for t in threshholds_healthy:

            if 'comb' in t:
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVol']) > threshholds_healthy[t] ,pos_label=1, average='binary')
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVol']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreCombiPerVol'] > threshholds_healthy[t]
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVol']) > threshholds_healthy[t])
            elif 'reg' in t:
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRegPerVol']) > threshholds_healthy[t] ,pos_label=1, average='binary')
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRegPerVol']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreRegPerVol'] > threshholds_healthy[t]
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRegPerVol']) > threshholds_healthy[t])
            elif 'reco' in t: 
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVol']) > threshholds_healthy[t] ,pos_label=1, average='binary')

                
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVol']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreRecoPerVol'] > threshholds_healthy[t]

                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVol']) > threshholds_healthy[t])
            
            elif 'L2_cmb' in t:
                print(t,"1")
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVolL2']) > threshholds_healthy[t] ,pos_label=1, average='binary')
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVolL2']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreCombiPerVolL2'] > threshholds_healthy[t]
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreCombiPerVolL2']) > threshholds_healthy[t])


            elif 'L2_rcns' in t:
                print(t,"2")
                eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVolL2']) > threshholds_healthy[t] ,pos_label=1, average='binary')
                tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVolL2']) > threshholds_healthy[t],labels=[0,1]).ravel()
                eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
                eval_dict_redflag['Prediction_'+t] = Set['AnomalyScoreRecoPerVolL2'] > threshholds_healthy[t]
                eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScoreRecoPerVolL2']) > threshholds_healthy[t])
           


            

          
    return eval_dict_redflag


def evaluateModelMultiClass(Set,thresh):

        
        eval_dict_redflag = {}
        if thresh is not None:
            threshholds_healthy = thresh
            eval_dict_redflag['redflag_thresholds'] = threshholds_healthy
    ##


def evaluateModel(Set,thresh):

    
    eval_dict_redflag = {}
    if thresh is not None:
        threshholds_healthy = thresh
        eval_dict_redflag['redflag_thresholds'] = threshholds_healthy
##
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScorePerVol']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScorePerVol']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUC'] = AUC
    eval_dict_redflag['AUPRC'] = AUPRC

    eval_dict_redflag['FPR'] = list(_fpr)
    eval_dict_redflag['TPR'] = list(_tpr)

    eval_dict_redflag['Prec_PRCurve'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve'] =   list(_recalls)
    
    #print(np.array(Set['AnomalyScorePerVol_one_instance']).shape,np.array(Set['labelPerVol']).shape)
    AUC, _fpr, _tpr, _threshs = compute_roc(np.array(Set['AnomalyScorePerVol_one_instance']),np.array(Set['labelPerVol']))
    AUPRC, _precisions, _recalls, _threshs = compute_prc(np.array(Set['AnomalyScorePerVol_one_instance']),np.array(Set['labelPerVol']))
    eval_dict_redflag['AUC_one_instance'] = AUC
    eval_dict_redflag['AUPRC_one_instance'] = AUPRC

    eval_dict_redflag['FPR_one_instance'] = list(_fpr)
    eval_dict_redflag['TPR_one_instance'] = list(_tpr)

    eval_dict_redflag['Prec_PRCurve_one_instance'] =  list(_precisions)
    eval_dict_redflag['Rec_PRCurve_one_instance'] =   list(_recalls)
    
    
    eval_dict_redflag['disease_to_patient_id'] = Set['patient_disease_id']
    eval_dict_redflag['labels'] = [x for x in list(Set['labelPerVol'])]
    eval_dict_redflag['smax'] = [x for x in list(Set['smax'])]
    eval_dict_redflag['disease_labels'] = [x for x in list(Set['diseaseLabelPerVol'])]
    
    eval_dict_redflag['confidence_one_instance'] = [x for x in list(Set['AnomalyScorePerVol_one_instance'])]
    
    eval_dict_redflag['confidence'] = [x for x in list(Set['AnomalyScorePerVol'])]
    eval_dict_redflag['confidence_std'] = [x for x in list(Set['AnomalyScorePerVol_std'])]
    eval_dict_redflag['crop_size'] = [x for x in list(Set['crop_size'])]
    
    # Threshold-based metrics
    if thresh is not None:
        for t in threshholds_healthy:
            eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol']) > threshholds_healthy[t] ,pos_label=1, average='binary')
            tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol']) > threshholds_healthy[t],labels=[0,1]).ravel()
            eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
            eval_dict_redflag['Prediction_'+t] = Set['AnomalyScorePerVol'] > threshholds_healthy[t]
            eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol']) > threshholds_healthy[t])
            eval_dict_redflag[t] =  threshholds_healthy[t]


           
            eval_dict_redflag['Precision_'+t+"_one_instance"], eval_dict_redflag['Recall_'+t+"_one_instance"], eval_dict_redflag['F1_'+t+"_one_instance"], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol_one_instance']) > threshholds_healthy[t] ,pos_label=1, average='binary')
            tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol_one_instance']) > threshholds_healthy[t],labels=[0,1]).ravel()
            eval_dict_redflag['Specificity_'+t+"_one_instance"] = tn / (tn+fp)
            eval_dict_redflag['Prediction_'+t+"_one_instance"] = Set['AnomalyScorePerVol_one_instance'] > threshholds_healthy[t]
            eval_dict_redflag['Accuracy_'+t+"_one_instance"] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol_one_instance']) > threshholds_healthy[t])
            eval_dict_redflag[t] =  threshholds_healthy[t]
            
        
    else: 

        t = "50percent"
        

        threshold = np.empty_like(np.array(Set['AnomalyScorePerVol']))
        threshold[:] = 0.5
        eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol']) > threshold ,pos_label=1, average='binary')
        tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol']) > threshold,labels=[0,1]).ravel()
        eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
        eval_dict_redflag['Prediction_'+t] = Set['AnomalyScorePerVol'] > threshold
        eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol']) > threshold)
        eval_dict_redflag[t] =  threshold

        
        t = "50percent_one_instance"
        eval_dict_redflag['Precision_'+t], eval_dict_redflag['Recall_'+t], eval_dict_redflag['F1_'+t], _  = precision_recall_fscore_support(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol_one_instance']) > threshold ,pos_label=1, average='binary')
        tn, fp, fn, tp = confusion_matrix(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol_one_instance']) > threshold,labels=[0,1]).ravel()
        eval_dict_redflag['Specificity_'+t] = tn / (tn+fp)
        eval_dict_redflag['Prediction_'+t] = Set['AnomalyScorePerVol_one_instance'] > threshold
        eval_dict_redflag['Accuracy_'+t] = accuracy_score(np.array(Set['labelPerVol']),np.array(Set['AnomalyScorePerVol_one_instance']) > threshold)
        eval_dict_redflag[t] =  threshold
        

    return eval_dict_redflag



def apply_brainmask(x, brainmask, erode , iterations):
    
    strel = scipy.ndimage.generate_binary_structure(2, 1)
    brainmask = np.expand_dims(brainmask, 2)
    if erode:
        brainmask = scipy.ndimage.morphology.binary_erosion(np.squeeze(brainmask), structure=strel, iterations=iterations)
    return np.multiply(np.squeeze(brainmask), np.squeeze(x))

def apply_brainmask_volume(vol,mask_vol,erode=True, iterations=10) : 
    for s in range(vol.squeeze().shape[2]): 
        slice = vol.squeeze()[:,:,s]
        mask_slice = mask_vol.squeeze()[:,:,s]
        eroded_vol_slice = apply_brainmask(slice, mask_slice, erode = True, iterations=vol.squeeze().shape[1]//25)
        vol.squeeze()[:,:,s] = eroded_vol_slice
    return vol

def apply_3d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume
def apply_2d_median_filter(volume, kernelsize=5):  # kernelsize 5 works quite well
    img = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize))
    return img
    
def squash_intensities(img):
    # logistic function intended to squash reconstruction errors from [0;0.2] to [0;1] (just an example)
    k = 100
    offset = 0.5
    return 2.0 * ((1.0 / (1.0 + np.exp(-k * img))) - offset)


def apply_colormap_3D(self,volume):
    
    axis_0_images = np.zeros((1,self.cfg["dimension"],self.cfg["dimension"],3))
    axis_1_images = np.zeros((1,self.cfg["dimension"],self.cfg["dimension"],3))
    axis_2_images = np.zeros((1,self.cfg["dimension"],self.cfg["dimension"],3))
    for i in range(volume.shape[0]):
        heatmap = cv.applyColorMap(volume[i,:,:].astype(np.uint8), cv.COLORMAP_JET)
        
        axis_0_images = np.concatenate((axis_0_images, np.expand_dims(heatmap,axis=0)), axis=0)

    for i in range(volume.shape[1]):
        heatmap = cv.applyColorMap(volume[:,i,:].astype(np.uint8), cv.COLORMAP_JET)
        axis_1_images = np.concatenate((axis_1_images, np.expand_dims(heatmap,axis=0)), axis=0)

    for i in range(volume.shape[2]):
        heatmap = cv.applyColorMap(volume[:,:,i].astype(np.uint8), cv.COLORMAP_JET)
        axis_2_images = np.concatenate((axis_2_images, np.expand_dims(heatmap,axis=0)), axis=0)

    
    return axis_0_images[1:,:,:,:],axis_1_images[1:,:,:,:],axis_2_images[1:,:,:,:]

def apply_colormap(img, colormap_handle):
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = Image.fromarray(np.uint8(colormap_handle(img) * 255))
    return img

def add_colorbar(img):
    for i in range(img.squeeze().shape[0]):
        img[i, -1] = float(i) / img.squeeze().shape[0]

    return img

def filter_3d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=3)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume



# From Zimmerer iterative algorithm for threshold search
def find_best_val(x, y, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):  #x: Image , y: Label
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5

    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75
    val_bottom = dice(x > q_bottom, y)
    #print(str(np.mean(x>q_bottom)) + str(np.mean(y)))
    val_top = dice(x > q_top, y)
    #print(str(np.mean(x>q_top)) + str(np.mean(y)))
    #val_bottom = val_fn(x, y, q_bottom) # val_fn is the dice calculation dice(p, g)
    #val_top = val_fn(x, y, q_top)

    if val_bottom >= val_top:
        if val_bottom >= max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top >= max_val:
            max_val = val_top
            max_point = q_top
        return find_best_val(x, y, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val,max_point=max_point)
def dice(P, G):
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score

    
def compute_roc(predictions, labels):
    _fpr, _tpr, _ = roc_curve(labels.astype(int), predictions,pos_label=1)
    roc_auc = auc(_fpr, _tpr)
    return roc_auc, _fpr, _tpr, _


def compute_prc(predictions, labels):

    precisions, recalls, thresholds = precision_recall_curve(labels.astype(int), predictions,pos_label=1)
    auprc = average_precision_score(labels.astype(int), predictions)
    return auprc, precisions, recalls, thresholds   

# Dice Score 
def xfrange(start, stop, step):
    i = 0
    while start + i * step < stop:
        yield start + i * step
        i += 1

# def confusion_matrix(P, G):
#     tp = np.sum(np.multiply(P.flatten(), G.flatten()))
#     fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
#     fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
#     tn = np.sum(np.multiply(np.invert(P.flatten()), np.invert(G.flatten())))
#     return tp, fp, tn, fn

def tpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fn = np.sum(np.multiply(np.invert(P.flatten()), G.flatten()))
    return tp / (tp + fn)


def fpr(P, G):
    tp = np.sum(np.multiply(P.flatten(), G.flatten()))
    fp = np.sum(np.multiply(P.flatten(), np.invert(G.flatten())))
    return fp / (fp + tp)


def normalize(tensor): # THanks DZimmerer
    tens_deta = tensor.detach().cpu()
    tens_deta -= float(np.min(tens_deta.numpy()))
    tens_deta /= float(np.max(tens_deta.numpy()))

    return tens_deta