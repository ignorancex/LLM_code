from train import *

def run_Unibrain():

    train_set_name="adhd_train_sample.npy"
    val_set_name="adhd_val_sample.npy"
    test_set_name="adhd_test_sample.npy"
    enc_nf=[1,16,32,32,64,64,128,128]
    dec_nf=[128,64,64,32,32,32,16,1]
    enc_affine=[2,16,32,64,128,256,512]
    enc_seg=[1,128,256,256,512,512]
    dec_seg=[512,256,256,256,128,4]
    enc_par=[1,128,256,256,512,512]
    dec_par=[512,256,256,256,128,117]
    feature_net_hidden=256
    feature_dim=128
    gnn_hidden=128
    reg_loss_name="GCC"
    
    ext_stage=1
    reg_stage=5
   
    weight_decay=0.000001
    
    if_train_aug=True
    if_pred_aal=False
    round_num=1
    
    lambda_ext=1            
    lambda_reg=0.1                 
    lambda_seg=1
    lambda_cls=1
   
    batch_size=1
    img_size=96
    num_epochs=1000

    learning_rate=0.00001   
    save_every_epoch=1
    save_start_epoch=0
    
    model_name=UniBrain(img_size,ext_stage , reg_stage,if_pred_aal,
                        enc_nf,dec_nf,enc_affine,enc_seg,dec_seg,enc_par,
                        dec_par,feature_net_hidden,feature_dim,gnn_hidden)
    
    train_UniBrain(img_size,
          reg_stage,
          ext_stage,
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
          round_num,lambda_ext,lambda_reg,lambda_seg,lambda_cls)
    return



if __name__ == "__main__":
    run_Unibrain()
    
    
    