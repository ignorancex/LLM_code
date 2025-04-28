import torch
import numpy as np
import statistics

class Compute_Clusters():
    def get_source_centers(self, G, C1, C2, train_data_batch_s, train_label_batch_s, num_classes):
        #Compute source clusters
        count_c_c=0
        cntr_gamma=0.5

        for i in range(int(train_data_batch_s.shape[0]/3)):
          train_data_batch_tmp_s=train_data_batch_s[i]
          train_tensor_s = torch.from_numpy(train_data_batch_tmp_s)
          train_tensor_s=train_tensor_s.to('cuda')
          train_tensor_s=train_tensor_s.float()
          train_label_batch_tmp_s=train_label_batch_s[i]
          train_label_tmp_tensor_s = torch.from_numpy(train_label_batch_tmp_s)
          train_label_tmp_one_hot_s=torch.nn.functional.one_hot(train_label_tmp_tensor_s, num_classes)
          train_label_tmp_tensor_s=train_label_tmp_one_hot_s.to('cuda')
          train_label_tmp_tensor_s = train_label_tmp_tensor_s.to(torch.float)

          train_tensor_s=torch.swapaxes(train_tensor_s, 1, 2)
          feat_s = G(train_tensor_s)
          output_C1_s, output_C1_prev_s = C1(feat_s)
          output_C2_s, output_C2_prev_s = C2(feat_s)
          label_batch_s=train_label_tmp_tensor_s.data.max(1)[1]
          idx_0_s=torch.where(label_batch_s==0)
          feat_0_s=feat_s[idx_0_s]

          if feat_0_s.shape[0]==0:
            center_0_curr_s=torch.zeros(feat_0_s.shape[1])
          else:
            center_0_curr_s=torch.mean(feat_0_s, 0)

          center_0_curr_s=center_0_curr_s.to('cuda')

          idx_1_s=torch.where(label_batch_s==1)
          feat_1_s=feat_s[idx_1_s]
          if feat_1_s.shape[0]==0:
            center_1_curr_s=torch.zeros(feat_1_s.shape[1])
          else:
            center_1_curr_s=torch.mean(feat_1_s, 0)
          center_1_curr_s=center_1_curr_s.to('cuda')

          if count_c_c==0:
            center_0_s=center_0_curr_s
            center_0_s=center_0_s.to('cuda')
            center_1_s=center_1_curr_s
            center_1_s=center_1_s.to('cuda')
          else:
            if feat_0_s.shape[0]==0:
              center_0_s=center_0_s
            else:
              center_0_s=(1-cntr_gamma) * center_0_s + cntr_gamma * center_0_curr_s
            if feat_1_s.shape[0]==0:
              center_1_s=center_1_s
            else:
              center_1_s=(1-cntr_gamma) * center_1_s + cntr_gamma * center_1_curr_s

          count_c_c=count_c_c+1

        center_0_s=center_0_s.detach().cpu().data.numpy()
        center_1_s=center_1_s.detach().cpu().data.numpy()
        all_centers_s=np.empty((num_classes,center_0_s.shape[0]))
        all_centers_s[0]=center_0_s
        all_centers_s[1]=center_1_s
        return all_centers_s


    def get_mean_dist(self, G, C1, C2, train_data_batch_s, train_label_batch_s, all_centers_s, num_classes):
        #Compute intra-cluster distance (mean and std)
        #Compute classifier discrepancy (mean and std)
        count_c_c=0
        cntr_gamma=0.5
        all_centers_s = torch.from_numpy(all_centers_s)
        all_centers_s=all_centers_s.to('cuda')
        all_centers_s = all_centers_s.to(torch.float)
        dist_0_all=[]
        dist_1_all=[]
        dist_classifiers_all=[]

        for i in range(int(train_data_batch_s.shape[0]/3)):

            train_data_batch_tmp_s=train_data_batch_s[i]
            train_tensor_s = torch.from_numpy(train_data_batch_tmp_s)
            train_tensor_s=train_tensor_s.to('cuda')
            train_tensor_s=train_tensor_s.float()
            train_label_batch_tmp_s=train_label_batch_s[i]
            train_label_tmp_tensor_s = torch.from_numpy(train_label_batch_tmp_s)
            train_label_tmp_one_hot_s=torch.nn.functional.one_hot(train_label_tmp_tensor_s, num_classes)
            train_label_tmp_tensor_s=train_label_tmp_one_hot_s.to('cuda')
            train_label_tmp_tensor_s = train_label_tmp_tensor_s.to(torch.float)

            train_tensor_s=torch.swapaxes(train_tensor_s, 1, 2)
            feat_s = G(train_tensor_s)
            output_C1_s, output_C1_prev_s = C1(feat_s)
            output_C2_s, output_C2_prev_s = C2(feat_s)
            output_C1_C2= (output_C1_s+output_C2_s)/2
            label_batch_s=train_label_tmp_tensor_s.data.max(1)[1]
            pred_batch_s=output_C1_C2.data.max(1)[1]
            idx_correct_s=torch.where(label_batch_s==pred_batch_s)

            correct_output_C1_prev_s=output_C1_prev_s[idx_correct_s]
            correct_output_C2_prev_s=output_C2_prev_s[idx_correct_s]
            correct_pred_batch_s=pred_batch_s[idx_correct_s]
            correct_feat_s=feat_s[idx_correct_s]
            sum_dist_classifiers=0
            count_correct_classifier=0
            sum_dist_feat_0=0
            sum_dist_feat_1=0
            count_correct_feat_0=0
            count_correct_feat_1=0
            print(correct_output_C1_prev_s.shape)

            for ii in range(correct_output_C1_prev_s.shape[0]):
                sum_dist_classifiers=sum_dist_classifiers+ self.L2Distance(correct_output_C1_prev_s[ii], correct_output_C2_prev_s[ii])
                dist_classifiers_all.append(self.L2Distance(correct_output_C1_prev_s[ii], correct_output_C2_prev_s[ii]).item())
                count_correct_classifier=count_correct_classifier+1

                if correct_pred_batch_s[ii]==0:
                  sum_dist_feat_0=sum_dist_feat_0 + self.L2Distance(all_centers_s[0], correct_feat_s[ii])
                  dist_0_all.append(self.L2Distance(all_centers_s[0], correct_feat_s[ii]).item())
                  count_correct_feat_0=count_correct_feat_0+1
                if correct_pred_batch_s[ii]==1:
                  sum_dist_feat_1=sum_dist_feat_1 + self.L2Distance(all_centers_s[1], correct_feat_s[ii])
                  dist_1_all.append(self.L2Distance(all_centers_s[1], correct_feat_s[ii]).item())
                  count_correct_feat_1=count_correct_feat_1+1

            mean_dist_classifier_curr_s= sum_dist_classifiers / count_correct_classifier
            mean_dist_feat_0_curr_s= sum_dist_feat_0 / count_correct_feat_0
            mean_dist_feat_1_curr_s= sum_dist_feat_1 / count_correct_feat_1

            if count_c_c==0:
              mean_dist_classifier_s=mean_dist_classifier_curr_s
              mean_dist_classifier_s=mean_dist_classifier_s.to('cuda')
              mean_dist_feat_0_s=mean_dist_feat_0_curr_s
              mean_dist_feat_0_s=mean_dist_feat_0_s.to('cuda')
              mean_dist_feat_1_s=mean_dist_feat_1_curr_s
              mean_dist_feat_1_s=mean_dist_feat_1_s.to('cuda')
            else:
                mean_dist_classifier_s=(1-cntr_gamma) * mean_dist_classifier_s + cntr_gamma * mean_dist_classifier_curr_s
                mean_dist_feat_0_s=(1-cntr_gamma) * mean_dist_feat_0_s + cntr_gamma * mean_dist_feat_0_curr_s
                mean_dist_feat_1_s=(1-cntr_gamma) * mean_dist_feat_1_s + cntr_gamma * mean_dist_feat_1_curr_s

            count_c_c=count_c_c+1

        mean_dist_feat_s = (mean_dist_feat_0_s + mean_dist_feat_1_s )/2
        mean_dist_classifier_s=mean_dist_classifier_s.detach().cpu().data.numpy()
        mean_dist_feat_s=mean_dist_feat_s.detach().cpu().data.numpy()
        mean_dist_feat_0_s=mean_dist_feat_0_s.detach().cpu().data.numpy()
        mean_dist_feat_1_s=mean_dist_feat_1_s.detach().cpu().data.numpy()
        all_mean_dist_feat_s=np.empty((num_classes))
        all_mean_dist_feat_s[0]=mean_dist_feat_0_s
        all_mean_dist_feat_s[1]=mean_dist_feat_1_s
        std_0 = statistics.pstdev(dist_0_all)
        std_1 = statistics.pstdev(dist_1_all)
        std_dist_classifier_s = statistics.pstdev(dist_classifiers_all)
        all_std_dist_feat_s=np.empty((num_classes))
        all_std_dist_feat_s[0]=std_0
        all_std_dist_feat_s[1]=std_1

        return mean_dist_classifier_s, std_dist_classifier_s, mean_dist_feat_s, all_mean_dist_feat_s, all_std_dist_feat_s

    def get_target_centers(self, G, C1, C2, train_data_batch_t, all_centers_s, mean_dist_classifier_s, mean_dist_feat_s, all_mean_dist_feat_s, num_classes):
        #Compute target clusters using confident predictions
        cntr_gamma=0.5
        count_c_c=0
        all_centers_s = torch.from_numpy(all_centers_s)
        all_centers_s=all_centers_s.to('cuda')
        all_centers_s = all_centers_s.to(torch.float)
        mean_dist_classifier_s=torch.from_numpy(mean_dist_classifier_s)
        mean_dist_classifier_s=mean_dist_classifier_s.to('cuda')
        mean_dist_classifier_s = mean_dist_classifier_s.to(torch.float)
        mean_dist_classifier_s=mean_dist_classifier_s
        mean_dist_classifier_s=mean_dist_classifier_s+1

        mean_dist_feat_s = torch.from_numpy(mean_dist_feat_s)
        mean_dist_feat_s=mean_dist_feat_s.to('cuda')
        mean_dist_feat_s = mean_dist_feat_s.to(torch.float)
        all_mean_dist_feat_s = torch.from_numpy(all_mean_dist_feat_s)
        all_mean_dist_feat_s=all_mean_dist_feat_s.to('cuda')
        all_mean_dist_feat_s = all_mean_dist_feat_s.to(torch.float)
        all_mean_dist_feat_s[0]=all_mean_dist_feat_s[0]+1
        all_mean_dist_feat_s[1]=all_mean_dist_feat_s[1]+1

        for i in range(int(train_data_batch_t.shape[0]/3)):
            train_data_batch_tmp_t=train_data_batch_t[i]
            train_tensor_t = torch.from_numpy(train_data_batch_tmp_t)
            train_tensor_t=train_tensor_t.to('cuda')
            train_tensor_t=train_tensor_t.float()
            train_tensor_t=torch.swapaxes(train_tensor_t, 1, 2)

            #Get outputs of Feature extractor and Classifier
            feat_t = G(train_tensor_t)
            output_C1_t, output_C1_prev_t = C1(feat_t)
            output_C2_t, output_C2_prev_t = C2(feat_t)
            output_C1_C2_t = (output_C1_t + output_C2_t)/2

            max_pred_val_t,max_pred_idx_t=torch.max(output_C1_C2_t,1)
            confident_pred_idx_t=torch.where((max_pred_val_t>0.99))       #Pick initial confident predictions with softmax value>0.99
            confident_pred_class_t=max_pred_idx_t[confident_pred_idx_t]
            confident_pred_output_t=max_pred_val_t[confident_pred_idx_t]
            confident_feat_t=feat_t[confident_pred_idx_t]
            confident_C1_t=output_C1_prev_t[confident_pred_idx_t]
            confident_C2_t=output_C2_prev_t[confident_pred_idx_t]
            jj=0

            for ii in range(confident_feat_t.shape[0]):
              if ((self.L2Distance(confident_C1_t[ii], confident_C2_t[ii])<mean_dist_classifier_s)): #Check classifier discrepancy with previously calculated mean classifier discrepancy
                confident_pred_class_t[jj]=confident_pred_class_t[ii]
                confident_pred_output_t[jj]=confident_pred_output_t[ii]
                confident_feat_t[jj]=confident_feat_t[ii]
                confident_C1_t[jj]=confident_C1_t[ii]
                confident_C2_t[jj]=confident_C2_t[ii]
                jj=jj+1
            confident_pred_class_t=confident_pred_class_t[0:jj]
            confident_pred_output_t=confident_pred_output_t[0:jj]
            confident_feat_t=confident_feat_t[0:jj]
            confident_C1_t=confident_C1_t[0:jj]
            confident_C2_t=confident_C2_t[0:jj]

            idx_0_t=torch.where(confident_pred_class_t==0)
            feat_0_t=confident_feat_t[idx_0_t]
            conf_pred_0=confident_pred_class_t[idx_0_t]
            conf_pred_frac_0=confident_pred_output_t[idx_0_t]
            conf_C1_0=confident_C1_t[idx_0_t]
            conf_C2_0=confident_C2_t[idx_0_t]
            jj=0

            for ii in range(feat_0_t.shape[0]):
              if ((self.L2Distance(all_centers_s[0], feat_0_t[ii]) < all_mean_dist_feat_s[0])): #Check feature distance with previously calculated mean intra-cluster distance
                feat_0_t[jj]=feat_0_t[ii]
                conf_pred_0[jj]=conf_pred_0[ii]
                conf_pred_frac_0[jj]=conf_pred_frac_0[ii]
                conf_C1_0[jj]=conf_C1_0[ii]
                conf_C2_0[jj]=conf_C2_0[ii]
                jj=jj+1

            feat_0_t=feat_0_t[0:jj]

            conf_pred_0=conf_pred_0[0:jj]
            conf_pred_frac_0=conf_pred_frac_0[0:jj]
            conf_C1_0=conf_C1_0[0:jj]
            conf_C2_0=conf_C2_0[0:jj]

            if feat_0_t.shape[0]==0:
              center_0_curr_t=torch.zeros(feat_0_t.shape[1])
            else:
              center_0_curr_t=torch.mean(feat_0_t, 0)

            center_0_curr_t=center_0_curr_t.to('cuda')

            idx_1_t=torch.where(confident_pred_class_t==1)
            feat_1_t=confident_feat_t[idx_1_t]
            conf_pred_1=confident_pred_class_t[idx_1_t]
            conf_pred_frac_1=confident_pred_output_t[idx_1_t]
            conf_C1_1=confident_C1_t[idx_1_t]
            conf_C2_1=confident_C2_t[idx_1_t]
            jj=0

            for ii in range(feat_1_t.shape[0]):
              if ((self.L2Distance(all_centers_s[1], feat_1_t[ii]) < all_mean_dist_feat_s[1])):
                feat_1_t[jj]=feat_1_t[ii]
                conf_pred_1[jj]=conf_pred_1[ii]
                conf_pred_frac_1[jj]=conf_pred_frac_1[ii]
                conf_C1_1[jj]=conf_C1_1[ii]
                conf_C2_1[jj]=conf_C2_1[ii]
                jj=jj+1

            feat_1_t=feat_1_t[0:jj]
            conf_pred_1=conf_pred_1[0:jj]
            conf_pred_frac_1=conf_pred_frac_1[0:jj]
            conf_C1_1=conf_C1_1[0:jj]
            conf_C2_1=conf_C2_1[0:jj]

            if feat_1_t.shape[0]==0:
              center_1_curr_t=torch.zeros(feat_1_t.shape[1])
            else:
              center_1_curr_t=torch.mean(feat_1_t, 0)
            center_1_curr_t=center_1_curr_t.to('cuda')

            if count_c_c==0:
              center_0_t=center_0_curr_t
              center_0_t=center_0_t.to('cuda')
              center_1_t=center_1_curr_t
              center_1_t=center_1_t.to('cuda')
            else:
              if feat_0_t.shape[0]==0:
                center_0_t=center_0_t
              else:
                center_0_t=(1-cntr_gamma) * center_0_t + cntr_gamma * center_0_curr_t
              if feat_1_t.shape[0]==0:
                center_1_t=center_1_t
              else:
                center_1_t=(1-cntr_gamma) * center_1_t + cntr_gamma * center_1_curr_t

            count_c_c=count_c_c+1

        center_0_t=center_0_t.detach().cpu().data.numpy()
        center_1_t=center_1_t.detach().cpu().data.numpy()
        all_centers_t=np.empty((num_classes,center_0_t.shape[0]))
        all_count_t=np.empty((num_classes))
        all_centers_t[0]=center_0_t
        all_centers_t[1]=center_1_t
        return all_centers_t

    def L2Distance(self, x, y, dim=0, if_mean=True):
        if if_mean:
            distance = torch.mean(torch.sqrt(torch.sum((x - y) ** 2, dim=dim)))
        else:
            distance = torch.sqrt(torch.sum((x - y) ** 2, dim=dim))
        return distance
