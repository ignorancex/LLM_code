from cdbs.funcs import *
from cdbs.network import FlexParaGlobal,FlexParaMultiChart

def train_flexpara(vertices, normals, faces, charts_number, num_iter, export_folder,N):
    if charts_number==1:
        net = FlexParaGlobal().train().cuda()
        max_lr, min_lr = 1e-3, 1e-5
        opt_diff_itv = 5
        optimizer = optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-8,eps=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=min_lr)
        L1Loss = nn.L1Loss()

        pre_samplings = np.concatenate((vertices,normals), axis=-1)
        pre_samplings_triangles = faces

        pre_samplings = torch.tensor(pre_samplings).unsqueeze(0).float().cuda()
        num_faces_pres = pre_samplings_triangles.shape[0]
        pre_samplings_triangles = torch.tensor(pre_samplings_triangles).unsqueeze(0).long().cuda()

        num_col_samp = 200
        collected_samplings = []
        collected_samplings_triangles = []
        for i in tqdm(range(num_col_samp)):
            sampling_base = pre_samplings_triangles[:, np.random.choice(num_faces_pres, N, replace=False), :]
            idx_combine,triangles_combine = combine(sampling_base)
            collected_samplings_triangles.append(triangles_combine)
            collected_samplings.append(index_points(pre_samplings, idx_combine))

        for epc_idx in tqdm(range(1, num_iter+1)):
            net.zero_grad()
            np.random.seed()
            select_sample = np.random.choice(num_col_samp)
            input_pc = collected_samplings[select_sample]# [1, ... , 6]
            input_triangles = collected_samplings_triangles[select_sample]# [1,N,3]

            N = input_pc.shape[1] # number of input 3D points at each training iteration
            grid_height, grid_width = int(np.sqrt(N)), int(np.sqrt(N))
            G = torch.tensor(build_2d_grids(grid_height, grid_width).reshape(-1, 2)).unsqueeze(0).float().cuda() # [1, M, 2]
            M = G.size(1) # number of grid 2D points

            
            P = input_pc[:, :, 0:3] # [1, N, 3], point coordinates at this training iteration
            P_gtn = input_pc[:, :, 3:6] # [1, N, 3], ground-truth point normals at this training iteration
            
            P_opened, Q, P_cycle, P_cycle_n, Q_hat, P_hat, P_hat_n, P_hat_opened, Q_hat_cycle = net(G, P)
            Q_normalized = uv_bounding_box_normalization(Q)
            Q_hat_normalized = uv_bounding_box_normalization(Q_hat)
            Q_hat_cycle_normalized = uv_bounding_box_normalization(Q_hat_cycle)


            #### wrapping loss
            L_wrap = chamfer_distance_cuda(P_hat, P)

            #### unwrapping loss
            rep_th = (2 / (np.ceil(np.sqrt(M)) - 1)) * 0.25
            L_unwrap = compute_repulsion_loss(Q_normalized, 8, rep_th) + compute_repulsion_loss(Q_hat_normalized, 8, rep_th) + compute_repulsion_loss(Q_hat_cycle_normalized, 8, rep_th)

            #### cycle consistency on points
            L_cc_p = L1Loss(P, P_cycle) + L1Loss(Q_hat, Q_hat_cycle)
            
            #### cycle consistency on normals
            L_cc_n = compute_normal_cos_sim_loss(P_gtn, P_cycle_n)

            if epc_idx==1 or np.mod(epc_idx, opt_diff_itv) == 0:
                _, e1, e2 = compute_differential_properties(P_cycle, Q)
                L_conf_diff = L1Loss(e1, e2)
                L_conf_tri = angle_preserving_loss(Q_normalized,P,input_triangles)
                loss = L_wrap + L_unwrap*0.01 + L_cc_p*0.01 + L_cc_n*0.005 + L_conf_diff*0.01 + L_conf_tri*0.00001
            else:
                loss = L_wrap + L_unwrap*0.01 + L_cc_p*0.01 + L_cc_n*0.005

            loss.backward()
            optimizer.step()
            scheduler.step()
        net.zero_grad()
        torch.cuda.empty_cache()
        torch.save(net.state_dict(), os.path.join(export_folder, "flexpara_global.pth"))

    elif charts_number>1:
        net = FlexParaMultiChart(charts_number).train().cuda()
        max_lr, min_lr = 1e-3, 1e-5
        optimizer = optim.AdamW(net.parameters(), lr=max_lr, weight_decay=1e-8,eps=1e-5)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=min_lr)

        pre_samplings = np.concatenate((vertices,normals), axis=-1)
        pre_samplings_triangles = faces

        pre_samplings = torch.tensor(pre_samplings).unsqueeze(0).float().cuda() # [1, num_pts_pres, 6]
        num_faces_pres = pre_samplings_triangles.shape[0]
        pre_samplings_triangles = torch.tensor(pre_samplings_triangles).unsqueeze(0).long().cuda()

        num_col_samp = 200
        collected_samplings = []
        collected_samplings_triangles = []
        for i in tqdm(range(num_col_samp)):
            sampling_base = pre_samplings_triangles[:, np.random.choice(num_faces_pres, N, replace=False), :]
            idx_combine,triangles_combine = combine(sampling_base)
            collected_samplings_triangles.append(triangles_combine)
            collected_samplings.append(index_points(pre_samplings, idx_combine))

        for epc_idx in tqdm(range(1, num_iter+1)):
            net.zero_grad()
            np.random.seed()
            select_sample = np.random.choice(num_col_samp)
            input_pc = collected_samplings[select_sample]# [1, ... , 6]
            
            input_triangles = collected_samplings_triangles[select_sample]# [1,N,3]
            P = input_pc[:, :, 0:3] # [1, N, 3], point coordinates at this training iteration
            P_gtn = input_pc[:, :, 3:6] # [1, N, 3], ground-truth point normals at this training iteration
            
            #### forward pass
            results,charts_prob = net(P) 
            L_unwrap_all = 0
            L_cc_p_all = 0
            L_cc_n_all = 0
            L_iso_all = 0

            for i in range(charts_number):
                P_opened, Q, P_cycle, P_cycle_n = results[i]
                Q_normalized = uv_bounding_box_normalization(Q)

                rep_th = (2 / (np.ceil(np.sqrt(N)) - 1)) * 0.25
                L_unwrap = compute_repulsion_loss_charts(Q_normalized, 8, rep_th,charts_prob[:,:,i]) 
                L_cc_p = L1_loss_charts(P,P_cycle,charts_prob[:,:,i])
                L_cc_n = compute_normal_cos_sim_loss_charts(P_gtn, P_cycle_n,charts_prob[:,:,i])
                L_iso = isometric_loss_by_prob_l1_percent(Q,P,input_triangles,charts_prob[:,:,i])
            
                L_unwrap_all = L_unwrap_all + L_unwrap
                L_cc_p_all = L_cc_p_all +L_cc_p
                L_cc_n_all = L_cc_n_all + L_cc_n
                L_iso_all = L_iso_all + L_iso**2
                
            loss = L_unwrap_all + L_cc_p_all*10 + L_cc_n_all*10 + L_iso_all
            loss.backward()

            optimizer.step()
            scheduler.step()
                      
        net.zero_grad()
        torch.cuda.empty_cache()
        torch.save(net.state_dict(), os.path.join(export_folder, "flexpara_multi_" + str(charts_number) + ".pth"))



def test_flexpara(vertices, normals, faces, load_ckpt_path, export_folder, charts_number):
    if load_ckpt_path == "flexpara_global.pth":
        net =  FlexParaGlobal().train().cuda()
        weight = torch.load(os.path.join(export_folder, load_ckpt_path))
        net.load_state_dict(weight)
        pre_samplings = np.concatenate((vertices,normals), axis=-1)
        pre_samplings = torch.tensor(pre_samplings).unsqueeze(0).float().cuda()
        P = pre_samplings[:, :, 0:3]
        P_N = pre_samplings[:, :, 3:6]

        N = pre_samplings.shape[1] 
        grid_height, grid_width = int(np.sqrt(N)), int(np.sqrt(N))
        G = torch.tensor(build_2d_grids(grid_height, grid_width).reshape(-1, 2)).unsqueeze(0).float().cuda() 

        with torch.no_grad():
            P_opened, Q, P_cycle, P_cycle_n, Q_hat, P_hat, P_hat_n, P_hat_opened, Q_hat_cycle = net(G, P)
        Q_eval_normalization = uv_bounding_box_normalization(Q)
        plt.figure(figsize=(16, 16))
        plt.axis('off')
        plt.scatter(ts2np(Q_eval_normalization.squeeze(0))[:, 0], ts2np(Q_eval_normalization.squeeze(0))[:, 1], s=5, c=((ts2np(P_N.squeeze(0)) + 1) / 2))
        plt.savefig(os.path.join(export_folder, "Q_eval_normalized.png"), dpi=400, bbox_inches="tight")

    else:
        net =  FlexParaMultiChart(charts_number).train().cuda()
        weight = torch.load(os.path.join(export_folder, load_ckpt_path))
        net.load_state_dict(weight)

        pre_samplings = np.concatenate((vertices,normals), axis=-1)
        pre_samplings = torch.tensor(pre_samplings).unsqueeze(0).float().cuda()
        P = pre_samplings[:, :, 0:3]
        P_N = pre_samplings[:, :, 3:6]
        with torch.no_grad():
            results_eval,charts_prob_eval = net(P)

        class_labels = torch.argmax(charts_prob_eval, dim=2).squeeze(0)
        for i in range(charts_number):
            Q_eval_chart = results_eval[i][1][:,class_labels == i,:]
            if Q_eval_chart.shape[1]==0:
                continue
            Q_eval_chart_normalization = uv_bounding_box_normalization(Q_eval_chart)
            P_gtn_eval_chart = P_N[:,class_labels == i,:]
            plt.figure(figsize=(16, 16))
            plt.axis('off')
            plt.scatter(ts2np(Q_eval_chart_normalization.squeeze(0))[:, 0], ts2np(Q_eval_chart_normalization.squeeze(0))[:, 1], s=5, c=((ts2np(P_gtn_eval_chart.squeeze(0)) + 1) / 2))
            plt.savefig(os.path.join(export_folder, "Q_eval_normalized_"+str(i)+".png"), dpi=400, bbox_inches="tight")



