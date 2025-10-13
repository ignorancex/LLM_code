#include <turboreg/core_turboreg_gpu.hpp>
#include <turboreg/utils_debug.hpp>
#include <torch/torch.h>

void turboreg::verification(const torch::Tensor &cliques_tensor,
                  const torch::Tensor &kpts_src,
                  const torch::Tensor &kpts_dst,
                  float inlier_threshold,
                  torch::Tensor &best_in_num,
                  torch::Tensor &best_in_indic,
                  torch::Tensor &best_trans,
                  torch::Tensor &res,
                  torch::Tensor &cliques_wise_trans,
                  torch::Tensor &cliquewise_in_num)
{
    torch::Tensor kpts_src_sub = kpts_src.index_select(0, cliques_tensor.view({-1})).view({-1, 3, 3});
    torch::Tensor kpts_dst_sub = kpts_dst.index_select(0, cliques_tensor.view({-1})).view({-1, 3, 3});
    cliques_wise_trans = rigid_transform_3d(kpts_src_sub, kpts_dst_sub, torch::Tensor(), 0.0);
    auto cliques_wise_trans_3x3 = cliques_wise_trans.slice(1, 0, 3).slice(2, 0, 3);
    auto cliques_wise_trans_3x1 = cliques_wise_trans.slice(1, 0, 3).slice(2, 3, 4);
    auto kpts_src_prime = torch::einsum("cnm,mk->cnk", {cliques_wise_trans_3x3, kpts_src.permute({1, 0})}) + cliques_wise_trans_3x1;
    kpts_src_prime = kpts_src_prime.permute({0, 2, 1}); 

    res = torch::norm(kpts_src_prime - kpts_dst.unsqueeze(0), 2, -1); // (C, N)
    auto indic_in = res < inlier_threshold;                          

    cliquewise_in_num = indic_in.sum(/*dim=*/-1).to(torch::kFloat32); // (C,)
    auto metrics = "IN"; 
    torch::Tensor idx_best_guess;

    if (metrics == "IN")
    {
        idx_best_guess = cliquewise_in_num.argmax(); 
    }
    else if (metrics == "MAE")
    {
        res.index_add_(0, indic_in.nonzero().squeeze(1), torch::zeros_like(res)); 
        auto cliquewise_mae = res.sum(/*dim=*/-1);                                
        idx_best_guess = cliquewise_mae.argmin();                                 
    }

    best_in_num = cliquewise_in_num.index_select(0, idx_best_guess).squeeze(0); 
    best_trans = cliques_wise_trans.index_select(0, idx_best_guess).squeeze(0); 
    best_in_indic = indic_in.index_select(0, idx_best_guess).squeeze(0);        


}

void turboreg::verificationV2Metric(const torch::Tensor &cliques_tensor, const torch::Tensor &kpts_src, const torch::Tensor &kpts_dst, ModelSelection &model_selector, torch::Tensor &best_in_num, torch::Tensor &best_trans, torch::Tensor &res, torch::Tensor &cliques_wise_trans)
{
    // (C, 3, 3)
    torch::Tensor kpts_src_sub = kpts_src.index_select(0, cliques_tensor.view({-1})).view({-1, 3, 3});
    torch::Tensor kpts_dst_sub = kpts_dst.index_select(0, cliques_tensor.view({-1})).view({-1, 3, 3});

    cliques_wise_trans = rigid_transform_3d(kpts_src_sub, kpts_dst_sub, torch::Tensor(), 0.0);
    torch::Tensor idx_best_guess = model_selector.calculate_best_clique(cliques_wise_trans, kpts_src, kpts_dst);

    best_trans = cliques_wise_trans.index_select(0, idx_best_guess).squeeze(0); 
}

#include <torch/torch.h>

// Function to integrate rotation matrix R and translation vector t into a 4x4 transformation matrix
torch::Tensor integrate_trans(const torch::Tensor &R, const torch::Tensor &t)
{
    assert(R.size(1) == 3 && R.size(2) == 3); // Check that R is 3x3
    assert(t.size(1) == 3 && t.size(2) == 1); // Check that t is 3x1

    int64_t bs = R.size(0);
    torch::Tensor trans = torch::eye(4).to(R.device()).unsqueeze(0).repeat({bs, 1, 1}); // [bs, 4, 4]

    trans.index_put_({"...", torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, R);
    trans.index_put_({"...", torch::indexing::Slice(0, 3), 3}, t.view({-1, 3}));

    return trans;
}

torch::Tensor turboreg::rigid_transform_3d(const torch::Tensor &A, const torch::Tensor &B,
                                 const torch::Tensor &weights = torch::Tensor(),
                                 float weight_threshold = 0)
{
    int64_t bs = A.size(0); // Batch size
    torch::Tensor W = weights;
    if (weights.numel() == 0)
    {

        W = torch::ones_like(A.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}));
    }
    W.masked_fill_(W < weight_threshold, 0);

    // Centroid of A and B (weighted)
    torch::Tensor centroid_A = torch::sum(A * W.unsqueeze(-1), 1, /*keepdim=*/true) /
                               (torch::sum(W, 1, /*keepdim=*/true).unsqueeze(-1) + 1e-6);
    torch::Tensor centroid_B = torch::sum(B * W.unsqueeze(-1), 1, /*keepdim=*/true) /
                               (torch::sum(W, 1, /*keepdim=*/true).unsqueeze(-1) + 1e-6);

    // Subtract centroid
    torch::Tensor Am = A - centroid_A;
    torch::Tensor Bm = B - centroid_B;

    torch::Tensor H = torch::bmm(Am.permute({0, 2, 1}), Bm * W.unsqueeze(-1)); // [bs, 3, 3]

    // Singular Value Decomposition (SVD) to get the rotation
    auto svd = torch::svd(H); // Returns a tuple (U, S, Vt)
    torch::Tensor U = std::get<0>(svd);
    torch::Tensor S = std::get<1>(svd);
    torch::Tensor Vt = std::get<2>(svd);

    // Ensure rotation matrix R
    torch::Tensor delta_UV = torch::det(Vt.bmm(U.permute({0, 2, 1})));
    torch::Tensor eye = torch::eye(3, A.options()).unsqueeze(0).repeat({bs, 1, 1}); // Corrected repeat dimensions
    eye.index_put_({"...", 2, 2}, delta_UV);
    torch::Tensor R = Vt.bmm(eye).bmm(U.permute({0, 2, 1}));

    // Compute the translation vector
    torch::Tensor t = centroid_B.permute({0, 2, 1}) - R.bmm(centroid_A.permute({0, 2, 1}));

    // Return the transformation matrix (4x4)
    return integrate_trans(R, t); // Return the 4x4 transformation matrix
}

torch::Tensor transform(const torch::Tensor &src_keypts, const torch::Tensor &initial_trans)
{
    // (R, t)
    torch::Tensor R = initial_trans.narrow(0, 0, 3).narrow(1, 0, 3); // 3x3
    torch::Tensor t = initial_trans.narrow(0, 0, 3).narrow(1, 3, 1); // 3x1

    // R * pts + t
    return torch::matmul(src_keypts, R.t()) + t.t();
}

torch::Tensor turboreg::post_refinement(
    torch::Tensor initial_trans,
    torch::Tensor src_keypts,
    torch::Tensor tgt_keypts,
    int it_num,
    float inlier_threshold = 0.1)
{
    // Initialize the inlier threshold list
    torch::Tensor inlier_threshold_list = torch::full({it_num}, inlier_threshold, initial_trans.options());

    int previous_inlier_num = 0;
    torch::Tensor pred_inlier;
    for (int i = 0; i < it_num; ++i)
    {
        // Apply the initial transformation to the source keypoints
        torch::Tensor warped_src_keypts = transform(src_keypts, initial_trans);

        // Calculate L2 distance between transformed source keypoints and target keypoints
        torch::Tensor L2_dis = torch::norm(warped_src_keypts - tgt_keypts, 2, 1);

        // Predicted inliers based on distance threshold
        // [N]
        pred_inlier = L2_dis < inlier_threshold_list[i];

        // Count inliers
        int inlier_num = pred_inlier.sum().item<int>();

        // If inlier count does not change, break out of the loop
        if (std::abs(inlier_num - previous_inlier_num) < 1)
        {
            break;
        }
        else
        {
            previous_inlier_num = inlier_num;
        }

        auto weight = (1 / (1 + (L2_dis.index({pred_inlier}) / inlier_threshold_list[i]).pow(2)));
        initial_trans = rigid_transform_3d(
                            src_keypts.index({pred_inlier}).unsqueeze(0),
                            tgt_keypts.index({pred_inlier}).unsqueeze(0),
                            weight.unsqueeze(0))
                            .squeeze(0);
    }

    return initial_trans;
}
