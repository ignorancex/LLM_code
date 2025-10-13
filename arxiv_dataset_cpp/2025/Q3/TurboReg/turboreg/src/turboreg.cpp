#include <turboreg/turboreg.hpp>
#include <turboreg/utils_debug.hpp>
#include <turboreg/core_turboreg_gpu.hpp>

namespace py = pybind11;
using namespace turboreg;

TurboRegGPU::TurboRegGPU(int max_N, float tau_length_consis, int num_pivot, float radiu_nms, float tau_inlier, const std::string &metric_str)
    : max_N(max_N), tau_length_consis(tau_length_consis), radiu_nms(radiu_nms), num_pivot(num_pivot), tau_inlier(tau_inlier)
{
    this->eval_metric = string_to_metric_type(metric_str);
}

RigidTransform TurboRegGPU::runRegCXX(torch::Tensor &kpts_src, torch::Tensor &kpts_dst)
{
    // Control the number of keypoints
    auto N_node = std::min(int(kpts_src.size(0)), max_N);
    if (N_node < kpts_src.size(0))
    {
        kpts_src = kpts_src.slice(0, 0, N_node);
        kpts_dst = kpts_dst.slice(0, 0, N_node);
    }

    // Compute C2
    auto src_dist = torch::norm(kpts_src.unsqueeze(1) - kpts_src.unsqueeze(0), 2, -1);
    auto target_dist = torch::norm(kpts_dst.unsqueeze(1) - kpts_dst.unsqueeze(0), 2, -1);
    auto cross_dist = torch::abs(src_dist - target_dist);
    torch::Tensor C2;

    if (!hard)
    {
        C2 = torch::relu(1 - torch::pow(cross_dist / tau_length_consis, 2));
    }
    else
    {
        C2 = (cross_dist < tau_length_consis).to(torch::kFloat32);
    }

    // Apply mask based on distance threshold
    auto mask = (src_dist + target_dist) <= radiu_nms;
    C2.masked_fill_(mask, 0);

    auto SC2 = torch::matmul(C2, C2) * C2;

    // Select pivots
    auto SC2_up = torch::triu(SC2, 1);                      // Upper triangular matrix, remove diagonal
    auto flat_SC2_up = SC2_up.flatten();                    // Flatten matrix
    auto topk_result = torch::topk(flat_SC2_up, num_pivot); // Select top-K elements
    auto scores_topk = std::get<0>(topk_result);            // Top-K scores
    auto idx_topk = std::get<1>(topk_result);               // Top-K indices

    auto pivots = torch::stack({(idx_topk / N_node).to(torch::kLong),
                                (idx_topk % N_node).to(torch::kLong)},
                               1);

    // Calculate threshold
    auto SC2_for_search = SC2_up.clone();

    // Find 3-cliques
    auto SC2_pivot_0 = SC2_for_search.index_select(0, pivots.select(1, 0)) > 0;
    auto SC2_pivot_1 = SC2_for_search.index_select(0, pivots.select(1, 1)) > 0;
    auto indic_c3_torch = SC2_pivot_0 & SC2_pivot_1;

    auto SC2_pivots = SC2_for_search.index({pivots.select(1, 0), pivots.select(1, 1)});

    // Calculate scores for each 3-clique using broadcasting
    auto SC2_ADD_C3 = SC2_pivots.unsqueeze(1) +
                      (SC2_for_search.index_select(0, pivots.select(1, 0)) +
                       SC2_for_search.index_select(0, pivots.select(1, 1)));

    // Mask the C3 scores
    auto SC2_C3 = SC2_ADD_C3 * indic_c3_torch.to(torch::kFloat32);

    // Get top-2 indices for each row
    auto topk_result_row = torch::topk(SC2_C3, /*k=*/2, /*dim=*/1);
    auto topk_K2 = std::get<1>(topk_result_row); // Top-K indices

    // Initialize cliques tensor, size (num_pivots*2, 3)
    int num_pivots = pivots.size(0);
    auto cliques_tensor = torch::zeros({num_pivots * 2, 3}, torch::kInt32).to(torch::kCUDA);

    // Upper part
    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), torch::indexing::Slice(0, 2)}, pivots);
    cliques_tensor.index_put_({torch::indexing::Slice(0, num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 0}));

    // Lower part
    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), torch::indexing::Slice(0, 2)}, pivots);
    cliques_tensor.index_put_({torch::indexing::Slice(num_pivots, 2 * num_pivots), 2}, topk_K2.index({torch::indexing::Slice(), 1}));

    torch::Tensor best_in_num, best_trans, res, cliques_wise_trans, cliquewise_in_num;
    ModelSelection model_selector(this->eval_metric, this->tau_inlier);
    verificationV2Metric(cliques_tensor, kpts_src, kpts_dst, model_selector,
                         best_in_num, best_trans, res,
                         cliques_wise_trans);

    // Post refinement
    torch::Tensor refined_trans = post_refinement(best_trans, kpts_src, kpts_dst, 20, this->tau_inlier);
    RigidTransform trans_final(refined_trans);
    return trans_final;
}

torch::Tensor TurboRegGPU::runRegCXXReturnTensor(torch::Tensor &kpts_src_all, torch::Tensor &kpts_dst_all)
{
    return runRegCXX(kpts_src_all, kpts_dst_all).getTransformation();
}