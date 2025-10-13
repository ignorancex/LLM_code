#include <turboreg/model_selection.hpp>
#include <torch/torch.h>
#include <unordered_map>
#include <stdexcept>
#include <turboreg/utils_debug.hpp>

// Utility function to convert string to MetricType enum
turboreg::MetricType turboreg::string_to_metric_type(const std::string &metric_str)
{
    static const std::unordered_map<std::string, MetricType> string_to_enum = {
        {"IN", MetricType::INLIER_COUNT},
        {"MAE", MetricType::MAE},
        {"MSE", MetricType::MSE}};

    auto it = string_to_enum.find(metric_str);
    if (it == string_to_enum.end())
    {
        throw std::invalid_argument("Invalid metric type string: " + metric_str);
    }
    return it->second;
}

turboreg::ModelSelection::ModelSelection(const std::string &metric_str, float inlier_threshold)
    : metric_type(string_to_metric_type(metric_str)), inlier_threshold(inlier_threshold) {}

turboreg::ModelSelection::ModelSelection(MetricType metric, float inlier_threshold) : metric_type(metric), inlier_threshold(inlier_threshold) {}

torch::Tensor turboreg::ModelSelection::calculate_best_clique(
    const torch::Tensor &cliques_wise_trans,
    const torch::Tensor &kpts_src,
    const torch::Tensor &kpts_dst)
{
    // Sub kpts
    auto cliques_wise_trans_3x3 = cliques_wise_trans.slice(1, 0, 3).slice(2, 0, 3);
    auto cliques_wise_trans_3x1 = cliques_wise_trans.slice(1, 0, 3).slice(2, 3, 4);

    // Transform the source keypoints
    auto kpts_src_prime = torch::einsum("cnm,mk->cnk", {cliques_wise_trans_3x3, kpts_src.permute({1, 0})}) + cliques_wise_trans_3x1;
    kpts_src_prime = kpts_src_prime.permute({0, 2, 1}); // Adjust dimensions

    // Calculate residuals: res = ||kpts_src_prime - kpts_dst[None, :]||
    auto res = torch::norm(kpts_src_prime - kpts_dst.unsqueeze(0), 2, -1); // (C, N)
    auto indic_in = res < inlier_threshold;                                // Inlier markers

    // Count inliers for each clique
    auto cliquewise_in_num = indic_in.sum(/*dim=*/-1).to(torch::kFloat32); // (C,)

    // Choose the best clique based on the selected metric
    torch::Tensor idx_best_guess;
    if (metric_type == MetricType::INLIER_COUNT)
    {
        idx_best_guess = cliquewise_in_num.argmax(); // Select the clique with the most inliers
    }
    else if (metric_type == MetricType::MAE)
    {
        auto mae_weights = (inlier_threshold - res).clamp_min(0) / inlier_threshold;
        return (indic_in.to(torch::kFloat32) * mae_weights).sum(-1).argmax();
    }
    else if (metric_type == MetricType::MSE)
    {
        auto mse_weights = (inlier_threshold - res).clamp_min(0).pow(2) / (inlier_threshold * inlier_threshold);
        return (indic_in.to(torch::kFloat32) * mse_weights).sum(/*dim=*/-1).argmax(); // Return the index of the clique with the smallest MSE
    }

    return idx_best_guess; // Return the index of the best clique
}
