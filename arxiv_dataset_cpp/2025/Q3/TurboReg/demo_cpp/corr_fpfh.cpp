#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <regex>

#include <turboreg/turboreg.hpp>
#include <turboreg/utils_pcr.hpp>
#include <turboreg/rigid_transform.hpp>

namespace fs = std::filesystem;

torch::Tensor load_keypoints(const std::string &filename)
{
    std::ifstream infile(filename);
    if (!infile.is_open())
        throw std::runtime_error("Cannot open file: " + filename);

    std::vector<float> points;
    std::string line;
    int count = 0;
    while (std::getline(infile, line))
    {
        std::istringstream ss(line);
        float x, y, z;
        ss >> x >> y >> z;
        points.push_back(x);
        points.push_back(y);
        points.push_back(z);
        count++;
    }
    infile.close();

    torch::Tensor tensor = torch::from_blob(points.data(), {count, 3}, torch::kFloat32).clone();
    return tensor;
}

torch::Tensor load_trans_libtorch(const std::string &filename)
{
    std::ifstream infile(filename);
    if (!infile.is_open())
        throw std::runtime_error("Cannot open transformation file: " + filename);

    std::vector<float> data;
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream ss(line);
        float val;
        while (ss >> val)
        {
            data.push_back(val);
        }
    }
    infile.close();

    if (data.size() != 16)
        throw std::runtime_error("Transformation file does not contain 16 elements: " + filename);

    torch::Tensor trans = torch::from_blob(data.data(), {4, 4}, torch::kFloat32).clone();
    return trans;
}

int main()
{
    std::string data_dir = "../demo_data";

    std::regex pattern("(\\d+)_fpfh_kpts_src.txt");
    std::vector<std::string> idx_list;

    for (const auto &entry : fs::directory_iterator(data_dir))
    {
        if (!entry.is_regular_file())
            continue;

        std::smatch match;
        std::string filename = entry.path().filename().string();
        if (std::regex_match(filename, match, pattern))
        {
            std::string idx = match[1];
            idx_list.push_back(idx);
        }
    }

    std::sort(idx_list.begin(), idx_list.end());

    turboreg::TurboRegGPU reger(6000, 0.1f, 2500, 0.15f, 0.4f, "IN");

    while (true)
    {

        for (const auto &idx_str : idx_list)
        {
            std::cout << "Processing index: " << idx_str << std::endl;

            std::string src_file = data_dir + "/" + idx_str + "_fpfh_kpts_src.txt";
            std::string dst_file = data_dir + "/" + idx_str + "_fpfh_kpts_dst.txt";
            std::string trans_file = data_dir + "/" + idx_str + "_trans.txt";

            try
            {
                auto kpts_src = load_keypoints(src_file);
                auto kpts_dst = load_keypoints(dst_file);
                auto trans_gt = load_trans_libtorch(trans_file);

                if (torch::cuda::is_available())
                {
                    kpts_src = kpts_src.to(torch::kCUDA);
                    kpts_dst= kpts_dst.to(torch::kCUDA);
                }

                auto start = std::chrono::high_resolution_clock::now();
                auto trans_pred = reger.runRegCXXReturnTensor(kpts_src, kpts_dst);
                auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::high_resolution_clock::now() - start)
                                   .count();

                auto trans_pred_cpu = trans_pred.cpu();
                std::cout << "Estimated transformation matrix:\n"
                          << trans_pred_cpu << std::endl;

                double RE, TE;
                bool is_succ = turboreg::evaluationEst(trans_pred_cpu, trans_gt, 5, 60, RE, TE);

                std::cout << "RE: " << RE << ", TE: " << TE << ", Success: " << (is_succ ? "Yes" : "No")
                          << ", Time(ms): " << time_ms << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error processing " << idx_str << ": " << e.what() << std::endl;
            }
        }
    }

    return 0;
}