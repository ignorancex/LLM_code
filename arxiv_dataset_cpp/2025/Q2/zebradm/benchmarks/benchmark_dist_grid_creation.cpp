/*
Copyright (c) 2024 Sebastian Sassi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
*/
#include <random>
#include <fstream>

#include "zest/zernike_glq_transformer.hpp"

#include "coordinate_transforms.hpp"

#include "nanobench.h"
#include "distributions.hpp"

void benchmark_distribution_zernike_grid_construction(
    ankerl::nanobench::Bench& bench, const char* name,
    DistributionCartesian dist, std::size_t order)
{
    zest::zt::BallGLQGridPoints points(order);
    zest::zt::BallGLQGrid<double> grid(order);
    auto dist_spherical = [&](double lon, double colat, double r)
    {
        return dist(zdm::coordinates::spherical_to_cartesian_phys(lon, colat, r));
    };
    bench.run(name, [&](){
        points.generate_values(grid, dist_spherical);
    });
}

void run_benchmarks(
    DistributionCartesian dist, const char* dist_name, std::span<const std::size_t> orders)
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));
    bench.maxEpochTime(std::chrono::nanoseconds(100000000000));

    bench.title("zest::zt::BallGLQGridPoints::generate_values");
    for (std::size_t order : orders)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", order);
        benchmark_distribution_zernike_grid_construction(
                bench, name, dist, order);
    }

    char fname[512] = {};
    std::sprintf(fname, "distribution_grid_construction_bench_%s.json", dist_name);

    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}

template <typename Object>
struct Labeled
{
    Object object;
    const char* label;
};

int main([[maybe_unused]] int argc, char** argv)
{
    constexpr std::array<Labeled<DistributionCartesian>, 5> distributions = {
        Labeled<DistributionCartesian>{aniso_gaussian, "aniso_gaussian"},
        Labeled<DistributionCartesian>{four_gaussians, "four_gaussians"},
        Labeled<DistributionCartesian>{shm_plus_stream, "shm_plus_stream"},
        Labeled<DistributionCartesian>{shmpp_aniso, "shmpp_aniso"},
        Labeled<DistributionCartesian>{shmpp, "shmpp"}
    };

    if (argc < 2)
        throw std::runtime_error(
            "Requires argument:\n"
            "   dist_ind:   index of distribution {0,1,2,3,4}");

    const std::size_t dist_ind = atoi(argv[1]);

    std::vector<std::size_t> orders = {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200
    };

    const Labeled<DistributionCartesian> dist = distributions[dist_ind];

    run_benchmarks(dist.object, dist.label, orders);
}
