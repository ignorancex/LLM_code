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

#include "zest/sh_glq_transformer.hpp"

#include "coordinate_transforms.hpp"

#include "nanobench.h"
#include "responses.hpp"

void benchmark_response_grid_construction(
    ankerl::nanobench::Bench& bench, const char* name,
    Response resp, double offset_len, std::size_t num_shells, std::size_t order)
{
    const std::size_t grid_size = zest::st::SphereGLQGridSpan<double>::size(order);
    std::vector<double> grids(num_shells*grid_size);

    zest::st::SphereGLQGrid<double> grid(order);
    zest::st::SphereGLQGridPoints<> points(order);
    zest::st::GLQTransformerGeo transformer(order);

    std::vector<double> shells(num_shells);
    for (std::size_t i = 0; i < num_shells; ++i)
        shells[i] = double(i)*(offset_len + 1.0)/double(num_shells - 1);

    bench.run(name, [&](){
        for (std::size_t i = 0; i < num_shells; ++i)
        {
            zest::st::SphereGLQGridSpan<double>
            grid(grids.data() + i*grid_size, order);
            auto f = [&](double lon, double colat)
            {
                return resp(shells[i], lon, colat);
            };

            points.generate_values(grid, f);
        }
    });
}

void run_benchmarks(
    Response resp, const char* resp_name, std::span<const std::size_t> orders,
    double offset_len, std::size_t num_shells)
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
        benchmark_response_grid_construction(
                bench, name, resp, offset_len, num_shells, order);
    }

    char fname[512] = {};
    std::sprintf(fname, "response_grid_construction_bench_%s_%.2f_%lu.json", resp_name, offset_len, num_shells);

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
    constexpr std::array<Labeled<Response>, 2> responses = {
        Labeled<Response>{smooth_exponential, "smooth_exponential"},
        Labeled<Response>{smooth_dots, "smooth_dots"}
    };

    if (argc < 2)
        throw std::runtime_error(
            "Requires argument:\n"
            "   resp_ind:   index of response {0,1}\n"
            "   offset_len:  length of offset vector (float)\n"
            "   num_shells: number of shell values (positive integer)");

    const std::size_t resp_ind = atoi(argv[1]);
    const double offset_len = atof(argv[2]);
    const std::size_t num_shells = atoi(argv[3]);

    std::vector<std::size_t> orders = {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 240, 280, 320, 400, 480
    };

    const Labeled<Response> dist = responses[resp_ind];

    run_benchmarks(dist.object, dist.label, orders, offset_len, num_shells);
}
