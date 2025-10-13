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

#include "zebra_angle_integrator.hpp"
#include "nanobench.h"

void benchmark_zebra_isotropic_angle_integrator(
    ankerl::nanobench::Bench& bench, const char* name, double offset_len, std::size_t num_offsets, std::size_t num_shells, std::size_t order)
{
    std::mt19937 gen;
    std::uniform_real_distribution dist{0.0, 1.0};

    zest::zt::RealZernikeExpansionNormalGeo distribution(order);
    for (auto& element : distribution.flatten())
        element = {dist(gen), dist(gen)};

    std::vector<std::array<double, 3>> offsets(num_offsets);
    for (auto& element : offsets)
    {
        const double ct = 2.0*dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*dist(gen);
        element = {
            offset_len*st*std::cos(az), offset_len*st*std::sin(az), ct
        };
    }

    std::vector<double> shells(num_shells);
    for (std::size_t i = 0; i < num_shells; ++i)
        shells[i] = double(i)*(offset_len + 1.0)/double(num_shells - 1);

    std::vector<double> out_buffer(num_offsets*num_shells);
    zest::MDSpan<double, 2> out(out_buffer.data(), {offsets.size(), shells.size()});

    zdm::zebra::IsotropicAngleIntegrator integrator(order);
    bench.run(name, [&](){
        integrator.integrate(
                distribution, offsets, shells, out);
    });
}

int main([[maybe_unused]] int argc, char** argv)
{
    ankerl::nanobench::Bench bench{};
    bench.performanceCounters(true);
    bench.minEpochTime(std::chrono::nanoseconds(1000000000));

    if (argc < 4)
        throw std::runtime_error(
            "Requires arguments:\n"
            "   offset_len:      length of offset vector (float)\n"
            "   num_offsets:     number of offset vectors (positive integer)\n"
            "   num_shells: number of shell values (positive integer)");

    const double offset_len = atof(argv[1]);
    const std::size_t num_offsets = atoi(argv[2]);
    const std::size_t num_shells = atoi(argv[3]);

    std::vector<std::size_t> order_vec = {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200
    };

    bench.title("zebra::IsotropicAngleIntegrator::integrate");
    for (const auto& order : order_vec)
    {
        char name[32] = {};
        std::sprintf(name, "%lu", order);
        benchmark_zebra_isotropic_angle_integrator(
                bench, name, offset_len, num_offsets, num_shells, order);
    }

    char fname[128] = {};
    std::sprintf(fname, "zebra_isotropic_angle_integrator_bench_%.2f_%lu_%lu.json", offset_len, num_offsets, num_shells);

    std::ofstream output{};
    output.open(fname);
    bench.render(ankerl::nanobench::templates::json(), output);
    output.close();
}
