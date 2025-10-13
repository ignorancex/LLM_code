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
#include "affine_legendre_integral.hpp"

#include <fstream>
#include <cmath>
#include <random>

[[maybe_unused]] void print(const char* fname, zdm::zebra::TrapezoidSpan<double> data)
{
    std::ofstream output{};
    output.open(fname);
    for (std::size_t n = 0; n < data.order(); ++n)
    {
        for (std::size_t l = 0; l <= n + data.extra_extent(); ++l)
        {
            char line[128] = {};
            std::sprintf(line, "%.16e ", std::fabs(data(n, l)));
            output << line;
        }
        for (std::size_t l = n + 1 + data.extra_extent(); l < data.order() + data.extra_extent(); ++l)
        {
            char line[128] = {};
            std::sprintf(line, "%.16e ", 0.0);
            output << line;
        }
        output << '\n';
    }
    output.close();
}

void exp_affine_legendre_integral_recursion_stability(
    double shift, double scale)
{
    constexpr std::size_t order = 200;
    constexpr std::size_t extra_extent = 500;

    zdm::zebra::AffineLegendreIntegrals recursion(order, extra_extent);

    std::vector<double> trapezoid_buffer(
        zdm::zebra::TrapezoidSpan<double>::Layout::size(order, extra_extent));
    
    zdm::zebra::TrapezoidSpan<double> test_trapezoid(
        trapezoid_buffer.data(), order, extra_extent);
    
    recursion.integrals(
            test_trapezoid, shift, scale);

    char fname[128] = {};
    std::sprintf(fname, "aff_leg_coeff_%.2f_%.2f.dat", shift, scale);
    print(fname, test_trapezoid);
}

int main()
{
    std::vector<double> shifts = {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5};
    std::vector<double> scales = {0.25, 0.5, 0.75, 1.0};

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> scale_dist(0.25, 0.75);
    std::uniform_real_distribution<> shift_dist(0.00, 1.50);

    //double maxerr = 0.0;
    for (std::size_t i = 0; i < 1000000; ++i)
    {
        const double shift = shift_dist(gen);
        const double scale = scale_dist(gen);
        exp_affine_legendre_integral_recursion_stability(shift, scale);
    }
}