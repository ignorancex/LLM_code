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
#include "zebra_angle_integrator.hpp"

#include <cassert>

[[maybe_unused]] constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

[[maybe_unused]] constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::max(std::fabs(a[0] - b[0]), std::fabs(a[1] - b[1])) < tol;
}

template <typename T, std::size_t N>
double horner(const std::array<T, N>& coeffs, T x)
{
    T res{};
    for (const T& coeff : coeffs | std::views::reverse)
        res = coeff + res*x;
    return res;
}

std::array<double, 2> angle_integrated_const_dist_radon_pair(
    double shell, const std::array<double, 3>& offset)
{
    const double offset_len = zdm::length(offset);
    const double v = offset_len;
    const double v2 = v*v;
    const double w = shell;
    const double w2 = w*w;
    const double zmin = std::max(-1.0, -(1.0 + w)/v);
    const double zmax = std::min(1.0, (1.0 - w)/v);

    const std::array<double, 5> trans_coeffs = {
        v2*(1.0 - w2) - w2 + 0.5*(1.0 + w2*w2),
        -w*v*(1.0 + v2 - w2),
        -(1.0/3.0)*(2.0 + v2 - 4.0*w2)*v2,
        w*v2*v,
        0.3*v2*v2
    };

    const std::array<double, 3> nontrans_coeffs = {
        1.0 - w2, -w*v, -(1.0/3.0)*v2
    };

    constexpr double two_pi_sq = 2.0*std::numbers::pi*std::numbers::pi;
    const double trans_res = zmax*horner(trans_coeffs, zmax) - zmin*horner(trans_coeffs, zmin);
    const double nontrans_res = zmax*horner(nontrans_coeffs, zmax) - zmin*horner(nontrans_coeffs, zmin);
    return {two_pi_sq*nontrans_res, two_pi_sq*trans_res};
}

bool test_transverse_angle_integrator_is_correct_for_constant_dist()
{
    std::vector<std::array<double, 3>> offsets = {
        {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    std::vector<double> shells = {0.0};//, 0.5, 1.0, 1.5};

    std::vector<std::array<double, 2>> reference_buffer(offsets.size()*shells.size());
    zest::MDSpan<std::array<double, 2>, 2> reference(
            reference_buffer.data(), {offsets.size(), shells.size()});

    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            reference(i, j) = angle_integrated_const_dist_radon_pair(
                    shells[j], offsets[i]);
        }
    }

    constexpr std::size_t order = 1;
    zest::zt::RealZernikeExpansionNormalGeo distribution(order);
    distribution(0,0,0) = {1.0/std::sqrt(3.0), 0.0};

    std::vector<std::array<double, 2>> test_buffer(offsets.size()*shells.size());
    zest::MDSpan<std::array<double, 2>, 2> test(
            test_buffer.data(), {offsets.size(), shells.size()});

    zdm::zebra::IsotropicTransverseAngleIntegrator(order)
        .integrate(distribution, offsets, shells, test);
    
    constexpr double tol = 1.0e-13;
    bool success = true;
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
            success = success && is_close(test(i,j), reference(i,j), tol);
    }

    if (!success)
    {
        std::printf("reference\n");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::printf("{%.16e, %.16e} ", reference(i, j)[0], reference(i, j)[1]);
            }
            std::printf("\n");
        }

        std::printf("\ntest\n");
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                std::printf("{%.16e, %.16e} ", test(i, j)[0], test(i, j)[1]);
            }
            std::printf("\n");
        }
    }

    return success;
}

int main()
{
    assert(test_transverse_angle_integrator_is_correct_for_constant_dist());
}
