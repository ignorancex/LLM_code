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
#include "zebra_radon.hpp"

#include <cassert>

#include <zest/zernike_conventions.hpp>

#include "radon_util.hpp"

namespace zdm
{
namespace zebra
{

void radon_transform(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    constexpr zest::zt::ZernikeNorm zernike_norm = ZernikeExpansionSpan<const std::array<double, 2>>::zernike_norm;

    assert(!util::have_overlap(in.flatten(), out.flatten()));
    assert(in.order() + 2 == out.order());
    const std::size_t out_order = out.order();
    const std::size_t in_order = in.order();

    if (in_order == 0) return;
    
    out(0,0,0) = {
        util::geg_rec_coeff<zernike_norm>(0)*in(0,0,0)[0],
        util::geg_rec_coeff<zernike_norm>(0)*in(0,0,0)[1]
    };

    if (in_order > 1)
    {
        const double coeff = util::geg_rec_coeff<zernike_norm>(1);
        out(1,1,0) = {coeff*in(1,1,0)[0], coeff*in(1,1,0)[1]};
        out(1,1,1) = {coeff*in(1,1,1)[0], coeff*in(1,1,1)[1]};
    }
    else
    {
        out(2,0,0) = {(-util::geg_rec_coeff<zernike_norm>(0))*in(0,0,0)[0], (-util::geg_rec_coeff<zernike_norm>(0))*in(0,0,0)[1]};
        out(2,2,0) = std::array<double, 2>{};
        out(2,2,1) = std::array<double, 2>{};
        out(2,2,2) = std::array<double, 2>{};
        return;
    }

    for (std::size_t n = 2; n < in_order; ++n)
    {
        auto out_n = out[n];
        auto in_n = in[n];
        auto in_nm2 = in[n - 2];

        const double coeff_n = util::geg_rec_coeff<zernike_norm>(n);
        const double coeff_nm2 = -util::geg_rec_coeff<zernike_norm>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto out_n_l = out_n[l];
            auto in_n_l = in_n[l];
            auto in_nm2_l = in_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                out_n_l[m][0] = coeff_n*in_n_l[m][0] + coeff_nm2*in_nm2_l[m][0];
                out_n_l[m][1] = coeff_n*in_n_l[m][1] + coeff_nm2*in_nm2_l[m][1];
            }
        }

        auto out_n_n = out_n[n];
        auto in_n_n = in_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            out_n_n[m][0] = coeff_n*in_n_n[m][0];
            out_n_n[m][1] = coeff_n*in_n_n[m][1];
        }
    }

    for (std::size_t n = std::max(in_order, 2UL); n < out_order; ++n)
    {
        auto out_n = out[n];
        auto in_nm2 = in[n - 2];

        const double coeff_nm2 = -util::geg_rec_coeff<zernike_norm>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                out_n_l[m][0] = coeff_nm2*in_nm2_l[m][0];
                out_n_l[m][1] = coeff_nm2*in_nm2_l[m][1];
            }
        }

        auto out_n_n = out_n[n];
        for (std::size_t m = 0; m <= n; ++m)
            out_n_n[m] = std::array<double, 2>{};
    }
}

void radon_transform_inplace(
    ZernikeExpansionSpan<std::array<double, 2>> exp) noexcept
{
    constexpr zest::zt::ZernikeNorm zernike_norm = ZernikeExpansionSpan<const std::array<double, 2>>::zernike_norm;

    const std::size_t order = exp.order();

    if (order < 3) return;

    for (std::size_t n = order - 1; n > std::max(order - 3, 1UL); --n)
    {
        auto exp_n = exp[n];
        auto exp_nm2 = exp[n - 2];

        const double coeff_nm2 = -util::geg_rec_coeff<zernike_norm>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto exp_n_l = exp_n[l];
            auto exp_nm2_l = exp_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                exp_n_l[m][0] = coeff_nm2*exp_nm2_l[m][0];
                exp_n_l[m][1] = coeff_nm2*exp_nm2_l[m][1];
            }
        }

        auto out_n_n = exp_n[n];
        for (std::size_t m = 0; m <= n; ++m)
            out_n_n[m] = std::array<double, 2>{};
    }

    for (std::size_t n = std::max(order - 3, 1UL); n > 1; --n)
    {
        auto exp_n = exp[n];
        auto exp_nm2 = exp[n - 2];

        const double coeff_n = util::geg_rec_coeff<zernike_norm>(n);
        const double coeff_nm2 = -util::geg_rec_coeff<zernike_norm>(n - 2);
        for (std::size_t l = n & 1; l <= n - 2; l += 2)
        {
            auto exp_n_l = exp_n[l];
            auto exp_nm2_l = exp_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                exp_n_l[m][0] = coeff_n*exp_n_l[m][0] + coeff_nm2*exp_nm2_l[m][0];
                exp_n_l[m][1] = coeff_n*exp_n_l[m][1] + coeff_nm2*exp_nm2_l[m][1];
            }
        }

        auto exp_n_n = exp_n[n];
        for (std::size_t m = 0; m <= n; ++m)
        {
            exp_n_n[m][0] = coeff_n*exp_n_n[m][0];
            exp_n_n[m][1] = coeff_n*exp_n_n[m][1];
        }
    }

    const double coeff = util::geg_rec_coeff<zernike_norm>(1);
    exp(1,1,0) = {coeff*exp(1,1,0)[0], coeff*exp(1,1,0)[1]};
    exp(1,1,1) = {coeff*exp(1,1,1)[0], coeff*exp(1,1,1)[1]};

    exp(0,0,0) = {
        util::geg_rec_coeff<zernike_norm>(0)*exp(0,0,0)[0],
        util::geg_rec_coeff<zernike_norm>(0)*exp(0,0,0)[1]
    };
}

} // namespace zebra
} // namespace zdm