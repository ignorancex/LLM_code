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
#include "zernike_recursions.hpp"

#include <algorithm>
#include <cassert>

#include "zebra_radon.hpp"
#include "radon_util.hpp"

namespace zdm
{
namespace zebra
{
namespace detail
{

ZernikeRecursionData::ZernikeRecursionData(std::size_t order):
    m_sqrt_n(2*order + 1), m_inv_sqrt_2np1_2np3(order + 2), m_order(order)
{
    if (order == 0) return;

    for (std::size_t n = 1; n < m_sqrt_n.size(); ++n)
        m_sqrt_n[n] = std::sqrt(double(n));
    
    for (std::size_t n = 0; n < m_inv_sqrt_2np1_2np3.size(); ++n)
    {
        const double _2n = double(2*n);
        m_inv_sqrt_2np1_2np3[n] = 1.0/std::sqrt((_2n + 1.0)*(_2n + 3.0));
    }
}

void ZernikeRecursionData::expand(std::size_t order)
{
    if (order <= m_order) return;

    const std::size_t old_n_size = m_sqrt_n.size();
    const std::size_t old_2np1_2np3_size = m_inv_sqrt_2np1_2np3.size();

    const std::size_t new_n_size = 2*order + 1;
    const std::size_t new_2np1_2np3_size = order + 2;

    m_sqrt_n.resize(new_n_size);
    m_inv_sqrt_2np1_2np3.resize(new_2np1_2np3_size);

    for (std::size_t n = old_n_size; n < new_n_size; ++n)
        m_sqrt_n[n] = std::sqrt(double(n));
    
    for (std::size_t n = old_2np1_2np3_size; n < new_2np1_2np3_size; ++n)
    {
        const double _2n = double(2*n);
        m_inv_sqrt_2np1_2np3[n] = 1.0/std::sqrt((_2n + 1.0)*(_2n + 3.0));
    }
}

enum class PlaneCoord { X, Y };

template <PlaneCoord coord_param>
void multiply_by_x_y_impl(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    /*
    Literally the worst function I have ever written.

    The base case is that `out(n,l,m)` is a linear combination of the coefficients
    ```
    in(n + 1, l + 1, m + 1)
    in(n + 1, l + 1, m - 1)
    in(n + 1, l - 1, m + 1)
    in(n + 1, l - 1, m - 1)
    in(n - 1, l + 1, m + 1)
    in(n - 1, l + 1, m - 1)
    in(n - 1, l - 1, m + 1)
    in(n - 1, l - 1, m - 1)
    ```
    However, for `in(n,l,m)`, we have
    ```
    abs(m) <= l <= n < in.order()
    ```
    These conditions on the indices lead to multiple edge cases where different coefficients are neglected.
    */

    constexpr double sqrt2 = std::numbers::sqrt2;
    constexpr double sqrt3 = std::numbers::sqrt3;
    constexpr double sqrt5 = 2.2360679774997896964091737;
    constexpr double sqrt7 = 2.6457513110645905905016158;

    assert(in.order() < out.order());
    assert(out.order() <= coeff_data.order());

    const std::size_t nmax = in.order();
    if (nmax == 0) return;

    std::ranges::fill(out.flatten(), std::array<double, 2>{});

    if constexpr (coord_param == PlaneCoord::X)
    {
        out(1,1,1)[0] = (-1.0/sqrt5)*in(0,0,0)[0];
        if (nmax == 1) return;

        out(0,0,0)[0] = (-1.0/sqrt5)*in(1,1,1)[0];
        out(2,0,0)[0] = (-2.0/(sqrt3*sqrt5*sqrt7))*in(1,1,1)[0];
        out(2,2,0)[0] = (1.0/(sqrt3*sqrt7))*in(1,1,1)[0];
        out(2,2,1)[0] = (-1.0/sqrt7)*in(1,1,0)[0];
        out(2,2,2)[0] = (-1.0/sqrt7)*in(1,1,1)[0];
        out(2,2,2)[1] = (-1.0/sqrt7)*in(1,1,1)[1];
        if (nmax == 2) return;
        
        out(1,1,0)[0] = (-1.0/sqrt7)*in(2,2,1)[0];
    
        out(1,1,1)[0] -= (2.0/(sqrt3*sqrt5*sqrt7))*in(2,0,0)[0]
                - (1.0/(sqrt3*sqrt7))*in(2,2,0)[0]
                + (1.0/sqrt7)*in(2,2,2)[0];

        out(1,1,1)[1] -= (1.0/sqrt7)*in(2,2,2)[1];
            
        if (nmax > 3)
        {
            out(2,0,0)[0] -= (5.0/(3.0*sqrt3*sqrt7))*in(3,1,1)[0];
            out(2,2,0)[0] += (2.0/(3.0*sqrt3*sqrt5*sqrt7))*in(3,1,1)[0]
                    - (sqrt2/(sqrt3*sqrt5))*in(3,3,1)[0];
            out(2,2,1)[0] -= (2.0/(3.0*sqrt5*sqrt7))*in(3,1,0)[0]
                    - (1.0/(sqrt3*sqrt5))*in(3,3,0)[0]
                    + (1.0/3.0)*in(3,3,2)[0];
            out(2,2,1)[1] -= (1.0/3.0)*in(3,3,2)[1];
            out(2,2,2)[0] -= (2.0/(3.0*sqrt5*sqrt7))*in(3,1,1)[0]
                    - (1.0/(3.0*sqrt2*sqrt5))*in(3,3,1)[0]
                    + (1.0/(sqrt2*sqrt3))*in(3,3,3)[0];
            out(2,2,2)[1] -= (2.0/(3.0*sqrt5*sqrt7))*in(3,1,1)[1]
                    - (1.0/(3.0*sqrt2*sqrt5))*in(3,3,1)[1]
                    + (1.0/(sqrt2*sqrt3))*in(3,3,3)[1];
        }
    }
    else
    {
        out(1,1,1)[1] = (-1.0/sqrt5)*in(0,0,0)[0];
        if (nmax == 1) return;

        out(0,0,0)[0] = (-1.0/sqrt5)*in(1,1,1)[1];
        out(2,0,0)[0] = (-2.0/(sqrt3*sqrt5*sqrt7))*in(1,1,1)[1];
        out(2,2,0)[0] = (+1.0/(sqrt3*sqrt7))*in(1,1,1)[1];
        out(2,2,1)[1] = (-1.0/sqrt7)*in(1,1,0)[0];
        out(2,2,2)[0] = (+1.0/sqrt7)*in(1,1,1)[1];
        out(2,2,2)[1] = (-1.0/sqrt7)*in(1,1,1)[0];
        if (nmax == 2) return;

        out(1,1,0)[0] = (-1.0/sqrt7)*in(2,2,1)[1];
    
        out(1,1,1)[0] -= (1.0/sqrt7)*in(2,2,2)[1];

        out(1,1,1)[1] -= (2.0/(sqrt3*sqrt5*sqrt7))*in(2,0,0)[0]
                - (1.0/(sqrt3*sqrt7))*in(2,2,0)[0]
                - (1.0/sqrt7)*in(2,2,2)[0];

        if (nmax > 3)
        {
            out(2,0,0)[0] -= (5.0/(3.0*sqrt3*sqrt7))*in(3,1,1)[1];
            out(2,2,0)[0] += (2.0/(3.0*sqrt3*sqrt5*sqrt7))*in(3,1,1)[1]
                    - (sqrt2/(sqrt3*sqrt5))*in(3,3,1)[1];
            out(2,2,1)[0] -= (1.0/3.0)*in(3,3,2)[1];
            out(2,2,1)[1] -= (2.0/(3.0*sqrt5*sqrt7))*in(3,1,0)[0]
                    - (1.0/(sqrt3*sqrt5))*in(3,3,0)[0]
                    - (1.0/3.0)*in(3,3,2)[0];
            out(2,2,2)[0] += (2.0/(3.0*sqrt5*sqrt7))*in(3,1,1)[1]
                    - (1.0/(3.0*sqrt2*sqrt5))*in(3,3,1)[1]
                    - (1.0/(sqrt2*sqrt3))*in(3,3,3)[1];
            out(2,2,2)[1] -= (2.0/(3.0*sqrt5*sqrt7))*in(3,1,1)[0]
                    - (1.0/(3.0*sqrt2*sqrt5))*in(3,3,1)[0]
                    - (1.0/(sqrt2*sqrt3))*in(3,3,3)[0];
        }
    }

    for (std::size_t n = 3; n < nmax - 1; ++n)
    {
        const double n_coeff_m = 0.5*coeff_data.inv_sqrt_2np1_2np3(n);
        const double n_coeff_p = 0.5*coeff_data.inv_sqrt_2np3_2np5(n);

        auto out_n = out[n];
        auto in_nm1 = in[n - 1];
        auto in_np1 = in[n + 1];

        const std::size_t n_parity = n & 1;
        
        // edge case: n > 2, l = 0
        // edge case: n > 2, l = 2
        if (n_parity == 0)
        {
            const double dn = double(n);
            constexpr double l_coeff = -2.0/sqrt3;
            constexpr std::size_t idx = std::size_t(coord_param == PlaneCoord::Y);
            out_n(0,0)[0] = l_coeff*(
                    n_coeff_m*dn*in_nm1(1,1)[idx]
                    + n_coeff_p*(dn + 3.0)*in_np1(1,1)[idx]);
            
            // l == 2 special case
            constexpr double l_coeff_m = 1.0/(sqrt3*sqrt5);
            constexpr double l_coeff_p = 1.0/(sqrt5*sqrt7);

            // l == 2 special case
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 3.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 2.0);
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*dn;
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*(dn + 5.0);
            
            constexpr double l0_coeff_m = 2.0;
            constexpr double l0_coeff_p = 2.0*sqrt2*sqrt3;
            constexpr double l1_coeff_mm = 2.0*sqrt3;
            constexpr double l1_coeff_pm = l1_coeff_mm;
            constexpr double l1_coeff_pp = 2.0*sqrt5;
            constexpr double l2_coeff_mm = 2.0*sqrt3;
            constexpr double l2_coeff_pm = sqrt2;
            constexpr double l2_coeff_pp = sqrt2*sqrt3*sqrt5;
            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n(2,0)[0] = nl_coeff_mm*l0_coeff_m*in_nm1(1,1)[0]
                        - nl_coeff_mp*l0_coeff_p*in_nm1(3,1)[0]
                        + nl_coeff_pm*l0_coeff_m*in_np1(1,1)[0]
                        - nl_coeff_pp*l0_coeff_p*in_np1(3,1)[0];
                
                out_n(2,1)[0]
                    = -nl_coeff_mm*l1_coeff_mm*in_nm1(1,0)[0]
                    - nl_coeff_mp*(
                        l1_coeff_pp*in_nm1(3,2)[0] - l1_coeff_pm*in_nm1(3,0)[0])
                    - nl_coeff_pm*l1_coeff_mm*in_np1(1,0)[0]
                    - nl_coeff_pp*(
                        l1_coeff_pp*in_np1(3,2)[0] - l1_coeff_pm*in_np1(3,0)[0]);
                
                out_n(2,1)[1] = -l1_coeff_pp*(
                        nl_coeff_mp*in_nm1(3,2)[1] + nl_coeff_pp*in_np1(3,2)[1]);
                
                out_n(2,2)[0]
                    = -nl_coeff_mm*l2_coeff_mm*in_nm1(1,1)[0]
                    - nl_coeff_mp*(
                        l2_coeff_pp*in_nm1(3,3)[0] - l2_coeff_pm*in_nm1(3,1)[0])
                    - nl_coeff_pm*l2_coeff_mm*in_np1(1,1)[0]
                    - nl_coeff_pp*(
                        l2_coeff_pp*in_np1(3,3)[0] - l2_coeff_pm*in_np1(3,1)[0]);
                
                out_n(2,2)[1]
                    = -nl_coeff_mm*l2_coeff_mm*in_nm1(1,1)[1]
                    - nl_coeff_mp*(
                        l2_coeff_pp*in_nm1(3,3)[1] - l2_coeff_pm*in_nm1(3,1)[1])
                    - nl_coeff_pm*l2_coeff_mm*in_np1(1,1)[1]
                    - nl_coeff_pp*(
                        l2_coeff_pp*in_np1(3,3)[1] - l2_coeff_pm*in_np1(3,1)[1]);
            }
            else
            {
                out_n(2,0)[0] = nl_coeff_mm*l0_coeff_m*in_nm1(1,1)[1]
                        - nl_coeff_mp*l0_coeff_p*in_nm1(3,1)[1]
                        + nl_coeff_pm*l0_coeff_m*in_np1(1,1)[1]
                        - nl_coeff_pp*l0_coeff_p*in_np1(3,1)[1];
                
                out_n(2,1)[0] = -l1_coeff_pp*(
                        nl_coeff_pp*in_np1(3,2)[1] + nl_coeff_mp*in_nm1(3,2)[1]);
                
                out_n(2,1)[1]
                    = -nl_coeff_mm*l1_coeff_mm*in_nm1(1,0)[0]
                    + nl_coeff_mp*(
                        l1_coeff_pp*in_nm1(3,2)[0] + l1_coeff_pm*in_nm1(3,0)[0])
                    - nl_coeff_pm*l1_coeff_mm*in_np1(1,0)[0]
                    + nl_coeff_pp*(
                        l1_coeff_pp*in_np1(3,2)[0] + l1_coeff_pm*in_np1(3,0)[0]);
                    
                out_n(2,2)[0]
                    = nl_coeff_pm*l2_coeff_mm*in_np1(1,1)[1]
                    + nl_coeff_mm*l2_coeff_mm*in_nm1(1,1)[1]
                    - nl_coeff_pp*(
                        l2_coeff_pp*in_np1(3,3)[1] + l2_coeff_pm*in_np1(3,1)[1])
                    - nl_coeff_mp*(
                        l2_coeff_pp*in_nm1(3,3)[1] + l2_coeff_pm*in_nm1(3,1)[1]);
                
                out_n(2,2)[1]
                    = -nl_coeff_pm*l2_coeff_mm*in_np1(1,1)[0]
                    - nl_coeff_mm*l2_coeff_mm*in_nm1(1,1)[0]
                    + nl_coeff_pp*(
                        l2_coeff_pp*in_np1(3,3)[0] + l2_coeff_pm*in_np1(3,1)[0])
                    + nl_coeff_mp*(
                        l2_coeff_pp*in_nm1(3,3)[0] + l2_coeff_pm*in_nm1(3,1)[0]);
            }
        }
        // edge case: n > 2, l = 1
        else
        {
            // l == 1 special case
            constexpr double l_coeff_m = 1.0/sqrt3;
            constexpr double l_coeff_p = 1.0/(sqrt3*sqrt5);

            // l == 1 special case
            const double dn = double(n);
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 2.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 1.0);
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*(dn + 1.0);
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*(dn + 4.0);

            constexpr double l1_coeff_mm = 2.0;
            constexpr double l1_coeff_pm = l1_coeff_mm;
            constexpr double l1_coeff_pp = 2.0*sqrt3;
            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n(1,0)[0] = (-2.0*sqrt3)*(
                    nl_coeff_mp*in_nm1(2,1)[0] + nl_coeff_pp*in_np1(2,1)[0]);

                out_n(1,1)[0]
                    = -nl_coeff_mm*l1_coeff_mm*in_nm1(0,0)[0]
                    - nl_coeff_mp*(
                        l1_coeff_pp*in_nm1(2,2)[0] - l1_coeff_pm*in_nm1(2,0)[0])
                    - nl_coeff_pm*l1_coeff_mm*in_np1(0,0)[0]
                    - nl_coeff_pp*(
                        l1_coeff_pp*in_np1(2,2)[0] - l1_coeff_pm*in_np1(2,0)[0]);

                out_n(1,1)[1] = (-l1_coeff_pp)*(
                        nl_coeff_mp*in_nm1(2,2)[1] + nl_coeff_pp*in_np1(2,2)[1]);
            }
            else
            {
                out_n(1,0)[0] = (-2.0*sqrt3)*(
                        nl_coeff_mp*in_nm1(2,1)[1] + nl_coeff_pp*in_np1(2,1)[1]);

                out_n(1,1)[0] = (-l1_coeff_pp)*(
                        nl_coeff_mp*in_nm1(2,2)[1] + nl_coeff_pp*in_np1(2,2)[1]);
                
                out_n(1,1)[1]
                    = -nl_coeff_mm*l1_coeff_mm*in_nm1(0,0)[0]
                    + nl_coeff_mp*(
                        l1_coeff_pp*in_nm1(2,2)[0] + l1_coeff_pm*in_nm1(2,0)[0])
                    - nl_coeff_pm*l1_coeff_mm*in_np1(0,0)[0]
                    + nl_coeff_pp*(
                        l1_coeff_pp*in_np1(2,2)[0] + l1_coeff_pm*in_np1(2,0)[0]);
            }
        }

        for (std::size_t l = 4 - n_parity; l < n; l += 2)
        {
            const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(l);
            const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(l);
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;
            const double nml2 = nml + 2.0;
            const double npl3 = npl + 3.0;

            const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*nml;
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*nml2;
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*npl3;

            auto out_n_l = out_n[l];
            auto in_nm1_lm1 = in_nm1[l - 1];
            auto in_nm1_lp1 = in_nm1[l + 1];
            auto in_np1_lm1 = in_np1[l - 1];
            auto in_np1_lp1 = in_np1[l + 1];

            // edge case: n > 2, l > 2, m = 0
            // edge case: n > 2, l > 2, m = 1
            const double l0_coeff_mp
                = sqrt2*coeff_data.sqrt_n(l - 1)*coeff_data.sqrt_n(l);
            const double l0_coeff_pp
                = sqrt2*coeff_data.sqrt_n(l + 1)*coeff_data.sqrt_n(l + 2);
            const double l1_coeff_mm
                = sqrt2*coeff_data.sqrt_n(l)*coeff_data.sqrt_n(l + 1);
            const double l1_coeff_pm = l1_coeff_mm;
            const double l1_coeff_mp
                = coeff_data.sqrt_n(l - 2)*coeff_data.sqrt_n(l - 1);
            const double l1_coeff_pp
                = coeff_data.sqrt_n(l + 2)*coeff_data.sqrt_n(l + 3);
            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n_l[0][0]
                    = nl_coeff_mm*l0_coeff_mp*in_nm1_lm1[1][0]
                    - nl_coeff_mp*l0_coeff_pp*in_nm1_lp1[1][0]
                    + nl_coeff_pm*l0_coeff_mp*in_np1_lm1[1][0]
                    - nl_coeff_pp*l0_coeff_pp*in_np1_lp1[1][0];
                
                out_n_l[1][0]
                    = nl_coeff_mm*(l1_coeff_mp*in_nm1_lm1[2][0] - l1_coeff_mm*in_nm1_lm1[0][0])
                    - nl_coeff_mp*(l1_coeff_pp*in_nm1_lp1[2][0] - l1_coeff_pm*in_nm1_lp1[0][0])
                    + nl_coeff_pm*(l1_coeff_mp*in_np1_lm1[2][0] - l1_coeff_mm*in_np1_lm1[0][0])
                    - nl_coeff_pp*(l1_coeff_pp*in_np1_lp1[2][0] - l1_coeff_pm*in_np1_lp1[0][0]);
                
                out_n_l[1][1]
                    = nl_coeff_mm*l1_coeff_mp*in_nm1_lm1[2][1]
                    - nl_coeff_mp*l1_coeff_pp*in_nm1_lp1[2][1]
                    + nl_coeff_pm*l1_coeff_mp*in_np1_lm1[2][1]
                    - nl_coeff_pp*l1_coeff_pp*in_np1_lp1[2][1];
            }
            else
            {
                out_n_l[0][0]
                    = nl_coeff_mm*l0_coeff_mp*in_nm1_lm1[1][1]
                    - nl_coeff_mp*l0_coeff_pp*in_nm1_lp1[1][1]
                    + nl_coeff_pm*l0_coeff_mp*in_np1_lm1[1][1]
                    - nl_coeff_pp*l0_coeff_pp*in_np1_lp1[1][1];
                
                out_n_l[1][0]
                    = -nl_coeff_pp*l1_coeff_pp*in_np1_lp1[2][1]
                    - l1_coeff_pp*nl_coeff_mp*in_nm1_lp1[2][1]
                    + nl_coeff_pm*l1_coeff_mp*in_np1_lm1[2][1]
                    + nl_coeff_mm*l1_coeff_mp*in_nm1_lm1[2][1];
                
                out_n_l[1][1]
                    = -nl_coeff_mm*(l1_coeff_mp*in_nm1_lm1[2][0] + l1_coeff_mm*in_nm1_lm1[0][0])
                    + nl_coeff_mp*(l1_coeff_pp*in_nm1_lp1[2][0] + l1_coeff_pm*in_nm1_lp1[0][0])
                    - nl_coeff_pm*(l1_coeff_mp*in_np1_lm1[2][0] + l1_coeff_mm*in_np1_lm1[0][0])
                    + nl_coeff_pp*(l1_coeff_pp*in_np1_lp1[2][0] + l1_coeff_pm*in_np1_lp1[0][0]);
            }

            for (std::size_t m = 2; m < l - 1; ++m)
            {
                const double lm_coeff_mm
                    = coeff_data.sqrt_n(l + m - 1)*coeff_data.sqrt_n(l + m);
                const double lm_coeff_mp
                    = coeff_data.sqrt_n(l - m - 1)*coeff_data.sqrt_n(l - m);
                const double lm_coeff_pm
                    = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l - m + 2);
                const double lm_coeff_pp
                    = coeff_data.sqrt_n(l + m + 1)*coeff_data.sqrt_n(l + m + 2);

                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_mm1 = in_nm1_lm1[m - 1];
                const std::array<double, 2> in_nm1_lm1_mp1 = in_nm1_lm1[m + 1];
                const std::array<double, 2> in_nm1_lp1_mm1 = in_nm1_lp1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mp1 = in_nm1_lp1[m + 1];
                const std::array<double, 2> in_np1_lm1_mm1 = in_np1_lm1[m - 1];
                const std::array<double, 2> in_np1_lm1_mp1 = in_np1_lm1[m + 1];
                const std::array<double, 2> in_np1_lp1_mm1 = in_np1_lp1[m - 1];
                const std::array<double, 2> in_np1_lp1_mp1 = in_np1_lp1[m + 1];

                // base case
                if constexpr (coord_param == PlaneCoord::X)
                {
                    out_n_l_m[0]
                        = nl_coeff_mm*(
                            lm_coeff_mp*in_nm1_lm1_mp1[0] - lm_coeff_mm*in_nm1_lm1_mm1[0])
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[0] - lm_coeff_pm*in_nm1_lp1_mm1[0])
                        + nl_coeff_pm*(
                            lm_coeff_mp*in_np1_lm1_mp1[0] - lm_coeff_mm*in_np1_lm1_mm1[0])
                        - nl_coeff_pp*(
                            lm_coeff_pp*in_np1_lp1_mp1[0] - lm_coeff_pm*in_np1_lp1_mm1[0]);
                    
                    out_n_l_m[1]
                        = nl_coeff_mm*(
                            lm_coeff_mp*in_nm1_lm1_mp1[1] - lm_coeff_mm*in_nm1_lm1_mm1[1])
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[1] - lm_coeff_pm*in_nm1_lp1_mm1[1])
                        + nl_coeff_pm*(
                            lm_coeff_mp*in_np1_lm1_mp1[1] - lm_coeff_mm*in_np1_lm1_mm1[1])
                        - nl_coeff_pp*(
                            lm_coeff_pp*in_np1_lp1_mp1[1] - lm_coeff_pm*in_np1_lp1_mm1[1]);
                }
                else
                {
                    out_n_l_m[0]
                        = nl_coeff_mm*(
                            lm_coeff_mp*in_nm1_lm1_mp1[1] + lm_coeff_mm*in_nm1_lm1_mm1[1])
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[1] + lm_coeff_pm*in_nm1_lp1_mm1[1])
                        + nl_coeff_pm*(
                            lm_coeff_mp*in_np1_lm1_mp1[1] + lm_coeff_mm*in_np1_lm1_mm1[1])
                        - nl_coeff_pp*(
                            lm_coeff_pp*in_np1_lp1_mp1[1] + lm_coeff_pm*in_np1_lp1_mm1[1]);
                    
                    out_n_l_m[1]
                        = nl_coeff_pp*(
                            lm_coeff_pp*in_np1_lp1_mp1[0] + lm_coeff_pm*in_np1_lp1_mm1[0])
                        - nl_coeff_pm*(
                            lm_coeff_mp*in_np1_lm1_mp1[0] + lm_coeff_mm*in_np1_lm1_mm1[0])
                        + nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[0] + lm_coeff_pm*in_nm1_lp1_mm1[0])
                        - nl_coeff_mm*(
                            lm_coeff_mp*in_nm1_lm1_mp1[0] + lm_coeff_mm*in_nm1_lm1_mm1[0]);
                }
            }

            // edge case: n > 2, l > 2, m = l - 1
            // edge case: n > 2, l > 2, m = l
            for (std::size_t m = std::max(l - 1, 2UL); m <= l; ++m)
            {
                const double lm_coeff_mm
                    = coeff_data.sqrt_n(l + m - 1)*coeff_data.sqrt_n(l + m);
                const double lm_coeff_pm
                    = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l - m + 2);
                const double lm_coeff_pp
                    = coeff_data.sqrt_n(l + m + 1)*coeff_data.sqrt_n(l + m + 2);

                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_mm1 = in_nm1_lm1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mm1 = in_nm1_lp1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mp1 = in_nm1_lp1[m + 1];
                const std::array<double, 2> in_np1_lm1_mm1 = in_np1_lm1[m - 1];
                const std::array<double, 2> in_np1_lp1_mm1 = in_np1_lp1[m - 1];
                const std::array<double, 2> in_np1_lp1_mp1 = in_np1_lp1[m + 1];

                if constexpr (coord_param == PlaneCoord::X)
                {
                    out_n_l_m[0]
                        = -nl_coeff_mm*lm_coeff_mm*in_nm1_lm1_mm1[0]
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[0] - lm_coeff_pm*in_nm1_lp1_mm1[0])
                        - nl_coeff_pm*lm_coeff_mm*in_np1_lm1_mm1[0]
                        - nl_coeff_pp*(
                            lm_coeff_pp*in_np1_lp1_mp1[0] - lm_coeff_pm*in_np1_lp1_mm1[0]);
                    
                    out_n_l_m[1]
                        = -nl_coeff_mm*lm_coeff_mm*in_nm1_lm1_mm1[1]
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[1] - lm_coeff_pm*in_nm1_lp1_mm1[1])
                        - nl_coeff_pm*lm_coeff_mm*in_np1_lm1_mm1[1]
                        - nl_coeff_pp*(
                            lm_coeff_pp*in_np1_lp1_mp1[1] - lm_coeff_pm*in_np1_lp1_mm1[1]);
                }
                else
                {
                    out_n_l_m[0]
                        = nl_coeff_mm*lm_coeff_mm*in_nm1_lm1_mm1[1]
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[1] + lm_coeff_pm*in_nm1_lp1_mm1[1])
                        + nl_coeff_pm*lm_coeff_mm*in_np1_lm1_mm1[1]
                        - nl_coeff_pp*(
                            lm_coeff_pp*in_np1_lp1_mp1[1] + lm_coeff_pm*in_np1_lp1_mm1[1]);
                    
                    out_n_l_m[1]
                        = nl_coeff_pp*(
                            lm_coeff_pp*in_np1_lp1_mp1[0] + lm_coeff_pm*in_np1_lp1_mm1[0])
                        - nl_coeff_pm*lm_coeff_mm*in_np1_lm1_mm1[0]
                        + nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[0] + lm_coeff_pm*in_nm1_lp1_mm1[0])
                        - nl_coeff_mm*lm_coeff_mm*in_nm1_lm1_mm1[0];
                }
            }
        }

        const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(n);
        const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(n);
        const double npl = double(2*n);
        const double npl1 = npl + 1.0;
        const double npl3 = npl + 3.0;

        const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
        const double nl_coeff_pm = n_coeff_p*l_coeff_m*2.0;
        const double nl_coeff_pp = n_coeff_p*l_coeff_p*npl3;

        auto out_n_n = out_n[n];
        auto in_nm1_nm1 = in_nm1[n - 1];
        auto in_np1_nm1 = in_np1[n - 1];
        auto in_np1_np1 = in_np1[n + 1];

        // edge case: n > 2, l = n, m = 0
        // edge case: n > 2, l = n, m = 1
        const double l0_coeff_mp
            = sqrt2*coeff_data.sqrt_n(n - 1)*coeff_data.sqrt_n(n);
        const double l0_coeff_pp
            = sqrt2*coeff_data.sqrt_n(n + 1)*coeff_data.sqrt_n(n + 2);
        const double l1_coeff_mm
            = sqrt2*coeff_data.sqrt_n(n)*coeff_data.sqrt_n(n + 1);
        const double l1_coeff_pm = l1_coeff_mm;
        const double l1_coeff_mp
            = coeff_data.sqrt_n(n - 2)*coeff_data.sqrt_n(n - 1);
        const double l1_coeff_pp
            = coeff_data.sqrt_n(n + 2)*coeff_data.sqrt_n(n + 3);
        if constexpr (coord_param == PlaneCoord::X)
        {
            out_n_n[0][0]
                = nl_coeff_mm*l0_coeff_mp*in_nm1_nm1[1][0]
                + nl_coeff_pm*l0_coeff_mp*in_np1_nm1[1][0]
                - nl_coeff_pp*l0_coeff_pp*in_np1_np1[1][0];
            
            out_n_n[1][0]
                = nl_coeff_mm*(l1_coeff_mp*in_nm1_nm1[2][0] - l1_coeff_mm*in_nm1_nm1[0][0])
                + nl_coeff_pm*(l1_coeff_mp*in_np1_nm1[2][0] - l1_coeff_mm*in_np1_nm1[0][0])
                - nl_coeff_pp*(l1_coeff_pp*in_np1_np1[2][0] - l1_coeff_pm*in_np1_np1[0][0]);
            
            out_n_n[1][1]
                = nl_coeff_mm*l1_coeff_mp*in_nm1_nm1[2][1]
                + nl_coeff_pm*l1_coeff_mp*in_np1_nm1[2][1]
                - nl_coeff_pp*l1_coeff_pp*in_np1_np1[2][1];
        }
        else
        {
            out_n_n[0][0]
                = nl_coeff_mm*l0_coeff_mp*in_nm1_nm1[1][1]
                + nl_coeff_pm*l0_coeff_mp*in_np1_nm1[1][1]
                - nl_coeff_pp*l0_coeff_pp*in_np1_np1[1][1];
            
            out_n_n[1][0]
                = -nl_coeff_pp*l1_coeff_pp*in_np1_np1[2][1]
                + nl_coeff_pm*l1_coeff_mp*in_np1_nm1[2][1]
                + nl_coeff_mm*l1_coeff_mp*in_nm1_nm1[2][1];
            
            out_n_n[1][1]
                = -nl_coeff_mm*(l1_coeff_mp*in_nm1_nm1[2][0] + l1_coeff_mm*in_nm1_nm1[0][0])
                - nl_coeff_pm*(l1_coeff_mp*in_np1_nm1[2][0] + l1_coeff_mm*in_np1_nm1[0][0])
                + nl_coeff_pp*(l1_coeff_pp*in_np1_np1[2][0] + l1_coeff_pm*in_np1_np1[0][0]);
        }
        
        for (std::size_t m = 2; m < n - 1; ++m)
        {
            const double lm_coeff_mm
                = coeff_data.sqrt_n(n + m - 1)*coeff_data.sqrt_n(n + m);
            const double lm_coeff_mp
                = coeff_data.sqrt_n(n - m - 1)*coeff_data.sqrt_n(n - m);
            const double lm_coeff_pm
                = coeff_data.sqrt_n(n - m + 1)*coeff_data.sqrt_n(n - m + 2);
            const double lm_coeff_pp
                = coeff_data.sqrt_n(n + m + 1)*coeff_data.sqrt_n(n + m + 2);

            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_mm1 = in_nm1_nm1[m - 1];
            const std::array<double, 2> in_nm1_nm1_mp1 = in_nm1_nm1[m + 1];
            const std::array<double, 2> in_np1_nm1_mm1 = in_np1_nm1[m - 1];
            const std::array<double, 2> in_np1_nm1_mp1 = in_np1_nm1[m + 1];
            const std::array<double, 2> in_np1_np1_mm1 = in_np1_np1[m - 1];
            const std::array<double, 2> in_np1_np1_mp1 = in_np1_np1[m + 1];

            // edge case: n > 2, l = n, m > 1
            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n_n_m[0]
                    = nl_coeff_mm*(
                        lm_coeff_mp*in_nm1_nm1_mp1[0] - lm_coeff_mm*in_nm1_nm1_mm1[0])
                    + nl_coeff_pm*(
                        lm_coeff_mp*in_np1_nm1_mp1[0] - lm_coeff_mm*in_np1_nm1_mm1[0])
                    - nl_coeff_pp*(
                        lm_coeff_pp*in_np1_np1_mp1[0] - lm_coeff_pm*in_np1_np1_mm1[0]);
                
                out_n_n_m[1]
                    = nl_coeff_mm*(
                        lm_coeff_mp*in_nm1_nm1_mp1[1] - lm_coeff_mm*in_nm1_nm1_mm1[1])
                    + nl_coeff_pm*(
                        lm_coeff_mp*in_np1_nm1_mp1[1] - lm_coeff_mm*in_np1_nm1_mm1[1])
                    - nl_coeff_pp*(
                        lm_coeff_pp*in_np1_np1_mp1[1] - lm_coeff_pm*in_np1_np1_mm1[1]);
            }
            else
            {
                out_n_n_m[0]
                    = nl_coeff_mm*(
                        lm_coeff_mp*in_nm1_nm1_mp1[1] + lm_coeff_mm*in_nm1_nm1_mm1[1])
                    + nl_coeff_pm*(
                        lm_coeff_mp*in_np1_nm1_mp1[1] + lm_coeff_mm*in_np1_nm1_mm1[1])
                    - nl_coeff_pp*(
                        lm_coeff_pp*in_np1_np1_mp1[1] + lm_coeff_pm*in_np1_np1_mm1[1]);
                
                out_n_n_m[1]
                    = nl_coeff_pp*(
                        lm_coeff_pp*in_np1_np1_mp1[0] + lm_coeff_pm*in_np1_np1_mm1[0])
                    - nl_coeff_pm*(
                        lm_coeff_mp*in_np1_nm1_mp1[0] + lm_coeff_mm*in_np1_nm1_mm1[0])
                    - nl_coeff_mm*(
                        lm_coeff_mp*in_nm1_nm1_mp1[0] + lm_coeff_mm*in_nm1_nm1_mm1[0]);
            }
        }

        // edge case: n > 2, l = n, m = l - 1
        // edge case: n > 2, l = n, m = l
        for (std::size_t m = std::max(n - 1, 2UL); m <= n; ++m)
        {
            const double lm_coeff_mm
                = coeff_data.sqrt_n(n + m - 1)*coeff_data.sqrt_n(n + m);
            const double lm_coeff_pm
                = coeff_data.sqrt_n(n - m + 1)*coeff_data.sqrt_n(n - m + 2);
            const double lm_coeff_pp
                = coeff_data.sqrt_n(n + m + 1)*coeff_data.sqrt_n(n + m + 2);

            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_mm1 = in_nm1_nm1[m - 1];
            const std::array<double, 2> in_np1_nm1_mm1 = in_np1_nm1[m - 1];
            const std::array<double, 2> in_np1_np1_mm1 = in_np1_np1[m - 1];
            const std::array<double, 2> in_np1_np1_mp1 = in_np1_np1[m + 1];

            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n_n_m[0]
                    = -nl_coeff_mm*lm_coeff_mm*in_nm1_nm1_mm1[0]
                    - nl_coeff_pm*lm_coeff_mm*in_np1_nm1_mm1[0]
                    - nl_coeff_pp*(
                        lm_coeff_pp*in_np1_np1_mp1[0] - lm_coeff_pm*in_np1_np1_mm1[0]);
                
                out_n_n_m[1]
                    = -nl_coeff_mm*lm_coeff_mm*in_nm1_nm1_mm1[1]
                    - nl_coeff_pm*lm_coeff_mm*in_np1_nm1_mm1[1]
                    - nl_coeff_pp*(
                        lm_coeff_pp*in_np1_np1_mp1[1] - lm_coeff_pm*in_np1_np1_mm1[1]);
            }
            else
            {
                out_n_n_m[0]
                    = nl_coeff_mm*lm_coeff_mm*in_nm1_nm1_mm1[1]
                    + nl_coeff_pm*lm_coeff_mm*in_np1_nm1_mm1[1]
                    - nl_coeff_pp*(
                        lm_coeff_pp*in_np1_np1_mp1[1] + lm_coeff_pm*in_np1_np1_mm1[1]);
                
                out_n_n_m[1]
                    = nl_coeff_pp*(
                        lm_coeff_pp*in_np1_np1_mp1[0] + lm_coeff_pm*in_np1_np1_mm1[0])
                    - nl_coeff_pm*lm_coeff_mm*in_np1_nm1_mm1[0]
                    - nl_coeff_mm*lm_coeff_mm*in_nm1_nm1_mm1[0];
            }
        }
    }

    for (std::size_t n = std::max(nmax - 1, 3UL); n <= nmax; ++n)
    {
        const double n_coeff_m = 0.5*coeff_data.inv_sqrt_2np1_2np3(n);

        auto out_n = out[n];
        auto in_nm1 = in[n - 1];
        const std::size_t n_parity = n & 1;
            
        // edge case: n = nmax - 1, l = 0
        // edge case: n = nmax - 1, l = 2
        // edge case: n = nmax, l = 0
        // edge case: n = nmax, l = 2
        if (n_parity == 0)
        {
            const double dn = double(n);
            constexpr double l_coeff = -2.0/sqrt3;
            constexpr std::size_t idx = std::size_t(coord_param == PlaneCoord::Y);
            out_n(0,0)[0] = l_coeff*n_coeff_m*dn*in_nm1(1,1)[idx];
            
            // l == 2 special case
            constexpr double l_coeff_m = 1.0/(sqrt3*sqrt5);
            constexpr double l_coeff_p = 1.0/(sqrt5*sqrt7);

            // l == 2 special case
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 3.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 2.0);
            
            constexpr double l0_coeff_m = 2.0;
            constexpr double l0_coeff_p = 2.0*sqrt2*sqrt3;
            constexpr double l1_coeff_mm = 2.0*sqrt3;
            constexpr double l1_coeff_pm = l1_coeff_mm;
            constexpr double l1_coeff_pp = 2.0*sqrt5;
            constexpr double l2_coeff_mm = 2.0*sqrt3;
            constexpr double l2_coeff_pm = sqrt2;
            constexpr double l2_coeff_pp = sqrt2*sqrt3*sqrt5;
            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n(2,0)[0] = nl_coeff_mm*l0_coeff_m*in_nm1(1,1)[0]
                        - nl_coeff_mp*l0_coeff_p*in_nm1(3,1)[0];
                
                out_n(2,1)[0]
                    = -nl_coeff_mm*l1_coeff_mm*in_nm1(1,0)[0]
                    - nl_coeff_mp*(
                        l1_coeff_pp*in_nm1(3,2)[0] - l1_coeff_pm*in_nm1(3,0)[0]);
                
                out_n(2,1)[1] = -nl_coeff_mp*l1_coeff_pp*in_nm1(3,2)[1];
                
                out_n(2,2)[0]
                    = -nl_coeff_mm*l2_coeff_mm*in_nm1(1,1)[0]
                    - nl_coeff_mp*(
                        l2_coeff_pp*in_nm1(3,3)[0] - l2_coeff_pm*in_nm1(3,1)[0]);
                
                out_n(2,2)[1]
                    = -nl_coeff_mm*l2_coeff_mm*in_nm1(1,1)[1]
                    - nl_coeff_mp*(
                        l2_coeff_pp*in_nm1(3,3)[1] - l2_coeff_pm*in_nm1(3,1)[1]);
            }
            else
            {
                out_n(2,0)[0] = nl_coeff_mm*l0_coeff_m*in_nm1(1,1)[1]
                        - nl_coeff_mp*l0_coeff_p*in_nm1(3,1)[1];
                
                out_n(2,1)[0] = -nl_coeff_mp*l1_coeff_pp*in_nm1(3,2)[1];
                
                out_n(2,1)[1]
                    = -nl_coeff_mm*l1_coeff_mm*in_nm1(1,0)[0]
                    + nl_coeff_mp*(
                        l1_coeff_pp*in_nm1(3,2)[0] + l1_coeff_pm*in_nm1(3,0)[0]);
                    
                out_n(2,2)[0]
                    = nl_coeff_mm*l2_coeff_mm*in_nm1(1,1)[1]
                    - nl_coeff_mp*(
                        l2_coeff_pp*in_nm1(3,3)[1] + l2_coeff_pm*in_nm1(3,1)[1]);
                
                out_n(2,2)[1]
                    = -nl_coeff_mm*l2_coeff_mm*in_nm1(1,1)[0]
                    + nl_coeff_mp*(
                        l2_coeff_pp*in_nm1(3,3)[0] + l2_coeff_pm*in_nm1(3,1)[0]);
            }
        }
        // edge case: n = nmax - 1, l = 1
        // edge case: n = nmax, l = 1
        else
        {
            // l == 1 special case
            constexpr double l_coeff_m = 1.0/(sqrt3);
            constexpr double l_coeff_p = 1.0/(sqrt3*sqrt5);

            // l == 1 special case
            const double dn = double(n);
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 2.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 1.0);

            constexpr double l1_coeff_mm = 2.0;
            constexpr double l1_coeff_pm = l1_coeff_mm;
            constexpr double l1_coeff_pp = 2.0*sqrt3;
            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n(1,0)[0] = (-2.0*sqrt3)*nl_coeff_mp*in_nm1(2,1)[0];

                out_n(1,1)[0]
                    = -nl_coeff_mm*l1_coeff_mm*in_nm1(0,0)[0]
                    - nl_coeff_mp*(
                        l1_coeff_pp*in_nm1(2,2)[0] - l1_coeff_pm*in_nm1(2,0)[0]);

                out_n(1,1)[1] = (-l1_coeff_pp)*nl_coeff_mp*in_nm1(2,2)[1];
            }
            else
            {
                out_n(1,0)[0] = (-2.0*sqrt3)*nl_coeff_mp*in_nm1(2,1)[1];

                out_n(1,1)[0] = (-l1_coeff_pp)*nl_coeff_mp*in_nm1(2,2)[1];
                
                out_n(1,1)[1]
                    = -nl_coeff_mm*l1_coeff_mm*in_nm1(0,0)[0]
                    + nl_coeff_mp*(
                        l1_coeff_pp*in_nm1(2,2)[0] + l1_coeff_pm*in_nm1(2,0)[0]);
            }
        }

        for (std::size_t l = 4 - n_parity; l < n; l += 2)
        {
            const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(l);
            const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(l);
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;

            const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*nml;

            auto out_n_l = out_n[l];
            auto in_nm1_lm1 = in_nm1[l - 1];
            auto in_nm1_lp1 = in_nm1[l + 1];

            // edge case: n = nmax - 1, l > 2, m = 0
            // edge case: n = nmax - 1, l > 2, m = 1
            // edge case: n = nmax, l > 2, m = 0
            // edge case: n = nmax, l > 2, m = 1
            const double l0_coeff_mp
                = sqrt2*coeff_data.sqrt_n(l - 1)*coeff_data.sqrt_n(l);
            const double l0_coeff_pp
                = sqrt2*coeff_data.sqrt_n(l + 1)*coeff_data.sqrt_n(l + 2);
            const double l1_coeff_mm
                = sqrt2*coeff_data.sqrt_n(l)*coeff_data.sqrt_n(l + 1);
            const double l1_coeff_pm = l1_coeff_mm;
            const double l1_coeff_mp
                = coeff_data.sqrt_n(l - 2)*coeff_data.sqrt_n(l - 1);
            const double l1_coeff_pp
                = coeff_data.sqrt_n(l + 2)*coeff_data.sqrt_n(l + 3);
            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n_l[0][0]
                    = nl_coeff_mm*l0_coeff_mp*in_nm1_lm1[1][0]
                    - nl_coeff_mp*l0_coeff_pp*in_nm1_lp1[1][0];
                
                out_n_l[1][0]
                    = nl_coeff_mm*(l1_coeff_mp*in_nm1_lm1[2][0] - l1_coeff_mm*in_nm1_lm1[0][0])
                    - nl_coeff_mp*(l1_coeff_pp*in_nm1_lp1[2][0] - l1_coeff_pm*in_nm1_lp1[0][0]);
                
                out_n_l[1][1]
                    = nl_coeff_mm*l1_coeff_mp*in_nm1_lm1[2][1]
                    - nl_coeff_mp*l1_coeff_pp*in_nm1_lp1[2][1];
            }
            else
            {
                out_n_l[0][0]
                    = nl_coeff_mm*l0_coeff_mp*in_nm1_lm1[1][1]
                    - nl_coeff_mp*l0_coeff_pp*in_nm1_lp1[1][1];
                
                out_n_l[1][0]
                    = -l1_coeff_pp*nl_coeff_mp*in_nm1_lp1[2][1]
                    + nl_coeff_mm*l1_coeff_mp*in_nm1_lm1[2][1];
                
                out_n_l[1][1]
                    = -nl_coeff_mm*(l1_coeff_mp*in_nm1_lm1[2][0] + l1_coeff_mm*in_nm1_lm1[0][0])
                    + nl_coeff_mp*(l1_coeff_pp*in_nm1_lp1[2][0] + l1_coeff_pm*in_nm1_lp1[0][0]);
            }

            for (std::size_t m = 2; m < l - 1; ++m)
            {
                const double lm_coeff_mm
                    = coeff_data.sqrt_n(l + m - 1)*coeff_data.sqrt_n(l + m);
                const double lm_coeff_mp
                    = coeff_data.sqrt_n(l - m - 1)*coeff_data.sqrt_n(l - m);
                const double lm_coeff_pm
                    = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l - m + 2);
                const double lm_coeff_pp
                    = coeff_data.sqrt_n(l + m + 1)*coeff_data.sqrt_n(l + m + 2);

                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_mm1 = in_nm1_lm1[m - 1];
                const std::array<double, 2> in_nm1_lm1_mp1 = in_nm1_lm1[m + 1];
                const std::array<double, 2> in_nm1_lp1_mm1 = in_nm1_lp1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mp1 = in_nm1_lp1[m + 1];

                // edge case: n = nmax - 1, l > 2, m > 1
                // edge case: n = nmax, l > 2, m > 1
                if constexpr (coord_param == PlaneCoord::X)
                {
                    out_n_l_m[0]
                        = nl_coeff_mm*(
                            lm_coeff_mp*in_nm1_lm1_mp1[0] - lm_coeff_mm*in_nm1_lm1_mm1[0])
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[0] - lm_coeff_pm*in_nm1_lp1_mm1[0]);
                    
                    out_n_l_m[1]
                        = nl_coeff_mm*(
                            lm_coeff_mp*in_nm1_lm1_mp1[1] - lm_coeff_mm*in_nm1_lm1_mm1[1])
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[1] - lm_coeff_pm*in_nm1_lp1_mm1[1]);
                }
                else
                {
                    out_n_l_m[0]
                        = nl_coeff_mm*(
                            lm_coeff_mp*in_nm1_lm1_mp1[1] + lm_coeff_mm*in_nm1_lm1_mm1[1])
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[1] + lm_coeff_pm*in_nm1_lp1_mm1[1]);
                    
                    out_n_l_m[1]
                        = nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[0] + lm_coeff_pm*in_nm1_lp1_mm1[0])
                        - nl_coeff_mm*(
                            lm_coeff_mp*in_nm1_lm1_mp1[0] + lm_coeff_mm*in_nm1_lm1_mm1[0]);
                }
            }

            // edge case: n = nmax - 1, l > 2, m = l - 1
            // edge case: n = nmax - 1, l > 2, m = l
            // edge case: n = nmax, l > 2, m = l - 1
            // edge case: n = nmax, l > 2, m = l
            for (std::size_t m = std::max(l - 1, 2UL); m <= l; ++m)
            {
                const double lm_coeff_mm
                    = coeff_data.sqrt_n(l + m - 1)*coeff_data.sqrt_n(l + m);
                const double lm_coeff_pm
                    = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l - m + 2);
                const double lm_coeff_pp
                    = coeff_data.sqrt_n(l + m + 1)*coeff_data.sqrt_n(l + m + 2);

                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_mm1 = in_nm1_lm1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mm1 = in_nm1_lp1[m - 1];
                const std::array<double, 2> in_nm1_lp1_mp1 = in_nm1_lp1[m + 1];

                if constexpr (coord_param == PlaneCoord::X)
                {
                    out_n_l_m[0]
                        = -nl_coeff_mm*lm_coeff_mm*in_nm1_lm1_mm1[0]
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[0] - lm_coeff_pm*in_nm1_lp1_mm1[0]);
                    
                    out_n_l_m[1]
                        = -nl_coeff_mm*lm_coeff_mm*in_nm1_lm1_mm1[1]
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[1] - lm_coeff_pm*in_nm1_lp1_mm1[1]);
                }
                else
                {
                    out_n_l_m[0]
                        = nl_coeff_mm*lm_coeff_mm*in_nm1_lm1_mm1[1]
                        - nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[1] + lm_coeff_pm*in_nm1_lp1_mm1[1]);
                    
                    out_n_l_m[1]
                        = nl_coeff_mp*(
                            lm_coeff_pp*in_nm1_lp1_mp1[0] + lm_coeff_pm*in_nm1_lp1_mm1[0])
                        - nl_coeff_mm*lm_coeff_mm*in_nm1_lm1_mm1[0];
                }
            }
        }

        const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(n);
        const double npl1 = double(2*n + 1);

        const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;

        auto out_n_n = out_n[n];
        auto in_nm1_nm1 = in_nm1[n - 1];

        // edge case: n = nmax - 1, l = n, m = 0
        // edge case: n = nmax - 1, l = n, m = 1
        // edge case: n = nmax, l = n, m = 0
        // edge case: n = nmax, l = n, m = 1
        const double l0_coeff_mp
            = sqrt2*coeff_data.sqrt_n(n - 1)*coeff_data.sqrt_n(n);
        const double l1_coeff_mm
            = sqrt2*coeff_data.sqrt_n(n)*coeff_data.sqrt_n(n + 1);
        const double l1_coeff_mp
            = coeff_data.sqrt_n(n - 2)*coeff_data.sqrt_n(n - 1);
        if constexpr (coord_param == PlaneCoord::X)
        {
            out_n_n[0][0] = nl_coeff_mm*l0_coeff_mp*in_nm1_nm1[1][0];
            
            out_n_n[1][0]
                = nl_coeff_mm*(l1_coeff_mp*in_nm1_nm1[2][0] - l1_coeff_mm*in_nm1_nm1[0][0]);
            
            out_n_n[1][1]
                = nl_coeff_mm*l1_coeff_mp*in_nm1_nm1[2][1];
        }
        else
        {
            out_n_n[0][0]
                = nl_coeff_mm*l0_coeff_mp*in_nm1_nm1[1][1];
            
            out_n_n[1][0]
                = nl_coeff_mm*l1_coeff_mp*in_nm1_nm1[2][1];
            
            out_n_n[1][1]
                = -nl_coeff_mm*(l1_coeff_mp*in_nm1_nm1[2][0] + l1_coeff_mm*in_nm1_nm1[0][0]);
        }
        
        for (std::size_t m = 2; m < n - 1; ++m)
        {
            const double lm_coeff_mm
                = coeff_data.sqrt_n(n + m - 1)*coeff_data.sqrt_n(n + m);
            const double lm_coeff_mp
                = coeff_data.sqrt_n(n - m - 1)*coeff_data.sqrt_n(n - m);

            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_mm1 = in_nm1_nm1[m - 1];
            const std::array<double, 2> in_nm1_nm1_mp1 = in_nm1_nm1[m + 1];

            // edge case: n = nmax - 1, l = n, m > 1
            // edge case: n > nmax, l = n, m > 1
            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n_n_m[0]
                    = nl_coeff_mm*(
                        lm_coeff_mp*in_nm1_nm1_mp1[0] - lm_coeff_mm*in_nm1_nm1_mm1[0]);
                
                out_n_n_m[1]
                    = nl_coeff_mm*(
                        lm_coeff_mp*in_nm1_nm1_mp1[1] - lm_coeff_mm*in_nm1_nm1_mm1[1]);
            }
            else
            {
                out_n_n_m[0]
                    = nl_coeff_mm*(
                        lm_coeff_mp*in_nm1_nm1_mp1[1] + lm_coeff_mm*in_nm1_nm1_mm1[1]);
                
                out_n_n_m[1]
                    = -nl_coeff_mm*(
                        lm_coeff_mp*in_nm1_nm1_mp1[0] + lm_coeff_mm*in_nm1_nm1_mm1[0]);
            }
        }

        // edge case: n = nmax - 1, l = n, m = l - 1
        // edge case: n = nmax - 1, l = n, m = l
        // edge case: n = nmax, l = n, m = l - 1
        // edge case: n = nmax, l = n, m = l
        for (std::size_t m = std::max(n - 1, 2UL); m <= n; ++m)
        {
            const double lm_coeff_mm
                = coeff_data.sqrt_n(n + m - 1)*coeff_data.sqrt_n(n + m);

            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_mm1 = in_nm1_nm1[m - 1];

            if constexpr (coord_param == PlaneCoord::X)
            {
                out_n_n_m[0] = -nl_coeff_mm*lm_coeff_mm*in_nm1_nm1_mm1[0];
                out_n_n_m[1] = -nl_coeff_mm*lm_coeff_mm*in_nm1_nm1_mm1[1];
            }
            else
            {
                out_n_n_m[0] = nl_coeff_mm*lm_coeff_mm*in_nm1_nm1_mm1[1];
                out_n_n_m[1] = -nl_coeff_mm*lm_coeff_mm*in_nm1_nm1_mm1[0];
            }
        }
    }
}

void multiply_by_x(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    multiply_by_x_y_impl<PlaneCoord::X>(coeff_data, in, out);
}

void multiply_by_y(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    multiply_by_x_y_impl<PlaneCoord::Y>(coeff_data, in, out);
}

/**
    @brief Compute coefficients of Zernike expansion multiplied by `z`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() < out.order() <= coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_z_impl(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    /*
    The base case is that `out(n,l,m)` is a linear combination of the coefficients
    ```
    in(n + 1, l + 1, m)
    in(n + 1, l - 1, m)
    in(n - 1, l + 1, m)
    in(n - 1, l - 1, m)
    ```
    However, for `in(n,l,m)`, we have
    ```
    abs(m) <= l <= n < in.order()
    ```
    These conditions on the indices lead to multiple edge cases where different coefficients are neglected.
    */

    constexpr double sqrt2 = std::numbers::sqrt2;
    constexpr double sqrt3 = std::numbers::sqrt3;
    constexpr double sqrt5 = 2.2360679774997896964091737;
    constexpr double sqrt7 = 2.6457513110645905905016158;

    assert(in.order() < out.order());
    assert(out.order() <= coeff_data.order());

    const std::size_t nmax = in.order();
    if (nmax == 0) return;

    std::ranges::fill(out.flatten(), std::array<double, 2>{});
    
    out(1,1,0)[0] = (1.0/sqrt5)*in(0,0,0)[0];
    if (nmax == 1) return;

    out(0,0,0)[0] = (1.0/sqrt5)*in(1,1,0)[0];
    out(2,0,0)[0] = (2.0/(sqrt3*sqrt5*sqrt7))*in(1,1,0)[0];
    out(2,2,0)[0] = (2.0/(sqrt3*sqrt7))*in(1,1,0)[0];

    out(2,2,1)[0] = (1.0/sqrt7)*in(1,1,1)[0];
    out(2,2,1)[1] = (1.0/sqrt7)*in(1,1,1)[1];
    if (nmax == 2) return;
    
    out(1,1,0)[0] += (2.0/(sqrt3*sqrt5*sqrt7))*in(2,0,0)[0]
            + (2.0/(sqrt3*sqrt7))*in(2,2,0)[0];

    out(1,1,1)[0] = (1.0/sqrt7)*in(2,2,1)[0];
    out(1,1,1)[1] = (1.0/sqrt7)*in(2,2,1)[1];
        
    if (nmax > 3)
    {
        out(2,0,0)[0] += (5.0/(3.0*sqrt3*sqrt7))*in(3,1,0)[0];
        out(2,2,0)[0] += (4.0/(3.0*sqrt3*sqrt5*sqrt7))*in(3,1,0)[0]
                + (1.0/sqrt5)*in(3,3,0)[0];
        
        out(2,2,1)[0] += (2.0/(3.0*sqrt5*sqrt7))*in(3,1,1)[0]
                + (2.0*sqrt2/(3.0*sqrt5))*in(3,3,1)[0];
        out(2,2,1)[1] += (2.0/(3.0*sqrt5*sqrt7))*in(3,1,1)[1]
                + (2.0*sqrt2/(3.0*sqrt5))*in(3,3,1)[1];
        
        out(2,2,2)[0] += (1.0/3.0)*in(3,3,2)[0];
        out(2,2,2)[1] += (1.0/3.0)*in(3,3,2)[1];
    }

    for (std::size_t n = 3; n < nmax - 1; ++n)
    {
        const double n_coeff_m = coeff_data.inv_sqrt_2np1_2np3(n);
        const double n_coeff_p = coeff_data.inv_sqrt_2np3_2np5(n);

        auto out_n = out[n];
        auto in_nm1 = in[n - 1];
        auto in_np1 = in[n + 1];

        const std::size_t parity = n & 1;

        // edge case: n > 2, l = 0
        // edge case: n > 2, l = 2
        if (!parity)
        {
            out_n(0,0)[0] = (1.0/sqrt3)*(
                    n_coeff_m*double(n)*in_nm1(1,0)[0]
                    + n_coeff_p*double(n + 3)*in_np1(1,0)[0]);
            
            // l == 2 special case
            constexpr double l_coeff_m = 1.0/(sqrt3*sqrt5);
            constexpr double l_coeff_p = 1.0/(sqrt5*sqrt7);

            // l == 2 special case
            const double dn = double(n);
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 3.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 2.0);
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*dn;
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*(dn + 5.0);

            out_n(2,0)[0]
                = nl_coeff_mm*2.0*in_nm1(1,0)[0]
                + nl_coeff_pm*2.0*in_np1(1,0)[0]
                + nl_coeff_mp*3.0*in_nm1(3,0)[0]
                + nl_coeff_pp*3.0*in_np1(3,0)[0];
            
            out_n(2,1)[0]
                = nl_coeff_mm*sqrt3*in_nm1(1,1)[0]
                + nl_coeff_pm*sqrt3*in_np1(1,1)[0]
                + nl_coeff_mp*(2.0*sqrt2)*in_nm1(3,1)[0]
                + nl_coeff_pp*(2.0*sqrt2)*in_np1(3,1)[0];
            
            out_n(2,1)[1]
                = nl_coeff_mm*sqrt3*in_nm1(1,1)[1]
                + nl_coeff_pm*sqrt3*in_np1(1,1)[1]
                + nl_coeff_mp*(2.0*sqrt2)*in_nm1(3,1)[1]
                + nl_coeff_pp*(2.0*sqrt2)*in_np1(3,1)[1];
            
            out_n(2,2)[0] = sqrt5*(
                    nl_coeff_mp*in_nm1(3,2)[0] + nl_coeff_pp*in_np1(3,2)[0]);
            out_n(2,2)[1] = sqrt5*(
                    nl_coeff_mp*in_nm1(3,2)[1] + nl_coeff_pp*in_np1(3,2)[1]);
        }
        // edge case: n > 2, l = 1
        else
        {
            // l == 1 special case
            constexpr double l_coeff_m = 1.0/sqrt3;
            constexpr double l_coeff_p = 1.0/(sqrt3*sqrt5);

            // l == 1 special case
            const double dn = double(n);
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 2.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 1.0);
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*(dn + 1.0);
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*(dn + 4.0);

            out_n(1,0)[0]
                = nl_coeff_mm*in_nm1(0,0)[0] + nl_coeff_pm*in_np1(0,0)[0]
                + nl_coeff_mp*2.0*in_nm1(2,0)[0] + nl_coeff_pp*2.0*in_np1(2,0)[0];
            
            out_n(1,1)[0] = sqrt3*(nl_coeff_mp*in_nm1(2,1)[0] + nl_coeff_pp*in_np1(2,1)[0]);
            out_n(1,1)[1] = sqrt3*(nl_coeff_mp*in_nm1(2,1)[1] + nl_coeff_pp*in_np1(2,1)[1]);
        }

        for (std::size_t l = 4 - parity; l < n; l += 2)
        {
            const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(l);
            const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(l);
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;
            const double nml2 = nml + 2.0;
            const double npl3 = npl + 3.0;

            const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*nml;
            const double nl_coeff_pm = n_coeff_p*l_coeff_m*nml2;
            const double nl_coeff_pp = n_coeff_p*l_coeff_p*npl3;

            auto out_n_l = out_n[l];
            auto in_nm1_lm1 = in_nm1[l - 1];
            auto in_nm1_lp1 = in_nm1[l + 1];
            auto in_np1_lm1 = in_np1[l - 1];
            auto in_np1_lp1 = in_np1[l + 1];

            // base case
            for (std::size_t m = 0; m < l; ++m)
            {
                const double lm_coeff_m
                    = coeff_data.sqrt_n(l - m)*coeff_data.sqrt_n(l + m);
                const double lm_coeff_p
                    = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l + m + 1);

                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_m = in_nm1_lm1[m];
                const std::array<double, 2> in_nm1_lp1_m = in_nm1_lp1[m];
                const std::array<double, 2> in_np1_lm1_m = in_np1_lm1[m];
                const std::array<double, 2> in_np1_lp1_m = in_np1_lp1[m];

                out_n_l_m[0]
                    = nl_coeff_mm*lm_coeff_m*in_nm1_lm1_m[0]
                    + nl_coeff_mp*lm_coeff_p*in_nm1_lp1_m[0]
                    + nl_coeff_pm*lm_coeff_m*in_np1_lm1_m[0]
                    + nl_coeff_pp*lm_coeff_p*in_np1_lp1_m[0];

                out_n_l_m[1]
                    = nl_coeff_mm*lm_coeff_m*in_nm1_lm1_m[1]
                    + nl_coeff_mp*lm_coeff_p*in_nm1_lp1_m[1]
                    + nl_coeff_pm*lm_coeff_m*in_np1_lm1_m[1]
                    + nl_coeff_pp*lm_coeff_p*in_np1_lp1_m[1];
            }

            std::array<double, 2>& out_n_l_l = out_n_l[l];
            const std::array<double, 2> in_nm1_lp1_l = in_nm1_lp1[l];
            const std::array<double, 2> in_np1_lp1_l = in_np1_lp1[l];

            const double ll_coeff_p = coeff_data.sqrt_n(2*l + 1);

            // edge case: n > 2, l > 2, m = l
            out_n_l_l[0] = ll_coeff_p*(
                    nl_coeff_mp*in_nm1_lp1_l[0] + nl_coeff_pp*in_np1_lp1_l[0]);
            out_n_l_l[1] = ll_coeff_p*(
                    nl_coeff_mp*in_nm1_lp1_l[1] + nl_coeff_pp*in_np1_lp1_l[1]);
        }

        const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(n);
        const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(n);
        const double npl1 = double(2*n + 1);
        const double npl3 = double(2*n + 3);

        const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
        const double nl_coeff_pm = n_coeff_p*l_coeff_m*2.0;
        const double nl_coeff_pp = n_coeff_p*l_coeff_p*npl3;

        auto out_n_n = out_n[n];
        auto in_nm1_nm1 = in_nm1[n - 1];
        auto in_np1_nm1 = in_np1[n - 1];
        auto in_np1_np1 = in_np1[n + 1];

        // edge case: n > 2, l = n
        for (std::size_t m = 0; m < n; ++m)
        {
            const double lm_coeff_m
                = coeff_data.sqrt_n(n - m)*coeff_data.sqrt_n(n + m);
            const double lm_coeff_p
                = coeff_data.sqrt_n(n - m + 1)*coeff_data.sqrt_n(n + m + 1);

            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_m = in_nm1_nm1[m];
            const std::array<double, 2> in_np1_nm1_m = in_np1_nm1[m];
            const std::array<double, 2> in_np1_np1_m = in_np1_np1[m];

            out_n_n_m[0]
                = nl_coeff_mm*lm_coeff_m*in_nm1_nm1_m[0]
                + nl_coeff_pm*lm_coeff_m*in_np1_nm1_m[0]
                + nl_coeff_pp*lm_coeff_p*in_np1_np1_m[0];

            out_n_n_m[1]
                = nl_coeff_mm*lm_coeff_m*in_nm1_nm1_m[1]
                + nl_coeff_pm*lm_coeff_m*in_np1_nm1_m[1]
                + nl_coeff_pp*lm_coeff_p*in_np1_np1_m[1];
        }

        std::array<double, 2>& out_n_n_n = out_n_n[n];
        const std::array<double, 2> in_np1_np1_n = in_np1_np1[n];

        const double ll_coeff_p = coeff_data.sqrt_n(2*n + 1);

        // edge case: n > 0, l = n, m = l
        out_n_n_n[0] = ll_coeff_p*nl_coeff_pp*in_np1_np1_n[0];
        out_n_n_n[1] = ll_coeff_p*nl_coeff_pp*in_np1_np1_n[1];
    }

    for (std::size_t n = std::max(nmax - 1, 3UL); n <= nmax; ++n)
    {
        const double n_coeff_m = coeff_data.inv_sqrt_2np1_2np3(n);

        auto out_n = out[n];
        auto in_nm1 = in[n - 1];
        const std::size_t parity = n & 1;

        // edge case: n = nmax - 1, l = 0
        // edge case: n = nmax - 1, l = 2
        // edge case: n = nmax, l = 0
        // edge case: n = nmax, l = 2
        if (!parity)
        {
            out_n(0,0)[0] = (1.0/sqrt3)*n_coeff_m*double(n)*in_nm1(1,0)[0];
            
            // l == 2 special case
            constexpr double l_coeff_m = 1.0/(sqrt3*sqrt5);
            constexpr double l_coeff_p = 1.0/(sqrt5*sqrt7);

            // l == 2 special case
            const double dn = double(n);
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 3.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 2.0);

            out_n(2,0)[0]
                = nl_coeff_mm*2.0*in_nm1(1,0)[0]
                + nl_coeff_mp*3.0*in_nm1(3,0)[0];
            
            out_n(2,1)[0]
                = nl_coeff_mm*sqrt3*in_nm1(1,1)[0]
                + nl_coeff_mp*(2.0*sqrt2)*in_nm1(3,1)[0];
            
            out_n(2,1)[1]
                = nl_coeff_mm*sqrt3*in_nm1(1,1)[1]
                + nl_coeff_mp*(2.0*sqrt2)*in_nm1(3,1)[1];
            
            out_n(2,2)[0] = sqrt5*nl_coeff_mp*in_nm1(3,2)[0];
            out_n(2,2)[1] = sqrt5*nl_coeff_mp*in_nm1(3,2)[1];
        }
        // edge case: n > 2, l = 1
        else
        {
            // l == 1 special case
            constexpr double l_coeff_m = 1.0/sqrt3;
            constexpr double l_coeff_p = 1.0/(sqrt3*sqrt5);

            // l == 1 special case
            const double dn = double(n);
            const double nl_coeff_mm = n_coeff_m*l_coeff_m*(dn + 2.0);
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*(dn - 1.0);

            out_n(1,0)[0] = nl_coeff_mm*in_nm1(0,0)[0] + nl_coeff_mp*2.0*in_nm1(2,0)[0];
            
            out_n(1,1)[0] = sqrt3*nl_coeff_mp*in_nm1(2,1)[0];
            out_n(1,1)[1] = sqrt3*nl_coeff_mp*in_nm1(2,1)[1];
        }

        for (std::size_t l = 4 - parity; l < n; l += 2)
        {
            const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(l);
            const double l_coeff_p = coeff_data.inv_sqrt_2np1_2np3(l);
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;

            const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;
            const double nl_coeff_mp = n_coeff_m*l_coeff_p*nml;

            auto out_n_l = out_n[l];
            auto in_nm1_lm1 = in_nm1[l - 1];
            auto in_nm1_lp1 = in_nm1[l + 1];

            // edge case: n = nmax - 1, l > 2
            // edge case: n = nmax, l > 2
            for (std::size_t m = 0; m < l; ++m)
            {
                const double lm_coeff_m
                    = coeff_data.sqrt_n(l - m)*coeff_data.sqrt_n(l + m);
                const double lm_coeff_p
                    = coeff_data.sqrt_n(l - m + 1)*coeff_data.sqrt_n(l + m + 1);

                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm1_lm1_m = in_nm1_lm1[m];
                const std::array<double, 2> in_nm1_lp1_m = in_nm1_lp1[m];

                out_n_l_m[0]
                    = nl_coeff_mm*lm_coeff_m*in_nm1_lm1_m[0]
                    + nl_coeff_mp*lm_coeff_p*in_nm1_lp1_m[0];

                out_n_l_m[1]
                    = nl_coeff_mm*lm_coeff_m*in_nm1_lm1_m[1]
                    + nl_coeff_mp*lm_coeff_p*in_nm1_lp1_m[1];
            }

            std::array<double, 2>& out_n_l_l = out_n_l[l];
            const std::array<double, 2> in_nm1_lp1_l = in_nm1_lp1[l];

            const double ll_coeff_p = coeff_data.sqrt_n(2*l + 1);

            // edge case: n = nmax - 1, l > 2, m = l
            // edge case: n = nmax, l > 2, m = l
            out_n_l_l[0] = ll_coeff_p*nl_coeff_mp*in_nm1_lp1_l[0];
            out_n_l_l[1] = ll_coeff_p*nl_coeff_mp*in_nm1_lp1_l[1];
        }

        const double l_coeff_m = coeff_data.inv_sqrt_2nm1_2np1(n);
        const double npl1 = double(2*n + 1);

        const double nl_coeff_mm = n_coeff_m*l_coeff_m*npl1;

        auto out_n_n = out_n[n];
        auto in_nm1_nm1 = in_nm1[n - 1];

        // edge case: n = nmax - 1, l = n
        // edge case: n = nmax, l = n
        for (std::size_t m = 0; m < n; ++m)
        {
            const double lm_coeff_m
                = coeff_data.sqrt_n(n - m)*coeff_data.sqrt_n(n + m);

            std::array<double, 2>& out_n_n_m = out_n_n[m];
            const std::array<double, 2> in_nm1_nm1_m = in_nm1_nm1[m];

            out_n_n_m[0] = nl_coeff_mm*lm_coeff_m*in_nm1_nm1_m[0];
            out_n_n_m[1] = nl_coeff_mm*lm_coeff_m*in_nm1_nm1_m[1];
        }
    }
}

void multiply_by_z(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    multiply_by_z_impl(coeff_data, in, out);
}

/**
    @brief Compute coefficients of Zernike expansion multiplied by `r*r`.

    @param coeff_data precomputed data to speed up computation
    @param in input expansion
    @param out output expansion

    @note This function expects `in.order() + 1 < out.order() <= coeff.order()`. If `in.order() == 0` no work is done.
*/
void multiply_by_r2_impl(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    /*
    The base case is that `out(n,l,m)` is a linear combination of the coefficients
    ```
    in(n + 2, l, m)
    in(n, l, m)
    in(n - 2, l, m)
    ```
    However, for `in(n,l,m)`, we have
    ```
    abs(m) <= l <= n < in.order()
    ```
    These conditions on the indices lead to multiple edge cases where different coefficients are neglected.
    */
    constexpr double sqrt3 = std::numbers::sqrt3;
    constexpr double sqrt5 = 2.2360679774997896964091737;
    constexpr double sqrt7 = 2.6457513110645905905016158;
    constexpr double sqrt11 = 3.316624790355399849114933;

    assert(in.order() + 1 < out.order());
    assert(out.order() <= coeff_data.order());

    const std::size_t nmax = in.order() + 1;
    if (nmax == 1) return;

    std::ranges::fill(out.flatten(), std::array<double, 2>{});

    out(0,0,0)[0] = (3.0/5.0)*in(0,0,0)[0];
    out(2,0,0)[0] = (2.0*sqrt3/(5.0*sqrt7))*in(0,0,0)[0];
    if (nmax == 2) return;

    out(1,1,0)[0] = (5.0/7.0)*in(1,1,0)[0];
    out(1,1,1)[0] = (5.0/7.0)*in(1,1,1)[0];
    out(1,1,1)[1] = (5.0/7.0)*in(1,1,1)[1];

    if (nmax > 3)
    {
        out(0,0,0)[0] += (2.0*sqrt3/(5.0*sqrt7))*in(2,0,0)[0];
        out(2,0,0)[0] += (23.0/45.0)*in(2,0,0)[0];
        out(2,2,0)[0] = (7.0/9.0)*in(2,2,0)[0];
        out(2,2,1)[0] = (7.0/9.0)*in(2,2,1)[0];
        out(2,2,1)[1] = (7.0/9.0)*in(2,2,1)[1];
        out(2,2,2)[0] = (7.0/9.0)*in(2,2,2)[0];
        out(2,2,2)[1] = (7.0/9.0)*in(2,2,2)[1];
    }

    if (nmax > 4)
    {
        out(1,1,0)[0] += (10.0/(21.0*sqrt5))*in(3,1,0)[0];
        out(1,1,1)[0] += (10.0/(21.0*sqrt5))*in(3,1,1)[0];
        out(1,1,1)[1] += (10.0/(21.0*sqrt5))*in(3,1,1)[1];
    }

    if (nmax > 5)
    {
        out(2,0,0)[0] += (20.0/(9.0*sqrt7*sqrt11))*in(4,0,0)[0];
        out(2,2,0)[0] += (2.0*sqrt7/(9.0*sqrt11))*in(4,2,0)[0];
        out(2,2,1)[0] += (2.0*sqrt7/(9.0*sqrt11))*in(4,2,1)[0];
        out(2,2,1)[1] += (2.0*sqrt7/(9.0*sqrt11))*in(4,2,1)[1];
        out(2,2,2)[0] += (2.0*sqrt7/(9.0*sqrt11))*in(4,2,2)[0];
        out(2,2,2)[1] += (2.0*sqrt7/(9.0*sqrt11))*in(4,2,2)[1];
    }

    for (std::size_t n = 3; n < nmax - 3; ++n)
    {
        const double d_2n = double(2*n);
        const double d_2np1 = d_2n + 1.0;
        const double d_2np5 = d_2n + 5.0;
        const double _2np1_2np5 = d_2np1*d_2np5;

        const double n_coeff_m
            = coeff_data.inv_sqrt_2nm1_2np1(n)*coeff_data.inv_sqrt_2np1_2np3(n);
        const double n_coeff_mid = 0.5/_2np1_2np5;
        const double n_coeff_p
            = coeff_data.inv_sqrt_2np3_2np5(n)*coeff_data.inv_sqrt_2np5_2np7(n);
        
        std::size_t parity = n & 1;

        auto out_n = out[n];
        auto in_nm2 = in[n - 2];
        auto in_n = in[n];
        auto in_np2 = in[n + 2];
        for (std::size_t l = parity; l < n - 2; l += 2)
        {
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double nml2 = nml + 2.0;
            const double npl1 = npl + 1.0;
            const double npl3 = npl + 3.0;
            const double _2lp1 = double(2*l) + 1.0;

            const double nl_coeff_m = nml*npl1*n_coeff_m;
            const double nl_coeff_mid = (_2lp1*_2lp1 + _2np1_2np5)*n_coeff_mid;
            const double nl_coeff_p = nml2*npl3*n_coeff_p;

            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            auto in_n_l = in_n[l];
            auto in_np2_l = in_np2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm2_l_m = in_nm2_l[m];
                const std::array<double, 2> in_n_l_m = in_n_l[m];
                const std::array<double, 2> in_np2_l_m = in_np2_l[m];

                out_n_l_m[0] = nl_coeff_m*in_nm2_l_m[0] + nl_coeff_mid*in_n_l_m[0]
                        + nl_coeff_p*in_np2_l_m[0];
                out_n_l_m[1] = nl_coeff_m*in_nm2_l_m[1] + nl_coeff_mid*in_n_l_m[1]
                        + nl_coeff_p*in_np2_l_m[1];
            }
        }

        // edge case: l = n - 2;
        {
            const double _2lp1 = d_2n - 3.0;

            const double nl_coeff_m
                = 2.0*(d_2n - 1.0)*coeff_data.inv_sqrt_2nm1_2np1(n)*coeff_data.inv_sqrt_2np1_2np3(n);
            const double nl_coeff_mid = (_2lp1*_2lp1 + _2np1_2np5)*n_coeff_mid;
            const double nl_coeff_p = 4.0*d_2np1*n_coeff_p;

            auto out_n_nm2 = out_n[n - 2];
            auto in_nm2_nm2 = in_nm2[n - 2];
            auto in_n_nm2 = in_n[n - 2];
            auto in_np2_nm2 = in_np2[n - 2];
            for (std::size_t m = 0; m <= n - 2; ++m)
            {
                std::array<double, 2>& out_n_nm2_m = out_n_nm2[m];
                const std::array<double, 2> in_nm2_nm2_m = in_nm2_nm2[m];
                const std::array<double, 2> in_n_nm2_m = in_n_nm2[m];
                const std::array<double, 2> in_np2_nm2_m = in_np2_nm2[m];

                out_n_nm2_m[0] = nl_coeff_m*in_nm2_nm2_m[0] + nl_coeff_mid*in_n_nm2_m[0]
                        + nl_coeff_p*in_np2_nm2_m[0];
                out_n_nm2_m[1] = nl_coeff_m*in_nm2_nm2_m[1] + nl_coeff_mid*in_n_nm2_m[1]
                        + nl_coeff_p*in_np2_nm2_m[1];
            }
        }

        // edge case: l = n;
        {
            const double d_2np3 = d_2n + 3.0;
            const double nl_coeff_mid = d_2np3/d_2np5;
            const double nl_coeff_p = 2.0*d_2np3*n_coeff_p;

            auto out_n_n = out_n[n];
            auto in_n_n = in_n[n];
            auto in_np2_n = in_np2[n];
            for (std::size_t m = 0; m <= n; ++m)
            {
                std::array<double, 2>& out_n_n_m = out_n_n[m];
                const std::array<double, 2> in_n_n_m = in_n_n[m];
                const std::array<double, 2> in_np2_n_m = in_np2_n[m];

                out_n_n_m[0] = nl_coeff_mid*in_n_n_m[0] + nl_coeff_p*in_np2_n_m[0];
                out_n_n_m[1] = nl_coeff_mid*in_n_n_m[1] + nl_coeff_p*in_np2_n_m[1];
            }
        }
    }

    for (std::size_t n = std::max(nmax - 3, 3UL); n < nmax - 1; ++n)
    {
        const double d_2n = double(2*n);
        const double d_2np5 = d_2n + 5.0;
        const double _2np1_2np5 = (d_2n + 1.0)*d_2np5;

        const double n_coeff_m
            = coeff_data.inv_sqrt_2nm1_2np1(n)*coeff_data.inv_sqrt_2np1_2np3(n);
        const double n_coeff_mid = 0.5/_2np1_2np5;

        std::size_t parity = n & 1;

        auto out_n = out[n];
        auto in_nm2 = in[n - 2];
        auto in_n = in[n];
        for (std::size_t l = parity; l < n - 2; l += 2)
        {
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;
            const double _2lp1 = double(2*l) + 1.0;

            const double nl_coeff_m = nml*npl1*n_coeff_m;
            const double nl_coeff_mid = (_2lp1*_2lp1 + _2np1_2np5)*n_coeff_mid;

            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            auto in_n_l = in_n[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm2_l_m = in_nm2_l[m];
                const std::array<double, 2> in_n_l_m = in_n_l[m];

                out_n_l_m[0] = nl_coeff_m*in_nm2_l_m[0] + nl_coeff_mid*in_n_l_m[0];
                out_n_l_m[1] = nl_coeff_m*in_nm2_l_m[1] + nl_coeff_mid*in_n_l_m[1];
            }
        }

        // edge case: l = n - 2;
        {
            const double _2lp1 = d_2n - 3.0;

            const double nl_coeff_m
                = 2.0*(d_2n - 1.0)*coeff_data.inv_sqrt_2nm1_2np1(n)*coeff_data.inv_sqrt_2np1_2np3(n);
            const double nl_coeff_mid = (_2lp1*_2lp1 + _2np1_2np5)*n_coeff_mid;

            auto out_n_nm2 = out_n[n - 2];
            auto in_nm2_nm2 = in_nm2[n - 2];
            auto in_n_nm2 = in_n[n - 2];
            for (std::size_t m = 0; m <= n - 2; ++m)
            {
                std::array<double, 2>& out_n_nm2_m = out_n_nm2[m];
                const std::array<double, 2> in_nm2_nm2_m = in_nm2_nm2[m];
                const std::array<double, 2> in_n_nm2_m = in_n_nm2[m];

                out_n_nm2_m[0] = nl_coeff_m*in_nm2_nm2_m[0] + nl_coeff_mid*in_n_nm2_m[0];
                out_n_nm2_m[1] = nl_coeff_m*in_nm2_nm2_m[1] + nl_coeff_mid*in_n_nm2_m[1];
            }
        }

        // edge case: l = n;
        {
            const double nl_coeff_mid = (d_2n + 3.0)/d_2np5;

            auto out_n_n = out_n[n];
            auto in_n_n = in_n[n];
            for (std::size_t m = 0; m <= n; ++m)
            {
                std::array<double, 2>& out_n_n_m = out_n_n[m];
                const std::array<double, 2> in_n_n_m = in_n_n[m];

                out_n_n_m[0] = nl_coeff_mid*in_n_n_m[0];
                out_n_n_m[1] = nl_coeff_mid*in_n_n_m[1];
            }
        }
    }

    for (std::size_t n = nmax - 1; n <= nmax; ++n)
    {
        const double d_2n = double(2*n);

        const double n_coeff_m
            = coeff_data.inv_sqrt_2nm1_2np1(n)*coeff_data.inv_sqrt_2np1_2np3(n);

        std::size_t parity = n & 1;

        auto out_n = out[n];
        auto in_nm2 = in[n - 2];
        for (std::size_t l = parity; l < n - 2; l += 2)
        {
            const double nml = double(n - l);
            const double npl = double(n + l);
            const double npl1 = npl + 1.0;

            const double nl_coeff_m = nml*npl1*n_coeff_m;

            auto out_n_l = out_n[l];
            auto in_nm2_l = in_nm2[l];
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::array<double, 2>& out_n_l_m = out_n_l[m];
                const std::array<double, 2> in_nm2_l_m = in_nm2_l[m];

                out_n_l_m[0] = nl_coeff_m*in_nm2_l_m[0];
                out_n_l_m[1] = nl_coeff_m*in_nm2_l_m[1];
            }
        }

        // edge case: l = n - 2;
        {
            const double nl_coeff_m
                = 2.0*(d_2n - 1.0)*coeff_data.inv_sqrt_2nm1_2np1(n)*coeff_data.inv_sqrt_2np1_2np3(n);

            auto out_n_nm2 = out_n[n - 2];
            auto in_nm2_nm2 = in_nm2[n - 2];
            for (std::size_t m = 0; m <= n - 2; ++m)
            {
                std::array<double, 2>& out_n_nm2_m = out_n_nm2[m];
                const std::array<double, 2> in_nm2_nm2_m = in_nm2_nm2[m];

                out_n_nm2_m[0] = nl_coeff_m*in_nm2_nm2_m[0];
                out_n_nm2_m[1] = nl_coeff_m*in_nm2_nm2_m[1];
            }
        }
    }
}

void multiply_by_r2(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    multiply_by_r2_impl(coeff_data, in, out);
}

void multiply_by_x_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    assert(in.order() + 2 < out.order());
    multiply_by_x_y_impl<PlaneCoord::X>(coeff_data, in, out);
    zebra::radon_transform_inplace(out);
}

void multiply_by_y_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    assert(in.order() + 2 < out.order());
    multiply_by_x_y_impl<PlaneCoord::Y>(coeff_data, in, out);
    zebra::radon_transform_inplace(out);
}

void multiply_by_z_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    assert(in.order() + 2 < out.order());
    multiply_by_z_impl(coeff_data, in, out);
    zebra::radon_transform_inplace(out);
}

void multiply_by_r2_and_radon_transform_inplace(
    const ZernikeRecursionData& coeff_data,
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out) noexcept
{
    assert(in.order() + 3 < out.order());
    multiply_by_r2_impl(coeff_data, in, out);
    zebra::radon_transform_inplace(out);
}

ZernikeCoordinateMultiplier::ZernikeCoordinateMultiplier(std::size_t order):
    m_coeff_data(order) {}

void ZernikeCoordinateMultiplier::expand(std::size_t order)
{
    m_coeff_data.expand(order);
}

void ZernikeCoordinateMultiplier::multiply_by_x(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out
) const noexcept
{
    multiply_by_x_y_impl<PlaneCoord::X>(m_coeff_data, in, out);
}

void ZernikeCoordinateMultiplier::multiply_by_y(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out
) const noexcept
{
    multiply_by_x_y_impl<PlaneCoord::X>(m_coeff_data, in, out);
}

void ZernikeCoordinateMultiplier::multiply_by_z(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out
) const noexcept
{
    multiply_by_z_impl(m_coeff_data, in, out);
}

void ZernikeCoordinateMultiplier::multiply_by_r2(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out
) const noexcept
{
    multiply_by_r2_impl(m_coeff_data, in, out);
}

void
ZernikeCoordinateMultiplier::multiply_by_x_and_radon_transform_inplace(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out
) const noexcept
{
    assert(in.order() + 2 < out.order());
    multiply_by_x_y_impl<PlaneCoord::X>(m_coeff_data, in, out);
    zebra::radon_transform_inplace(out);
}

void
ZernikeCoordinateMultiplier::multiply_by_y_and_radon_transform_inplace(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out
) const noexcept
{
    assert(in.order() + 2 < out.order());
    multiply_by_x_y_impl<PlaneCoord::Y>(m_coeff_data, in, out);
    zebra::radon_transform_inplace(out);
}

void
ZernikeCoordinateMultiplier::multiply_by_z_and_radon_transform_inplace(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out
) const noexcept
{
    assert(in.order() + 2 < out.order());
    multiply_by_z_impl(m_coeff_data, in, out);
    zebra::radon_transform_inplace(out);
}

void
ZernikeCoordinateMultiplier::multiply_by_r2_and_radon_transform_inplace(
    ZernikeExpansionSpan<const std::array<double, 2>> in,
    ZernikeExpansionSpan<std::array<double, 2>> out
) const noexcept
{
    assert(in.order() + 3 < out.order());
    multiply_by_r2_impl(m_coeff_data, in, out);
    zebra::radon_transform_inplace(out);
}

} // namespace detail
} // namespace zebra
} // namespace zdm
