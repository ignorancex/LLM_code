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
#include "affine_legendre.hpp"

#include <algorithm>
#include <cassert>

namespace zdm
{
namespace zebra
{

AffineLegendreRecursion::AffineLegendreRecursion(std::size_t max_order):
    m_a(max_order), m_b(max_order), m_c(max_order), m_d(max_order),
    m_max_order(max_order)
{
    m_c[1] = 2.0/5.0;
    m_d[1] = 1.0;
    for (std::size_t n = 2; n < max_order; ++n)
    {
        const double dn = double(n);
        const double inv_dn = 1.0/dn;
        m_a[n] = (2.0*dn - 1.0)*inv_dn;
        m_b[n] = (dn - 1.0)*inv_dn;
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0);
        m_d[n] = dn/(2.0*dn - 1.0);
    }
}

void AffineLegendreRecursion::expand(std::size_t max_order)
{
    if (max_order <= m_max_order) return;

    m_a.resize(max_order);
    m_b.resize(max_order);
    m_c.resize(max_order);
    m_d.resize(max_order);

    for (std::size_t n = m_max_order; n < max_order; ++n)
    {
        const double dn = double(n);
        const double inv_dn = 1.0/dn;
        m_a[n] = (2.0*dn - 1.0)*inv_dn;
        m_b[n] = (dn - 1.0)*inv_dn;
        m_c[n] = (dn + 1.0)/(2.0*dn + 3.0);
        m_d[n] = dn/(2.0*dn - 1.0);
    }

    m_max_order = max_order;
}

void AffineLegendreRecursion::evaluate_affine(
    zest::TriangleSpan<double, zest::TriangleLayout<zest::IndexingMode::nonnegative>> expansion, double shift, double scale)
{
    if (expansion.order() == 0) return;
    expand(expansion.order());

    expansion(0, 0) = 1.0;
    if (expansion.order() == 1) return;

    expansion(1, 0) = shift;
    expansion(1, 1) = scale;
    for (std::size_t n = 2; n < expansion.order(); ++n)
    {
        expansion(n, 0) = m_a[n]*shift*expansion(n - 1, 0)
                - m_b[n]*expansion(n - 2, 0)
                + m_a[n]*(1.0/3.0)*scale*expansion(n - 1, 1);
        for (std::size_t l = 1; l <= n - 2; ++l)
        {
            expansion(n, l) = m_a[n]*shift*expansion(n - 1, l)
                    - m_b[n]*expansion(n - 2, l)
                    + m_a[n]*m_c[l]*scale*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*scale*expansion(n - 1, l - 1);
        }

        expansion(n, n - 1) = m_a[n]*shift*expansion(n - 1, n - 1)
                + m_a[n]*m_d[n - 1]*scale*expansion(n - 1, n - 2);
        expansion(n, n) = scale*expansion(n - 1, n - 1);
    }
}

void AffineLegendreRecursion::evaluate_shifted(
    zest::TriangleSpan<double, zest::TriangleLayout<zest::IndexingMode::nonnegative>> expansion, double shift)
{
    if (expansion.order() == 0) return;
    expand(expansion.order());

    expansion(0, 0) = 1.0;
    if (expansion.order() == 1) return;

    expansion(1, 0) = shift;
    expansion(1, 1) = 1.0;
    for (std::size_t n = 2; n < expansion.order(); ++n)
    {
        expansion(n, 0) = m_a[n]*shift*expansion(n - 1, 0)
                - m_b[n]*expansion(n - 2, 0)
                + m_a[n]*(1.0/3.0)*expansion(n - 1, 1);
        for (std::size_t l = 1; l <= n - 2; ++l)
        {
            expansion(n, l) = m_a[n]*shift*expansion(n - 1, l)
                    - m_b[n]*expansion(n - 2, l)
                    + m_a[n]*m_c[l]*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*expansion(n - 1, l - 1);
        }

        expansion(n, n - 1) = m_a[n]*shift*expansion(n - 1, n - 1)
                + m_a[n]*m_d[n - 1]*expansion(n - 1, n - 2);
        expansion(n, n) = expansion(n - 1, n - 1);
    }
}

void AffineLegendreRecursion::evaluate_scaled(
    zest::TriangleSpan<double, zest::TriangleLayout<zest::IndexingMode::nonnegative>> expansion, double scale)
{
    if (expansion.order() == 0) return;
    expand(expansion.order());

    std::ranges::fill(expansion.flatten(), 0.0);
    expansion(0, 0) = 1.0;
    if (expansion.order() == 1) return;

    expansion(1, 1) = scale;
    for (std::size_t n = 2; n < expansion.order(); ++n)
    {
        expansion(n, 0) = m_a[n]*(1.0/3.0)*scale*expansion(n - 1, 1)
                - m_b[n]*expansion(n - 2, 0);
        
        const std::size_t lmin = 1 + ((n & 1) ^ 1);
        for (std::size_t l = lmin; l <= n - 2; l += 2)
        {
            expansion(n, l) = m_a[n]*m_c[l]*scale*expansion(n - 1, l + 1)
                    + m_a[n]*m_d[l]*scale*expansion(n - 1, l - 1)
                    - m_b[n]*expansion(n - 2, l);
        }

        expansion(n, n) = scale*expansion(n - 1, n - 1);
    }
}

} // namespace zebra
} // namespace zdm