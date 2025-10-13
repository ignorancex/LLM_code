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
#include "legendre.hpp"

#include <stdexcept>
#include <cmath>

namespace zdm
{
namespace zebra
{

void legendre_recursion_vec(
    zest::MDSpan<double, 2> legendre, std::span<double> x)
{
    const std::size_t order = legendre.extents()[0];
    const std::size_t size = legendre.extents()[1];

    if (order == 0) return;

    if (size != x.size())
        throw std::invalid_argument(
                "size of x is incompatible with size of leg");
    
    for (auto xi : x)
    {
        if (std::fabs(xi) > 1.0)
            throw std::invalid_argument("x must be between -1 and 1");
    }

    zest::MDSpan<double, 1> leg_0 = legendre[0];
    for (std::size_t i = 0; i < size; ++i)
        leg_0[i] = 1.0;
    
    if (order == 1) return;

    zest::MDSpan<double, 1> leg_1 = legendre[1];
    for (std::size_t i = 0; i < size; ++i)
        leg_1[i] = x[i];
    
    for (std::size_t n = 2; n < order; ++n)
    {
        zest::MDSpan<double, 1> leg_n = legendre[n];
        zest::MDSpan<double, 1> leg_nm1 = legendre[n - 1];
        zest::MDSpan<double, 1> leg_nm2 = legendre[n - 2];
        const double inv_n = 1.0/double(n);
        const double a = double(2*n - 1)*inv_n;
        const double b = double(n - 1)*inv_n;
        for (std::size_t i = 0; i < size; ++i)
            leg_n[i] = a*x[i]*leg_nm1[i] - b*leg_nm2[i];
    }
}

void legendre_recursion(std::span<double> legendre, double x)
{
    const std::size_t order = legendre.size();

    if (order == 0) return;
    
    if (std::fabs(x) > 1.0)
        throw std::invalid_argument("x must be between -1 and 1");

    legendre[0] = 1.0;
    if (order == 1) return;

    legendre[1] = x;
    
    for (std::size_t n = 2; n < order; ++n)
    {
        const double inv_n = 1.0/double(n);
        const double a = double(2*n - 1)*inv_n;
        const double b = double(n - 1)*inv_n;
        legendre[n] = a*x*legendre[n - 1] - b*legendre[n - 2];
    }
}

LegendreArrayRecursion::LegendreArrayRecursion(std::size_t size):
    m_buffers{std::vector<double>(size), std::vector<double>(size), std::vector<double>(size)}, m_x(size), m_size(size) {}

LegendreArrayRecursion::LegendreArrayRecursion(std::span<const double> x):
    m_buffers{std::vector<double>(x.size()), std::vector<double>(x.size()), std::vector<double>(x.size())}, m_x(x.size()), m_size(x.size())
{
    init(x);
}

void LegendreArrayRecursion::resize(std::size_t size)
{
    if (size != m_size)
    {
        for (auto& buffer : m_buffers)
            buffer.resize(size);
        m_x.resize(size);
        m_size = size;
    }
}

void LegendreArrayRecursion::init(std::span<const double> x)
{
    resize(x.size());

    std::ranges::fill(m_buffers[0], 1.0);
    std::ranges::copy(x, m_x.begin());
    std::ranges::copy(m_x, m_buffers[1].begin());
    reset();
}

void LegendreArrayRecursion::iterate() noexcept
{
    double* temp = m_second_prev;
    m_second_prev = m_prev;
    m_prev = m_current;
    m_current = temp;

    if (m_l > 0)
    {
        const double inv_l = 1.0/double(m_l + 1);
        const double a = double(2*m_l + 1)*inv_l;
        const double b = double(m_l)*inv_l;
        for (std::size_t i = 0; i < m_size; ++i)
            m_current[i] = a*m_x[i]*m_prev[i] - b*m_second_prev[i];
    }
    ++m_l;
}

void LegendreArrayRecursion::iterate(std::size_t n) noexcept
{
    if (n == 0) return;
    if (m_l == 0)
    {
        double* temp = m_second_prev;
        m_second_prev = m_prev;
        m_prev = m_current;
        m_current = temp;
        ++m_l;
        --n;
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        double* temp = m_second_prev;
        m_second_prev = m_prev;
        m_prev = m_current;
        m_current = temp;

        const double inv_l = 1.0/double(m_l + 1);
        const double a = double(2*m_l + 1)*inv_l;
        const double b = double(m_l)*inv_l;
        for (std::size_t i = 0; i < m_size; ++i)
            m_current[i] = a*m_x[i]*m_prev[i] - b*m_second_prev[i];
        ++m_l;
    }
}

std::span<const double> LegendreArrayRecursion::next() noexcept
{
    iterate();
    return current();
}

void LegendreArrayRecursion::reset() noexcept
{
    m_current = m_buffers[0].data();
    m_second_prev = m_buffers[1].data();
    m_prev = m_buffers[2].data();
    m_l = 0;
}

LegendreIntegralRecursion::LegendreIntegralRecursion(std::size_t order):
    m_a(order), m_b(order), m_order(order)
{
    for (std::size_t l = 2; l < order; ++l)
    {
        const double dl = double(l);
        m_a[l] = (2.0*dl - 1.0)/(dl + 1.0);
        m_b[l] = (dl - 2.0)/(dl + 1.0);
    }
}

void LegendreIntegralRecursion::expand(std::size_t order)
{
    if (order <= m_order) return;
    m_a.resize(order);
    m_b.resize(order);

    for (std::size_t l = m_order; l < order; ++l)
    {
        const double dl = double(l);
        m_a[l] = (2.0*dl - 1.0)/(dl + 1.0);
        m_b[l] = (dl - 2.0)/(dl + 1.0);
    }

    m_order = order;
}

void LegendreIntegralRecursion::legendre_integral(
    std::span<double> pl, double x)
{
    if (pl.size() == 0) return;

    std::size_t lmax = pl.size() - 1;
    expand(lmax);

    pl[0] = x + 1.0;

    if (lmax == 0) return;
    pl[1] = 0.5*(x - 1.0)*(x + 1.0);

    for (std::size_t l = 2; l <= lmax; ++l)
        pl[l] = m_a[l]*x*pl[l - 1] - m_b[l]*pl[l - 2];
}

} // namespace zebra
} // namespace zdm