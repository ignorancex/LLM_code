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

#include <cassert>
#include <cstdio>
#include <cmath>

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol*0.5*std::fabs(a + b) + tol;
}

bool test_legendre_array_recursion_next_is_correct_to_order_6(double x)
{
    constexpr std::size_t order = 6;
    const std::array<double, order> p_ref = {
        1.0,
        x,
        0.5*(3.0*x*x - 1.0),
        0.5*(5.0*x*x*x - 3.0*x),
        0.125*(35.0*x*x*x*x - 30.0*x*x + 3.0),
        0.125*(63.0*x*x*x*x*x - 70.0*x*x*x + 15.0*x),
    };

    zdm::zebra::LegendreArrayRecursion recursion(std::span<double>(&x, 1));

    std::array<double, order> p{};
    p[0] = recursion.current()[0];
    for (std::size_t i = 1; i < p_ref.size(); ++i)
    {
        std::span<const double> pi = recursion.next();
        p[i] = pi[0];
    }

    constexpr double tol = 1.0e-13;
    bool success = true;
    for (std::size_t i = 0; i < p_ref.size(); ++i)
        success = success && (is_close(p[i], p_ref[i], tol));
    
    if (!success)
    {
        for (std::size_t i = 0; i < p_ref.size(); ++i)
            std::printf("P%lu: %f %f\n", i, p[i], p_ref[i]);
    }

    return success;
}

bool test_legendre_array_recursion_iterate_is_correct_to_order_6(double x)
{
    constexpr std::size_t order = 6;
    const std::array<double, order> p_ref = {
        1.0,
        x,
        0.5*(3.0*x*x - 1.0),
        0.5*(5.0*x*x*x - 3.0*x),
        0.125*(35.0*x*x*x*x - 30.0*x*x + 3.0),
        0.125*(63.0*x*x*x*x*x - 70.0*x*x*x + 15.0*x),
    };

    zdm::zebra::LegendreArrayRecursion recursion(std::span<double>(&x, 1));
    recursion.iterate(order - 1);

    constexpr double tol = 1.0e-13;
    bool success =  (is_close(recursion.current()[0], p_ref[order - 1], tol))
        && (is_close(recursion.prev()[0], p_ref[order - 2], tol))
        && (is_close(recursion.second_prev()[0], p_ref[order - 3], tol));
    
    if (!success)
    {
        std::printf("P%lu: %f %f\n", order - 1, recursion.current()[0], p_ref[order - 1]);
        std::printf("P%lu: %f %f\n", order - 2, recursion.prev()[0], p_ref[order - 2]);
        std::printf("P%lu: %f %f\n", order - 3, recursion.second_prev()[0], p_ref[order - 3]);
    }

    return success;
}

bool test_legendre_integral_recursion_is_correct_to_order_6(double x)
{
    constexpr std::size_t order = 6;
    const std::array<double, order> integral_ref = {
        x + 1.0,
        0.5*x*x - 0.5,
        0.5*(x*x*x - 1.0*x),
        0.5*((5.0/4.0)*x*x*x*x - (3.0/2.0)*x*x) + 0.125,
        0.125*(7.0*x*x*x*x*x - 10.0*x*x*x + 3.0*x),
        0.125*((63.0/6.0)*x*x*x*x*x*x - (70.0/4.0)*x*x*x*x + (15.0/2.0)*x*x) - 0.0625,
    };

    zdm::zebra::LegendreIntegralRecursion recursion(order);

    std::array<double, order> integrals;
    recursion.legendre_integral(integrals, x);

    bool success = true;
    for (std::size_t i = 0; i < integral_ref.size(); ++i)
        success = success && (integrals[i] == integral_ref[i]);
    
    if (!success)
    {
        for (std::size_t i = 0; i < integral_ref.size(); ++i)
            std::printf("P%lu: %f %f\n", i, integrals[i], integral_ref[i]);
    }

    return success;
}

int main()
{
    assert(test_legendre_array_recursion_next_is_correct_to_order_6(-1.0));
    assert(test_legendre_array_recursion_next_is_correct_to_order_6(1.0));
    assert(test_legendre_array_recursion_next_is_correct_to_order_6(0.0));
    assert(test_legendre_array_recursion_next_is_correct_to_order_6(0.3));

    assert(test_legendre_array_recursion_iterate_is_correct_to_order_6(-1.0));
    assert(test_legendre_array_recursion_iterate_is_correct_to_order_6(1.0));
    assert(test_legendre_array_recursion_iterate_is_correct_to_order_6(0.0));
    assert(test_legendre_array_recursion_iterate_is_correct_to_order_6(0.3));

    assert(test_legendre_integral_recursion_is_correct_to_order_6(-1.0));
    assert(test_legendre_integral_recursion_is_correct_to_order_6(1.0));
    assert(test_legendre_integral_recursion_is_correct_to_order_6(0.0));
    assert(test_legendre_integral_recursion_is_correct_to_order_6(0.2));
}