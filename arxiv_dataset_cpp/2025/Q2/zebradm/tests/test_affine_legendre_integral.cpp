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

#include "zest/gauss_legendre.hpp"

#include <cmath>
#include <cassert>

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol*0.5*std::fabs(a + b) + tol;
}

bool test_affine_legendre_integral_recursion_takes_zero_size_span()
{
    zdm::zebra::AffineLegendreIntegrals recursion(0, 0);
    recursion.integrals(zdm::zebra::TrapezoidSpan<double>{}, 0.0, 1.0);

    return true;
}

bool test_affine_legendre_integral_recursion_is_correct_for_order_5_extra_extent_0(double shift, double scale)
{
    constexpr std::size_t order = 5;
    constexpr std::size_t extra_extent = 0;

    const double a = std::clamp(-(1.0 + shift)/scale, -1.0, 1.0);
    const double b = std::clamp((1.0 - shift)/scale, -1.0, 1.0);

    const double A00 = b - a;

    const double c = shift + a*scale;
    const double d = shift + b*scale;
    const double A10 = (1.0/scale)*0.5*(d*d - c*c);
    const double A20 = (0.5*(d*d - 1.0)*d - 0.5*(c*c - 1.0)*c)/scale;
    const double A30 = (0.5*((5.0/4.0)*d*d - (3.0/2.0))*d*d
            - 0.5*((5.0/4.0)*c*c - (3.0/2.0))*c*c)/scale;
    const double A40 = (0.125*((7.0*d*d - 10.0)*d*d + 3.0)*d
            - 0.125*((7.0*c*c - 10.0)*c*c + 3.0)*c)/scale;

    const double a112 = 0.5*shift;
    const double a113 = scale/3.0;
    const double A11 = b*b*(a112 + b*a113) - a*a*(a112 + a*a113);

    const double a212 = 0.75*shift*shift - 0.25;
    const double a213 = shift*scale;
    const double a214 = 3.0*scale*scale/8.0;
    const double A21
        = b*b*(a212 + b*(a213 + b*a214)) - a*a*(a212 + a*(a213 + a*a214));
    
    const double a312 = (1.25*shift*shift - 0.75)*shift;
    const double a313 = (2.5*shift*shift - 0.5)*scale;
    const double a314 = 15.0*shift*scale*scale/8.0;
    const double a315 = 0.5*scale*scale*scale;
    const double A31
        = b*b*(a312 + b*(a313 + b*(a314 + b*a315)))
        - a*a*(a312 + a*(a313 + a*(a314 + a*a315)));
    
    const double a221 = 0.25 - 0.75*shift*shift;
    const double a222 = -0.75*shift*scale;
    const double a223 = 0.75*shift*shift - 0.25*scale*scale - 0.25;
    const double a224 = (9.0/8.0)*shift*scale;
    const double a225 = (9.0/20.0)*scale*scale;
    const double A22
        = b*(a221 + b*(a222 + b*(a223 + b*(a224 + b*a225))))
        - a*(a221 + a*(a222 + a*(a223 + a*(a224 + a*a225))));

    zdm::zebra::AffineLegendreIntegrals recursion(order, extra_extent);

    std::vector<double> trapezoid_buffer(
        zdm::zebra::TrapezoidSpan<double>::Layout::size(order, extra_extent));
    
    zdm::zebra::TrapezoidSpan<double> test_trapezoid(
        trapezoid_buffer.data(), order, extra_extent);
    recursion.integrals(test_trapezoid, shift, scale);

    constexpr double tol = 1.0e-13;
    bool success = is_close(test_trapezoid(0, 0), A00, tol)
            && is_close(test_trapezoid(1, 0), A10, tol)
            && is_close(test_trapezoid(2, 0), A20, tol)
            && is_close(test_trapezoid(3, 0), A30, tol)
            && is_close(test_trapezoid(4, 0), A40, tol)
            && is_close(test_trapezoid(1, 1), A11, tol)
            && is_close(test_trapezoid(2, 1), A21, tol)
            && is_close(test_trapezoid(3, 1), A31, tol)
            && is_close(test_trapezoid(2, 2), A22, tol);
    
    if (!success)
    {
        std::printf("%f %f\n", shift, scale);
        std::printf("A00: %.16e %.16e\n", test_trapezoid(0, 0), A00);
        std::printf("A10: %.16e %.16e\n", test_trapezoid(1, 0), A10);
        std::printf("A20: %.16e %.16e\n", test_trapezoid(2, 0), A20);
        std::printf("A30: %.16e %.16e\n", test_trapezoid(3, 0), A30);
        std::printf("A40: %.16e %.16e\n", test_trapezoid(4, 0), A40);
        std::printf("A11: %.16e %.16e\n", test_trapezoid(1, 1), A11);
        std::printf("A21: %.16e %.16e\n", test_trapezoid(2, 1), A21);
        std::printf("A31: %.16e %.16e\n", test_trapezoid(3, 1), A31);
        std::printf("A22: %.16e %.16e\n", test_trapezoid(2, 2), A22);
    }

    return success;
}

bool test_affine_legendre_integral_recursion_is_correct_for_order_5_extra_extent_3(double shift, double scale)
{
    constexpr std::size_t order = 5;
    constexpr std::size_t extra_extent = 3;

    const double a = std::clamp(-(1.0 + shift)/scale, -1.0, 1.0);
    const double b = std::clamp((1.0 - shift)/scale, -1.0, 1.0);

    const double A00 = b - a;
    const double A01 = 0.5*(b*b - a*a);
    const double A02 = 0.5*(b*b - 1.0)*b - 0.5*(a*a - 1.0)*a;
    const double A03 = 0.5*((5.0/4.0)*b*b - (3.0/2.0))*b*b - 0.5*((5.0/4.0)*a*a - (3.0/2.0))*a*a;

    const double c = shift + a*scale;
    const double d = shift + b*scale;
    const double A10 = (1.0/scale)*0.5*(d*d - c*c);
    const double A20 = (0.5*(d*d - 1.0)*d - 0.5*(c*c - 1.0)*c)/scale;
    const double A30 = (0.5*((5.0/4.0)*d*d - (3.0/2.0))*d*d - 0.5*((5.0/4.0)*c*c - (3.0/2.0))*c*c)/scale;
    const double A40 = (0.125*((7.0*d*d - 10.0)*d*d + 3.0)*d - 0.125*((7.0*c*c - 10.0)*c*c + 3.0)*c)/scale;

    zdm::zebra::AffineLegendreIntegrals recursion(order, extra_extent);

    std::vector<double> trapezoid_buffer(
        zdm::zebra::TrapezoidSpan<double>::Layout::size(order, extra_extent));
    
    zdm::zebra::TrapezoidSpan<double> test_trapezoid(
        trapezoid_buffer.data(), order, extra_extent);
    recursion.integrals(test_trapezoid, shift, scale);

    constexpr double tol = 1.0e-13;
    bool success = is_close(test_trapezoid(0, 0), A00, tol)
            && is_close(test_trapezoid(0, 1), A01, tol)
            && is_close(test_trapezoid(0, 2), A02, tol)
            && is_close(test_trapezoid(0, 3), A03, tol)
            && is_close(test_trapezoid(1, 0), A10, tol)
            && is_close(test_trapezoid(2, 0), A20, tol)
            && is_close(test_trapezoid(3, 0), A30, tol)
            && is_close(test_trapezoid(4, 0), A40, tol);
    
    if (!success)
    {
        std::printf("%f %f\n", shift, scale);
        std::printf("A00: %.16e %.16e\n", test_trapezoid(0, 0), A00);
        std::printf("A01: %.16e %.16e\n", test_trapezoid(0, 1), A01);
        std::printf("A02: %.16e %.16e\n", test_trapezoid(0, 2), A02);
        std::printf("A03: %.16e %.16e\n", test_trapezoid(0, 3), A03);
        std::printf("A10: %.16e %.16e\n", test_trapezoid(1, 0), A10);
        std::printf("A20: %.16e %.16e\n", test_trapezoid(2, 0), A20);
        std::printf("A30: %.16e %.16e\n", test_trapezoid(3, 0), A30);
        std::printf("A40: %.16e %.16e\n", test_trapezoid(4, 0), A40);
    }

    return success;
}

bool test_affine_legendre_integral_recursion_matches_numerical_integral_for_order_20_extra_extent_4(double shift, double scale)
{
    constexpr std::size_t order = 20;
    constexpr std::size_t extra_extent = 4;
    constexpr std::size_t last_extent = order + extra_extent;
    constexpr std::size_t num_glq_nodes = (order + last_extent)/2;
    std::vector<double> glq_nodes(num_glq_nodes);
    std::vector<double> glq_weights(num_glq_nodes);
    zest::gl::gl_nodes_and_weights<zest::gl::UnpackedLayout, zest::gl::GLNodeStyle::cos>(
            glq_nodes, glq_weights, glq_weights.size() & 1);

    const double a = std::clamp(-(1.0 + shift)/scale, -1.0, 1.0);
    const double b = std::clamp((1.0 - shift)/scale, -1.0, 1.0);

    const double half_width = 0.5*(b - a);
    const double mid_point = 0.5*(b + a);

    for (std::size_t i = 0; i < glq_nodes.size(); ++i)
        glq_nodes[i] = half_width*glq_nodes[i] + mid_point;
    
    std::vector<double> legendre_buffer(last_extent*num_glq_nodes);
    zest::MDSpan<double, 2> legendre(
            legendre_buffer.data(), {last_extent, glq_weights.size()});
    if (scale > shift - 1.0)
        zdm::zebra::legendre_recursion_vec(legendre, glq_nodes);

    for (std::size_t i = 0; i < glq_nodes.size(); ++i)
        glq_nodes[i] = shift + scale*glq_nodes[i];

    std::vector<double> affine_legendre_buffer(order*num_glq_nodes);
    zest::MDSpan<double, 2> affine_legendre(
            affine_legendre_buffer.data(), {order, glq_weights.size()});
    if (scale > shift - 1.0)
        zdm::zebra::legendre_recursion_vec(affine_legendre, glq_nodes);

    std::vector<double> reference_integral_buffer(
            zdm::zebra::TrapezoidSpan<double>::Layout::size(order, extra_extent));
    zdm::zebra::TrapezoidSpan<double> reference_integrals(
            reference_integral_buffer.data(), order, extra_extent);
    
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = 0; l < n + extra_extent + 1; ++l)
        {
            double res = 0.0;
            for (std::size_t i = 0; i < glq_nodes.size(); ++i)
                res += glq_weights[i]*affine_legendre(n, i)*legendre(l, i);
            reference_integrals(n, l) = half_width*res;
        }
    }

    std::vector<double> test_integral_buffer(
            zdm::zebra::TrapezoidSpan<double>::Layout::size(order, extra_extent));
    zdm::zebra::TrapezoidSpan<double> test_integrals(
            test_integral_buffer.data(), order, extra_extent);
    
    zdm::zebra::AffineLegendreIntegrals recursion(order, extra_extent);
    recursion.integrals(test_integrals, shift, scale);

    constexpr double tol = 1.0e-13;
    bool success = true;
    for (std::size_t n = 0; n < order; ++n)
    {
        for (std::size_t l = 0; l < n + extra_extent + 1; ++l)
        {
            success = success && is_close(test_integrals(n, l), reference_integrals(n, l), tol);
        }
    }

    if (!success)
    {
        std::printf("%f %f\n", shift, scale);
        for (std::size_t n = 0; n < order; ++n)
        {
            for (std::size_t l = 0; l < n + extra_extent + 1; ++l)
            {
                std::printf("(%lu,%lu): %.16e %.16e ", n, l, test_integrals(n, l), reference_integrals(n, l));
                if (std::fabs(reference_integrals(n, l) - test_integrals(n, l)) > tol)
                    std::printf("%.16e\n", reference_integrals(n, l) - test_integrals(n, l));
                else
                    std::printf("\n");
            }
        }
    }

    return success;
}

int main()
{
    assert(test_affine_legendre_integral_recursion_takes_zero_size_span());

    constexpr std::array<double, 7> shift_list = {0.0, 0.235467, 1.0, 1.5, 1.8};
    constexpr std::array<double, 5> scale_list = {0.3, 0.5, 0.235467, 1.0, 1.2};

    for (auto shift : shift_list)
    {
        for (auto scale : scale_list)
        {
            assert(test_affine_legendre_integral_recursion_is_correct_for_order_5_extra_extent_0(shift, scale));
            assert(test_affine_legendre_integral_recursion_is_correct_for_order_5_extra_extent_3(shift, scale));
            assert(test_affine_legendre_integral_recursion_matches_numerical_integral_for_order_20_extra_extent_4(shift, scale));
        }
    }
}