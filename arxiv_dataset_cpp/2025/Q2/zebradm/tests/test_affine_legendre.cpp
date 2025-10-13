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

#include <array>
#include <cmath>
#include <cstdio>
#include <cassert>

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol*0.5*std::fabs(a + b) + tol;
}

constexpr double P_0([[maybe_unused]] double x) { return 1.0; }
constexpr double P_1(double x) { return x; }
constexpr double P_2(double x) { return (1.0/2.0)*(3.0*x*x - 1.0); }
constexpr double P_3(double x) { return (1.0/2.0)*(5.0*x*x - 3.0)*x; }
constexpr double P_4(double x) { return (1.0/8.0)*((35.0*x*x - 30.0)*x*x + 3.0); }
constexpr double P_5(double x) { return (1.0/8.0)*((63.0*x*x - 70.0)*x*x + 15.0)*x; }

bool legendre_affine_coeffs_expand_legendre(double a, double b, double x)
{
    const double P_0x = P_0(x);
    const double P_1x = P_1(x);
    const double P_2x = P_2(x);
    const double P_3x = P_3(x);
    const double P_4x = P_4(x);
    const double P_5x = P_5(x);

    const double y = a + b*x;
    const double P_0y = P_0(y);
    const double P_1y = P_1(y);
    const double P_2y = P_2(y);
    const double P_3y = P_3(y);
    const double P_4y = P_4(y);
    const double P_5y = P_5(y);

    constexpr std::size_t order = 6;

    using TriangleSpan = zest::TriangleSpan<double, zest::TriangleLayout<zest::IndexingMode::nonnegative>>;

    zdm::zebra::AffineLegendreRecursion recursion(order);
    std::vector<double> coeffs(zest::TriangleLayout<zest::IndexingMode::nonnegative>::size(order));

    recursion.evaluate_affine(TriangleSpan(coeffs, order), a, b);

    const double P_0y_exp = coeffs[0]*P_0x;
    const double P_1y_exp = coeffs[1]*P_0x + coeffs[2]*P_1x;
    const double P_2y_exp = coeffs[3]*P_0x + coeffs[4]*P_1x + coeffs[5]*P_2x;
    const double P_3y_exp = coeffs[6]*P_0x + coeffs[7]*P_1x + coeffs[8]*P_2x + coeffs[9]*P_3x;
    const double P_4y_exp = coeffs[10]*P_0x + coeffs[11]*P_1x + coeffs[12]*P_2x + coeffs[13]*P_3x + coeffs[14]*P_4x;
    const double P_5y_exp = coeffs[15]*P_0x + coeffs[16]*P_1x + coeffs[17]*P_2x + coeffs[18]*P_3x + coeffs[19]*P_4x + coeffs[20]*P_5x;

    constexpr double tol = 1.0e-13;
    bool success = is_close(P_0y_exp, P_0y, tol)
            && is_close(P_1y_exp, P_1y, tol)
            && is_close(P_2y_exp, P_2y, tol)
            && is_close(P_3y_exp, P_3y, tol)
            && is_close(P_4y_exp, P_4y, tol)
            && is_close(P_5y_exp, P_5y, tol);
    
    if (!success)
    {
        std::printf("a = %f, b = %f, x = %f\n", a, b, x);
        std::printf("P_0(a + bx): %.16e %.16e\n", P_0y_exp, P_0y);
        std::printf("P_1(a + bx): %.16e %.16e\n", P_1y_exp, P_1y);
        std::printf("P_2(a + bx): %.16e %.16e\n", P_2y_exp, P_2y);
        std::printf("P_3(a + bx): %.16e %.16e\n", P_3y_exp, P_3y);
        std::printf("P_4(a + bx): %.16e %.16e\n", P_4y_exp, P_4y);
        std::printf("P_5(a + bx): %.16e %.16e\n", P_5y_exp, P_5y);
    }
    
    return success;
}

bool legendre_shift_coeffs_expand_legendre(double a, double x)
{
    const double P_0x = P_0(x);
    const double P_1x = P_1(x);
    const double P_2x = P_2(x);
    const double P_3x = P_3(x);
    const double P_4x = P_4(x);
    const double P_5x = P_5(x);

    const double y = a + x;
    const double P_0y = P_0(y);
    const double P_1y = P_1(y);
    const double P_2y = P_2(y);
    const double P_3y = P_3(y);
    const double P_4y = P_4(y);
    const double P_5y = P_5(y);

    constexpr std::size_t order = 6;

    using TriangleSpan = zest::TriangleSpan<double, zest::TriangleLayout<zest::IndexingMode::nonnegative>>;

    zdm::zebra::AffineLegendreRecursion recursion(order);
    std::vector<double> coeffs(zest::TriangleLayout<zest::IndexingMode::nonnegative>::size(order));

    recursion.evaluate_shifted(TriangleSpan(coeffs, order), a);

    const double P_0y_exp = coeffs[0]*P_0x;
    const double P_1y_exp = coeffs[1]*P_0x + coeffs[2]*P_1x;
    const double P_2y_exp = coeffs[3]*P_0x + coeffs[4]*P_1x + coeffs[5]*P_2x;
    const double P_3y_exp = coeffs[6]*P_0x + coeffs[7]*P_1x + coeffs[8]*P_2x + coeffs[9]*P_3x;
    const double P_4y_exp = coeffs[10]*P_0x + coeffs[11]*P_1x + coeffs[12]*P_2x + coeffs[13]*P_3x + coeffs[14]*P_4x;
    const double P_5y_exp = coeffs[15]*P_0x + coeffs[16]*P_1x + coeffs[17]*P_2x + coeffs[18]*P_3x + coeffs[19]*P_4x + coeffs[20]*P_5x;

    constexpr double tol = 1.0e-13;
    bool success = is_close(P_0y_exp, P_0y, tol)
            && is_close(P_1y_exp, P_1y, tol)
            && is_close(P_2y_exp, P_2y, tol)
            && is_close(P_3y_exp, P_3y, tol)
            && is_close(P_4y_exp, P_4y, tol)
            && is_close(P_5y_exp, P_5y, tol);
    
    if (!success)
    {
        std::printf("a = %f, x = %f\n", a, x);
        std::printf("P_0(a + x): %.16e %.16e\n", P_0y_exp, P_0y);
        std::printf("P_1(a + x): %.16e %.16e\n", P_1y_exp, P_1y);
        std::printf("P_2(a + x): %.16e %.16e\n", P_2y_exp, P_2y);
        std::printf("P_3(a + x): %.16e %.16e\n", P_3y_exp, P_3y);
        std::printf("P_4(a + x): %.16e %.16e\n", P_4y_exp, P_4y);
        std::printf("P_5(a + x): %.16e %.16e\n", P_5y_exp, P_5y);
    }
    
    return success;
}

bool legendre_scale_coeffs_expand_legendre(double b, double x)
{
    const double P_0x = P_0(x);
    const double P_1x = P_1(x);
    const double P_2x = P_2(x);
    const double P_3x = P_3(x);
    const double P_4x = P_4(x);
    const double P_5x = P_5(x);

    const double y = b*x;
    const double P_0y = P_0(y);
    const double P_1y = P_1(y);
    const double P_2y = P_2(y);
    const double P_3y = P_3(y);
    const double P_4y = P_4(y);
    const double P_5y = P_5(y);

    constexpr std::size_t order = 6;

    using TriangleSpan = zest::TriangleSpan<double, zest::TriangleLayout<zest::IndexingMode::nonnegative>>;

    zdm::zebra::AffineLegendreRecursion recursion(order);
    std::vector<double> coeffs(zest::TriangleLayout<zest::IndexingMode::nonnegative>::size(order));

    recursion.evaluate_scaled(TriangleSpan(coeffs, order), b);

    const double P_0y_exp = coeffs[0]*P_0x;
    const double P_1y_exp = coeffs[1]*P_0x + coeffs[2]*P_1x;
    const double P_2y_exp = coeffs[3]*P_0x + coeffs[4]*P_1x + coeffs[5]*P_2x;
    const double P_3y_exp = coeffs[6]*P_0x + coeffs[7]*P_1x + coeffs[8]*P_2x + coeffs[9]*P_3x;
    const double P_4y_exp = coeffs[10]*P_0x + coeffs[11]*P_1x + coeffs[12]*P_2x + coeffs[13]*P_3x + coeffs[14]*P_4x;
    const double P_5y_exp = coeffs[15]*P_0x + coeffs[16]*P_1x + coeffs[17]*P_2x + coeffs[18]*P_3x + coeffs[19]*P_4x + coeffs[20]*P_5x;

    constexpr double tol = 1.0e-13;
    bool success = is_close(P_0y_exp, P_0y, tol)
            && is_close(P_1y_exp, P_1y, tol)
            && is_close(P_2y_exp, P_2y, tol)
            && is_close(P_3y_exp, P_3y, tol)
            && is_close(P_4y_exp, P_4y, tol)
            && is_close(P_5y_exp, P_5y, tol);
    
    if (!success)
    {
        std::printf("b = %f, x = %f\n", b, x);
        std::printf("P_0(bx): %.16e %.16e\n", P_0y_exp, P_0y);
        std::printf("P_1(bx): %.16e %.16e\n", P_1y_exp, P_1y);
        std::printf("P_2(bx): %.16e %.16e\n", P_2y_exp, P_2y);
        std::printf("P_3(bx): %.16e %.16e\n", P_3y_exp, P_3y);
        std::printf("P_4(bx): %.16e %.16e\n", P_4y_exp, P_4y);
        std::printf("P_5(bx): %.16e %.16e\n", P_5y_exp, P_5y);
    }
    
    return success;
}

int main()
{
    constexpr std::array<double, 7> a_list = {
        -1.5, -1.0, -0.235467, 0.0, 0.235467, 1.0, 1.5
    };
    constexpr std::array<double, 5> b_list = {0.3, 0.5, 0.235467, 1.0, 1.2};
    constexpr std::array<double, 5> x_list = {-1.0, -0.235467, 0.0, 0.235467, 1.0};
    
    for (auto a : a_list)
    {
        for (auto b : b_list)
        {
            for (auto x : x_list)
            {
                assert(legendre_affine_coeffs_expand_legendre(a, b, x));
            }
        }
    }
    
    for (auto a : a_list)
    {
        for (auto x : x_list)
        {
            assert(legendre_shift_coeffs_expand_legendre(a, x));
        }
    }
    
    for (auto b : b_list)
    {
        for (auto x : x_list)
        {
            assert(legendre_scale_coeffs_expand_legendre(b, x));
        }
    }
}