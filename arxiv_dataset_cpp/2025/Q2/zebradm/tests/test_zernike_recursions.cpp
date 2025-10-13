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
#include <random>

#include "zest/zernike_glq_transformer.hpp"

constexpr bool is_close(
    std::array<double, 2> a, std::array<double, 2> b, double tol)
{
    return std::fabs(a[0] - b[0]) < tol*0.5*std::fabs(a[0] + b[0]) + tol && std::fabs(a[1] - b[1]) < tol*0.5*std::fabs(a[1] + b[1]) + tol;
}

bool multiply_empty_expansion_by_x_does_nothing()
{
    zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>> in{};

    zest::zt::RealZernikeExpansionNormalGeo out(4);
    std::ranges::fill(out.flatten(), std::array<double, 2>{2.0, 3.0});

    zdm::zebra::detail::ZernikeRecursionData coeff_data(4);

    zdm::zebra::detail::multiply_by_x(coeff_data, in, out);

    bool success = true;
    for (const auto& element : out.flatten())
    {
        success = success && (element == std::array<double, 2>{2.0, 3.0});
        if (!success) break;
    }

    if (!success)
    for (const auto& element : out.flatten())
        std::printf("{%f, %f}\n", element[0], element[1]);
    
    return success;
}

bool multiply_empty_expansion_by_y_does_nothing()
{
    zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>> in{};

    zest::zt::RealZernikeExpansionNormalGeo out(4);
    std::ranges::fill(out.flatten(), std::array<double, 2>{2.0, 3.0});

    zdm::zebra::detail::ZernikeRecursionData coeff_data(4);

    zdm::zebra::detail::multiply_by_y(coeff_data, in, out);

    bool success = true;
    for (const auto& element : out.flatten())
    {
        success = success && (element == std::array<double, 2>{2.0, 3.0});
        if (!success) break;
    }

    if (!success)
    for (const auto& element : out.flatten())
        std::printf("{%f, %f}\n", element[0], element[1]);
    
    return success;
}

bool multiply_empty_expansion_by_z_does_nothing()
{
    zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>> in{};

    zest::zt::RealZernikeExpansionNormalGeo out(4);
    std::ranges::fill(out.flatten(), std::array<double, 2>{2.0, 3.0});

    zdm::zebra::detail::ZernikeRecursionData coeff_data(4);

    zdm::zebra::detail::multiply_by_z(coeff_data, in, out);

    bool success = true;
    for (const auto& element : out.flatten())
    {
        success = success && (element == std::array<double, 2>{2.0, 3.0});
        if (!success) break;
    }

    if (!success)
    for (const auto& element : out.flatten())
        std::printf("{%f, %f}\n", element[0], element[1]);
    
    return success;
}

bool multiply_empty_expansion_by_r2_does_nothing()
{
    zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>> in{};

    zest::zt::RealZernikeExpansionNormalGeo out(4);
    std::ranges::fill(out.flatten(), std::array<double, 2>{2.0, 3.0});

    zdm::zebra::detail::ZernikeRecursionData coeff_data(4);

    zdm::zebra::detail::multiply_by_r2(coeff_data, in, out);

    bool success = true;
    for (const auto& element : out.flatten())
    {
        success = success && (element == std::array<double, 2>{2.0, 3.0});
        if (!success) break;
    }

    if (!success)
    for (const auto& element : out.flatten())
        std::printf("{%f, %f}\n", element[0], element[1]);
    
    return success;
}

template <std::size_t in_order_param>
bool multiply_Z000_by_x_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(0,0,0) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_x(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 1 && l == 1 && m == 1)
                    success = success
                            && is_close(out(n,l,m), {-1.0/std::sqrt(5.0), 0.0}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
bool multiply_Z000_by_y_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(0,0,0) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_y(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 1 && l == 1 && m == 1)
                    success = success
                            && is_close(out(n,l,m), {0.0, -1.0/std::sqrt(5.0)}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
bool multiply_Z000_by_z_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(0,0,0) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_z(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 1 && l == 1 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {1.0/std::sqrt(5.0), 0.0}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
bool multiply_Z000_by_r2_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 2;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(0,0,0) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_r2(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 0 && l == 0 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {3.0/5.0, 0.0}, tol);
                else if (n == 2 && l == 0 && m == 0)
                    success = success
                            && is_close(out(n,l,m),
                                {2.0*std::sqrt(3.0)/(5.0*std::sqrt(7.0)), 0.0}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z111_by_x_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,1) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_x(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 0 && l == 0 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {-1.0/std::sqrt(5.0), 0.0}, tol);
                else if (n == 2 && l == 0 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {-2.0/std::sqrt(105.0), 0.0}, tol);
                else if (n == 2 && l == 2 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {1.0/std::sqrt(21.0), 0.0}, tol);
                else if (n == 2 && l == 2 && m == 2)
                    success = success
                            && is_close(out(n,l,m), {-1.0/std::sqrt(7.0), 0.0}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z111_by_y_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,1) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_y(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 2 && m == 2)
                    success = success
                            && is_close(out(n,l,m), {0.0, -1.0/std::sqrt(7.0)}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z111_by_z_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,1) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_z(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 2 && m == 1)
                    success = success
                            && is_close(out(n,l,m), {1.0/std::sqrt(7.0), 0.0}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z11m1_by_x_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,1) = {0.0, 1.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_x(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 2 && m == 2)
                    success = success
                            && is_close(out(n,l,m), {0.0, -1.0/std::sqrt(7.0)}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z11m1_by_y_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,1) = {0.0, 1.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_y(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 0 && l == 0 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {-1.0/std::sqrt(5.0), 0.0}, tol);
                else if (n == 2 && l == 0 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {-2.0/std::sqrt(105.0), 0.0}, tol);
                else if (n == 2 && l == 2 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {1.0/std::sqrt(21.0), 0.0}, tol);
                else if (n == 2 && l == 2 && m == 2)
                    success = success
                            && is_close(out(n,l,m), {1.0/std::sqrt(7.0), 0.0}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z11m1_by_z_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,1) = {0.0, 1.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_z(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 2 && m == 1)
                    success = success
                            && is_close(out(n,l,m), {0.0, 1.0/std::sqrt(7.0)}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z110_by_x_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,0) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_x(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 2 && m == 1)
                    success = success
                            && is_close(out(n,l,m), {-1.0/std::sqrt(7.0), 0.0}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z110_by_y_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,0) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_y(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 2 && l == 2 && m == 1)
                    success = success
                            && is_close(out(n,l,m), {0.0, -1.0/std::sqrt(7.0)}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

template <std::size_t in_order_param>
    requires (in_order_param > 1)
bool multiply_Z110_by_z_is_correct_for_order()
{
    constexpr std::size_t in_order = in_order_param;
    constexpr std::size_t out_order = in_order_param + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    
    in(1,1,0) = {1.0, 0.0};

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);

    zdm::zebra::detail::multiply_by_z(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                if (n == 0 && l == 0 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {1.0/std::sqrt(5.0), 0.0}, tol);
                else if (n == 2 && l == 0 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {2.0/std::sqrt(105.0), 0.0}, tol);
                else if (n == 2 && l == 2 && m == 0)
                    success = success
                            && is_close(out(n,l,m), {2.0/std::sqrt(21.0), 0.0}, tol);
                else
                    success = success
                            && is_close(out(n,l,m), {0.0, 0.0}, tol);
            }
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf("(%lu, %lu, %lu): {%f, %f}\n", n, l, m, out(n,l,m)[0], out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

bool multiply_unit_input_by_x_is_correct_for_order(std::size_t in_order)
{
    const std::size_t out_order = in_order + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    zest::zt::RealZernikeExpansionNormalGeo reference_out(out_order);

    for (std::size_t n = 0; n < in.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            in(n, l, 0) = {1.0, 0.0};
            for (std::size_t m = 1; m <= l; ++m)
                in(n, l, m) = {1.0, 1.0};
        }
    }

    std::ranges::copy(in.flatten(), reference_out.flatten().begin());

    zest::zt::GLQTransformerNormalGeo transformer(out_order);
    zest::zt::BallGLQGrid reference_grid
        = transformer.backward_transform(reference_out, out_order);

    auto gen_x = [](double lon, double colat, double r){
        return r*std::sin(colat)*std::cos(lon);
    };
    zest::zt::BallGLQGridPoints points(out_order);
    zest::zt::BallGLQGrid x = points.generate_values(gen_x, out_order);

    for (std::size_t i = 0; i < reference_grid.flatten().size(); ++i)
        reference_grid.flatten()[i] *= x.flatten()[i];
    
    transformer.forward_transform(reference_grid, reference_out);

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);
    zdm::zebra::detail::multiply_by_x(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                success = success
                        && is_close(out(n,l,m), reference_out(n,l,m), tol);
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf(
                    "(%lu, %lu, %lu): {%f, %f} {%f, %f}\n", n, l, m,
                    out(n,l,m)[0], out(n,l,m)[1],
                    reference_out(n,l,m)[0], reference_out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

bool multiply_random_input_by_x_is_correct_for_order(std::size_t in_order)
{
    const std::size_t out_order = in_order + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    zest::zt::RealZernikeExpansionNormalGeo reference_out(out_order);

    std::mt19937 rng(29837490);
    std::uniform_real_distribution dist;

    for (std::size_t n = 0; n < in.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            in(n, l, 0) = {dist(rng), 0.0};
            for (std::size_t m = 1; m <= l; ++m)
                in(n, l, m) = {dist(rng), dist(rng)};
        }
    }

    std::ranges::copy(in.flatten(), reference_out.flatten().begin());

    zest::zt::GLQTransformerNormalGeo transformer(out_order);
    zest::zt::BallGLQGrid reference_grid
        = transformer.backward_transform(reference_out, out_order);

    auto gen_x = [](double lon, double colat, double r){
        return r*std::sin(colat)*std::cos(lon);
    };
    zest::zt::BallGLQGridPoints points(out_order);
    zest::zt::BallGLQGrid x = points.generate_values(gen_x, out_order);

    for (std::size_t i = 0; i < reference_grid.flatten().size(); ++i)
        reference_grid.flatten()[i] *= x.flatten()[i];
    
    transformer.forward_transform(reference_grid, reference_out);

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);
    zdm::zebra::detail::multiply_by_x(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                success = success
                        && is_close(out(n,l,m), reference_out(n,l,m), tol);
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf(
                    "(%lu, %lu, %lu): {%f, %f} {%f, %f}\n", n, l, m,
                    out(n,l,m)[0], out(n,l,m)[1],
                    reference_out(n,l,m)[0], reference_out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

bool multiply_random_input_by_y_is_correct_for_order(std::size_t in_order)
{
    const std::size_t out_order = in_order + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    zest::zt::RealZernikeExpansionNormalGeo reference_out(out_order);

    std::mt19937 rng(29837490);
    std::uniform_real_distribution dist;

    for (std::size_t n = 0; n < in.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            in(n, l, 0) = {dist(rng), 0.0};
            for (std::size_t m = 1; m <= l; ++m)
                in(n, l, m) = {dist(rng), dist(rng)};
        }
    }

    std::ranges::copy(in.flatten(), reference_out.flatten().begin());

    zest::zt::GLQTransformerNormalGeo transformer(out_order);
    zest::zt::BallGLQGrid reference_grid
        = transformer.backward_transform(reference_out, out_order);

    auto gen_y = [](double lon, double colat, double r){
        return r*std::sin(colat)*std::sin(lon);
    };
    zest::zt::BallGLQGridPoints points(out_order);
    zest::zt::BallGLQGrid y = points.generate_values(gen_y, out_order);

    for (std::size_t i = 0; i < reference_grid.flatten().size(); ++i)
        reference_grid.flatten()[i] *= y.flatten()[i];
    
    transformer.forward_transform(reference_grid, reference_out);

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);
    zdm::zebra::detail::multiply_by_y(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                success = success
                        && is_close(out(n,l,m), reference_out(n,l,m), tol);
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf(
                    "(%lu, %lu, %lu): {%f, %f} {%f, %f}\n", n, l, m,
                    out(n,l,m)[0], out(n,l,m)[1],
                    reference_out(n,l,m)[0], reference_out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

bool multiply_unit_input_by_z_is_correct_for_order(std::size_t in_order)
{
    const std::size_t out_order = in_order + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    zest::zt::RealZernikeExpansionNormalGeo reference_out(out_order);

    std::mt19937 rng(29837490);
    std::uniform_real_distribution dist;

    for (std::size_t n = 0; n < in.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            in(n, l, 0) = {1.0, 0.0};
            for (std::size_t m = 1; m <= l; ++m)
                in(n, l, m) = {1.0, 1.0};
        }
    }

    std::ranges::copy(in.flatten(), reference_out.flatten().begin());

    zest::zt::GLQTransformerNormalGeo transformer(out_order);
    zest::zt::BallGLQGrid reference_grid
        = transformer.backward_transform(reference_out, out_order);

    auto gen_z = []([[maybe_unused]] double lon, double colat, double r){
        return r*std::cos(colat);
    };
    zest::zt::BallGLQGridPoints points(out_order);
    zest::zt::BallGLQGrid z = points.generate_values(gen_z, out_order);

    for (std::size_t i = 0; i < reference_grid.flatten().size(); ++i)
        reference_grid.flatten()[i] *= z.flatten()[i];
    
    transformer.forward_transform(reference_grid, reference_out);

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);
    zdm::zebra::detail::multiply_by_z(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                success = success
                        && is_close(out(n,l,m), reference_out(n,l,m), tol);
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf(
                    "(%lu, %lu, %lu): {%f, %f} {%f, %f}\n", n, l, m,
                    out(n,l,m)[0], out(n,l,m)[1],
                    reference_out(n,l,m)[0], reference_out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

bool multiply_random_input_by_z_is_correct_for_order(std::size_t in_order)
{
    const std::size_t out_order = in_order + 1;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    zest::zt::RealZernikeExpansionNormalGeo reference_out(out_order);

    std::mt19937 rng(29837490);
    std::uniform_real_distribution dist;

    for (std::size_t n = 0; n < in.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            in(n, l, 0) = {dist(rng), 0.0};
            for (std::size_t m = 1; m <= l; ++m)
                in(n, l, m) = {dist(rng), dist(rng)};
        }
    }

    std::ranges::copy(in.flatten(), reference_out.flatten().begin());

    zest::zt::GLQTransformerNormalGeo transformer(out_order);
    zest::zt::BallGLQGrid reference_grid
        = transformer.backward_transform(reference_out, out_order);

    auto gen_z = []([[maybe_unused]] double lon, double colat, double r){
        return r*std::cos(colat);
    };
    zest::zt::BallGLQGridPoints points(out_order);
    zest::zt::BallGLQGrid z = points.generate_values(gen_z, out_order);

    for (std::size_t i = 0; i < reference_grid.flatten().size(); ++i)
        reference_grid.flatten()[i] *= z.flatten()[i];
    
    transformer.forward_transform(reference_grid, reference_out);

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);
    zdm::zebra::detail::multiply_by_z(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                success = success
                        && is_close(out(n,l,m), reference_out(n,l,m), tol);
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf(
                    "(%lu, %lu, %lu): {%f, %f} {%f, %f}\n", n, l, m,
                    out(n,l,m)[0], out(n,l,m)[1],
                    reference_out(n,l,m)[0], reference_out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

bool multiply_random_input_by_r2_is_correct_for_order(std::size_t in_order)
{
    const std::size_t out_order = in_order + 2;
    zest::zt::RealZernikeExpansionNormalGeo in(in_order);
    zest::zt::RealZernikeExpansionNormalGeo out(out_order);
    zest::zt::RealZernikeExpansionNormalGeo reference_out(out_order);

    std::mt19937 rng(29837490);
    std::uniform_real_distribution dist;

    for (std::size_t n = 0; n < in.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            in(n, l, 0) = {dist(rng), 0.0};
            for (std::size_t m = 1; m <= l; ++m)
                in(n, l, m) = {dist(rng), dist(rng)};
        }
    }

    std::ranges::copy(in.flatten(), reference_out.flatten().begin());

    zest::zt::GLQTransformerNormalGeo transformer(out_order);
    zest::zt::BallGLQGrid reference_grid
        = transformer.backward_transform(reference_out, out_order);

    auto gen_r2 = [](
        [[maybe_unused]] double lon, [[maybe_unused]] double colat, double r)
    { return r*r; };
    zest::zt::BallGLQGridPoints points(out_order);
    zest::zt::BallGLQGrid r2 = points.generate_values(gen_r2, out_order);

    for (std::size_t i = 0; i < reference_grid.flatten().size(); ++i)
        reference_grid.flatten()[i] *= r2.flatten()[i];
    
    transformer.forward_transform(reference_grid, reference_out);

    zdm::zebra::detail::ZernikeRecursionData coeff_data(out_order);
    zdm::zebra::detail::multiply_by_r2(coeff_data, in, out);

    constexpr double tol = 1.0e-14;
    bool success = true;
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
                success = success
                        && is_close(out(n,l,m), reference_out(n,l,m), tol);
        }
    }

    if (!success)
    for (std::size_t n = 0; n < out.order(); ++n)
    {
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            for (std::size_t m = 0; m <= l; ++m)
            {
                std::printf(
                    "(%lu, %lu, %lu): {%f, %f} {%f, %f}\n", n, l, m,
                    out(n,l,m)[0], out(n,l,m)[1],
                    reference_out(n,l,m)[0], reference_out(n,l,m)[1]);
            }
        }
    }
    
    return success;
}

int main()
{
    assert(multiply_empty_expansion_by_x_does_nothing());
    assert(multiply_empty_expansion_by_y_does_nothing());
    assert(multiply_empty_expansion_by_z_does_nothing());
    assert(multiply_empty_expansion_by_r2_does_nothing());

    assert(multiply_Z000_by_x_is_correct_for_order<1>());
    assert(multiply_Z000_by_y_is_correct_for_order<1>());
    assert(multiply_Z000_by_z_is_correct_for_order<1>());
    assert(multiply_Z000_by_r2_is_correct_for_order<1>());

    assert(multiply_Z000_by_x_is_correct_for_order<2>());
    assert(multiply_Z000_by_y_is_correct_for_order<2>());
    assert(multiply_Z000_by_z_is_correct_for_order<2>());

    assert(multiply_Z000_by_r2_is_correct_for_order<3>());

    assert(multiply_Z111_by_x_is_correct_for_order<2>());
    assert(multiply_Z111_by_y_is_correct_for_order<2>());
    assert(multiply_Z111_by_z_is_correct_for_order<2>());
    assert(multiply_Z11m1_by_x_is_correct_for_order<2>());
    assert(multiply_Z11m1_by_y_is_correct_for_order<2>());
    assert(multiply_Z11m1_by_z_is_correct_for_order<2>());
    assert(multiply_Z110_by_x_is_correct_for_order<2>());
    assert(multiply_Z110_by_y_is_correct_for_order<2>());
    assert(multiply_Z110_by_z_is_correct_for_order<2>());

    assert(multiply_Z111_by_x_is_correct_for_order<3>());
    assert(multiply_Z111_by_y_is_correct_for_order<3>());
    assert(multiply_Z111_by_z_is_correct_for_order<3>());
    assert(multiply_Z11m1_by_x_is_correct_for_order<3>());
    assert(multiply_Z11m1_by_y_is_correct_for_order<3>());
    assert(multiply_Z11m1_by_z_is_correct_for_order<3>());
    assert(multiply_Z110_by_x_is_correct_for_order<3>());
    assert(multiply_Z110_by_y_is_correct_for_order<3>());
    assert(multiply_Z110_by_z_is_correct_for_order<3>());

    assert(multiply_unit_input_by_x_is_correct_for_order(10));
    assert(multiply_unit_input_by_z_is_correct_for_order(10));

    assert(multiply_random_input_by_x_is_correct_for_order(10));
    assert(multiply_random_input_by_y_is_correct_for_order(10));
    assert(multiply_random_input_by_z_is_correct_for_order(10));
    assert(multiply_random_input_by_r2_is_correct_for_order(10));

    assert(multiply_random_input_by_r2_is_correct_for_order(1));
}
