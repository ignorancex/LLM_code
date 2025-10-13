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
#include "zonal_glq_transformer.hpp"

constexpr bool is_close(double a, double b, double tol)
{
    return std::fabs(a - b) < tol;
}

template <typename GridLayout, zest::st::SHNorm sh_norm_param>
bool test_zonal_glq_forward_transform_expands_Y00(std::size_t order)
{
    auto function = []([[maybe_unused]] double lon, [[maybe_unused]] double colat)
    {
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm;
    };
    
    zdm::zebra::ZonalGLQTransformer<sh_norm_param, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (l == 0)
        {
            if (is_close(expansion[l], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[l], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm sh_norm_param>
bool test_zonal_glq_forward_transform_expands_Y10(std::size_t order)
{
    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(3.0)*z;
    };
    
    zdm::zebra::ZonalGLQTransformer<sh_norm_param, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (l == 1)
        {
            if (is_close(expansion[l], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[l], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm sh_norm_param>
bool test_zonal_glq_forward_transform_expands_Y20(std::size_t order)
{
    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*0.5*std::sqrt(5.0)*(3.0*z*z - 1.0);
    };
    
    zdm::zebra::ZonalGLQTransformer<sh_norm_param, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (l == 2)
        {
            if (is_close(expansion[l], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[l], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm sh_norm_param>
bool test_zonal_glq_forward_transform_expands_Y21(std::size_t order)
{
    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(15.0)*std::sqrt(1.0 - z*z)*z*std::cos(lon);
    };
    
    zdm::zebra::ZonalGLQTransformer<sh_norm_param, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (is_close(expansion[l], 0.0, tol))
            success = success && true;
        else
            success = success && false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm sh_norm_param>
bool test_zonal_glq_forward_transform_expands_Y30(std::size_t order)
{
    auto function = []([[maybe_unused]] double lon, double colat)
    {
        const double z = std::cos(colat);

        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*0.5*std::sqrt(7.0)*(5.0*z*z - 3.0)*z;
    };
    
    zdm::zebra::ZonalGLQTransformer<sh_norm_param, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double reference_coeff = 1.0;
    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (l == 3)
        {
            if (is_close(expansion[l], reference_coeff, tol))
                success = success && true;
            else
                success = success && false;
        }
        else
        {
            if (is_close(expansion[l], 0.0, tol))
                success = success && true;
            else
                success = success && false;
        }
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm sh_norm_param>
bool test_zonal_glq_forward_transform_expands_Y31(std::size_t order)
{
    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(21.0/8.0)*std::sqrt(1.0 - z*z)*(5.0*z*z - 1.0)*std::cos(lon);
    };
    
    zdm::zebra::ZonalGLQTransformer<sh_norm_param, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (is_close(expansion[l], 0.0, tol))
            success = success && true;
        else
            success = success && false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}

template <typename GridLayout, zest::st::SHNorm sh_norm_param>
bool test_zonal_glq_forward_transform_expands_Y4m3(std::size_t order)
{
    auto function = [](double lon, double colat)
    {
        const double z = std::cos(colat);
        constexpr double shnorm = (sh_norm_param == zest::st::SHNorm::qm) ?
            0.5*std::numbers::inv_sqrtpi : 1.0;
        return shnorm*std::sqrt(315.0/8.0)*std::sqrt(1.0 - z*z)*(1.0 - z*z)*z*std::sin(3.0*lon);
    };
    
    zdm::zebra::ZonalGLQTransformer<sh_norm_param, GridLayout> transformer(order);

    zest::st::SphereGLQGridPoints<GridLayout> points{};
    zest::st::SphereGLQGrid<double, GridLayout> grid = points.generate_values(function, order);
    auto expansion = transformer.forward_transform(grid, order);

    constexpr double tol = 1.0e-13;

    bool success = true;
    for (std::size_t l = 0; l < expansion.size(); ++l)
    {
        if (is_close(expansion[l], 0.0, tol))
            success = success && true;
        else
            success = success && false;
    }

    if (!success)
    {
        for (std::size_t l = 0; l < order; ++l)
                std::printf("%lu %f\n", l, expansion[l]);
    }
    return success;
}


template <typename GridLayout, zest::st::SHNorm sh_norm_param>
void test_glq(std::size_t order)
{
    assert((test_zonal_glq_forward_transform_expands_Y00<GridLayout, sh_norm_param>(order)));
    assert((test_zonal_glq_forward_transform_expands_Y10<GridLayout, sh_norm_param>(order)));
    assert((test_zonal_glq_forward_transform_expands_Y20<GridLayout, sh_norm_param>(order)));
    assert((test_zonal_glq_forward_transform_expands_Y21<GridLayout, sh_norm_param>(order)));
    assert((test_zonal_glq_forward_transform_expands_Y30<GridLayout, sh_norm_param>(order)));
    assert((test_zonal_glq_forward_transform_expands_Y31<GridLayout, sh_norm_param>(order)));
    assert((test_zonal_glq_forward_transform_expands_Y4m3<GridLayout, sh_norm_param>(order)));
}

int main()
{
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::geo>(6);
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::geo>(7);
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::geo>(8);
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::geo>(9);

    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::geo>(6);
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::geo>(7);
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::geo>(8);
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::geo>(9);

    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::qm>(6);
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::qm>(7);
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::qm>(8);
    test_glq<zest::st::LatLonLayout<>, zest::st::SHNorm::qm>(9);

    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::qm>(6);
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::qm>(7);
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::qm>(8);
    test_glq<zest::st::LonLatLayout<>, zest::st::SHNorm::qm>(9);
}