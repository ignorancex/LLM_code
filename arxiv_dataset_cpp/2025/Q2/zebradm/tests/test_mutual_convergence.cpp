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
#include "radon_integrator.hpp"
#include "zebra_angle_integrator.hpp"

#include "coordinate_transforms.hpp"

double quadratic_form(
    const std::array<std::array<double, 3>, 3>& arr,
    const std::array<double, 3>& vec)
{
    double res = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
            res += vec[i]*arr[i][j]*vec[j];
    }

    return res;
}

template <typename DistType>
void test_mutual_convergence_isotropic(
    DistType&& dist, std::span<const std::array<double, 3>> offsets,
    std::span<const double> shells)
{
    std::vector<double> integrator_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> integrator_test(
            integrator_test_buffer.data(), {offsets.size(), shells.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(dist, offsets, shells, 0.0, 1.0e-9, integrator_test);

    std::vector<double> transformer_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> transformer_test(
            transformer_test_buffer.data(), {offsets.size(), shells.size()});

    constexpr std::size_t order = 200;
    zest::zt::RealZernikeExpansionNormalGeo distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(order).transform(
                dist, 1.0, order);

    zdm::zebra::IsotropicAngleIntegrator(order).integrate(
            distribution, offsets, shells, transformer_test);

    std::printf("integrator\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", integrator_test(i, j));
        }
        std::printf("\n");
    }

    std::printf("\ntransformer\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", transformer_test(i, j));
        }
        std::printf("\n");
    }

    std::printf("\nrelative error\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", 1.0 - integrator_test(i, j)/transformer_test(i, j));
        }
        std::printf("\n");
    }
    std::printf("\n");
}

template <typename DistType>
void test_mutual_convergence_transverse_isotropic(
    DistType&& dist, std::span<const std::array<double, 3>> offsets,
    std::span<const double> shells)
{
    std::vector<std::array<double, 2>> integrator_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<std::array<double, 2>, 2> integrator_test(
            integrator_test_buffer.data(), {offsets.size(), shells.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate_transverse(dist, offsets, shells, 0.0, 1.0e-9, integrator_test);

    std::vector<std::array<double, 2>> transformer_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<std::array<double, 2>, 2> transformer_test(
            transformer_test_buffer.data(), {offsets.size(), shells.size()});

    constexpr std::size_t order = 200;
    zest::zt::RealZernikeExpansionNormalGeo distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(order).transform(
                dist, 1.0, order);

    zdm::zebra::IsotropicTransverseAngleIntegrator(order).integrate(
            distribution, offsets, shells, transformer_test);

    std::printf("integrator\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("{%.16e, %.16e}", integrator_test(i, j)[0], integrator_test(i, j)[1]);
        }
        std::printf("\n");
    }

    std::printf("\ntransformer\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("{%.16e, %.16e}", transformer_test(i, j)[0], transformer_test(i, j)[1]);
        }
        std::printf("\n");
    }

    std::printf("\nrelative error\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("{%.16e, %.16e}", 1.0 - integrator_test(i, j)[0]/transformer_test(i, j)[0], 1.0 - integrator_test(i, j)[1]/transformer_test(i, j)[1]);
        }
        std::printf("\n");
    }
    std::printf("\n");
}



template <typename DistType, typename RespType>
void test_mutual_convergence_anisotropic(
    DistType&& dist, RespType&& resp, std::span<const std::array<double, 3>> offsets,
    std::span<const double> shells, std::span<const double> rotation_angles)
{
    std::vector<double> integrator_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> integrator_test(
            integrator_test_buffer.data(), {offsets.size(), shells.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(dist, resp, offsets, rotation_angles, shells, 0.0, 1.0e-7, integrator_test);

    std::vector<double> transformer_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> transformer_test(
            transformer_test_buffer.data(), {offsets.size(), shells.size()});

    constexpr std::size_t dist_order = 80;
    constexpr std::size_t resp_order = 100;
    zest::zt::RealZernikeExpansionNormalGeo distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(dist_order).transform(
                dist, 1.0, dist_order);

    std::vector<std::array<double, 2>> response_buffer(
        shells.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(resp_order));
    zdm::SHExpansionVectorSpan<std::array<double, 2>>
    response(response_buffer.data(), {shells.size()}, resp_order);
    zdm::zebra::ResponseTransformer(resp_order).transform(resp, shells, response);

    zdm::zebra::AnisotropicAngleIntegrator(dist_order, resp_order).integrate(
            distribution, response, offsets, rotation_angles, shells, transformer_test);

    std::printf("integrator\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", integrator_test(i, j));
        }
        std::printf("\n");
    }

    std::printf("\ntransformer\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", transformer_test(i, j));
        }
        std::printf("\n");
    }

    std::printf("\nrelative error\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", 1.0 - integrator_test(i, j)/transformer_test(i, j));
        }
        std::printf("\n");
    }
    std::printf("\n");
}

template <typename DistType, typename RespType>
void test_mutual_convergence_transverse_anisotropic(
    DistType&& dist, RespType&& resp, std::span<const std::array<double, 3>> offsets,
    std::span<const double> shells, std::span<const double> rotation_angles)
{
    std::vector<std::array<double, 2>> integrator_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<std::array<double, 2>, 2> integrator_test(
            integrator_test_buffer.data(), {offsets.size(), shells.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate_transverse(dist, resp, offsets, rotation_angles, shells, 0.0, 1.0e-7, integrator_test);

    std::vector<std::array<double, 2>> transformer_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<std::array<double, 2>, 2> transformer_test(
            transformer_test_buffer.data(), {offsets.size(), shells.size()});

    constexpr std::size_t dist_order = 80;
    constexpr std::size_t resp_order = 100;
    zest::zt::RealZernikeExpansionNormalGeo distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(dist_order).transform(
                dist, 1.0, dist_order);

    std::vector<std::array<double, 2>> response_buffer(
        shells.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(resp_order));
    zdm::SHExpansionVectorSpan<std::array<double, 2>>
    response(response_buffer.data(), {shells.size()}, resp_order);
    zdm::zebra::ResponseTransformer(resp_order).transform(resp, shells, response);

    zdm::zebra::AnisotropicTransverseAngleIntegrator(dist_order, resp_order)
        .integrate(
            distribution, response, offsets, rotation_angles, shells, transformer_test);

    std::printf("integrator\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("{%.16e, %.16e} ", integrator_test(i, j)[0], integrator_test(i, j)[1]);
        }
        std::printf("\n");
    }

    std::printf("\ntransformer\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("{%.16e, %.16e} ", transformer_test(i, j)[0], transformer_test(i, j)[1]);
        }
        std::printf("\n");
    }

    std::printf("\nrelative error\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("{%.16e, %.16e} ", 1.0 - integrator_test(i, j)[0]/transformer_test(i, j)[0], 1.0 - integrator_test(i, j)[1]/transformer_test(i, j)[1]);
        }
        std::printf("\n");
    }
    std::printf("\n");
}

int main()
{
    [[maybe_unused]] auto gaussian = [](const std::array<double, 3>& v){
        constexpr double disp = 0.4;
        const double speed = zdm::length(v);
        const double ratio = speed/disp;
        return std::exp(-ratio*ratio);
    };

    [[maybe_unused]] auto aniso_gaussian = [](const std::array<double, 3>& v)
    {
        constexpr std::array<std::array<double, 3>, 3> sigma = {
            std::array<double, 3>{3.0, 1.4, 0.5},
            std::array<double, 3>{1.4, 0.3, 2.1},
            std::array<double, 3>{0.5, 2.1, 1.7}
        };
        return std::exp(-0.5*quadratic_form(sigma, v));
    };

    [[maybe_unused]] auto test_response = []([[maybe_unused]] double shell, [[maybe_unused]] double lon, [[maybe_unused]] double colat) -> double
    {
        constexpr double slope = 10.0;
        const std::array<double, 3> dir
            = zdm::coordinates::spherical_to_cartesian_phys(lon, colat);
        return 0.5*(1.0 + std::tanh(slope*(dir[0] + (2.0/1.5)*shell - 1.0)));
    };

    [[maybe_unused]] auto smooth_exponential = [](double shell, double longitude, double colatitude)
    {
        static const std::array<double, 3> ref_dir
            = zdm::normalize(std::array<double, 3>{0.5, 0.5, 0.5});
        const std::array<double, 3> dir
            = zdm::coordinates::spherical_to_cartesian_phys(longitude, colatitude);
        constexpr double rate = 2.0;
        const double u2 = shell*shell;
        const double u4 = u2*u2;
        return (u4/(1 + u4))*std::exp(rate*(zdm::dot(dir, ref_dir)));
    };

    /*
    const std::vector<std::array<double, 3>> offsets = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {-0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {-0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {-0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {-0.5, 0.0, 0.0}, {0.0, -0.5, 0.0}, {0.0, 0.0, 0.5}
    };

    const std::vector<double> rotation_angles = {
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.5*std::numbers::pi, 0.5*std::numbers::pi, 0.5*std::numbers::pi, 0.5*std::numbers::pi, 0.5*std::numbers::pi,
        std::numbers::pi, std::numbers::pi, std::numbers::pi, std::numbers::pi, std::numbers::pi,
        1.5*std::numbers::pi, 1.5*std::numbers::pi, 1.5*std::numbers::pi, 1.5*std::numbers::pi, 1.5*std::numbers::pi
    };
    */

   const std::vector<std::array<double, 3>> offsets = {
        {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.5, 0.0, 0.0},
        {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}, {-0.5, 0.0, 0.0}
    };

    const std::vector<double> rotation_angles = {
        0.0*std::numbers::pi, 0.25*std::numbers::pi, 0.5*std::numbers::pi, 0.75*std::numbers::pi, 1.0*std::numbers::pi, 1.25*std::numbers::pi, 1.5*std::numbers::pi, 1.75*std::numbers::pi,
        0.0*std::numbers::pi, 0.25*std::numbers::pi, 0.5*std::numbers::pi, 0.75*std::numbers::pi, 1.0*std::numbers::pi, 1.25*std::numbers::pi, 1.5*std::numbers::pi, 1.75*std::numbers::pi
    };

    const std::vector<double> shells = {
        /*0.0, */0.15//, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35
    };

    //test_mutual_convergence_isotropic(gaussian, offsets, shells);
    //test_mutual_convergence_isotropic(aniso_gaussian, offsets, shells);

    //test_mutual_convergence_transverse_isotropic(gaussian, offsets, shells);
    //test_mutual_convergence_transverse_isotropic(aniso_gaussian, offsets, shells);

    //test_mutual_convergence_anisotropic(gaussian, smooth_exponential, offsets, shells, rotation_angles);
    test_mutual_convergence_anisotropic(aniso_gaussian, smooth_exponential, offsets, shells, rotation_angles);

    //test_mutual_convergence_transverse_anisotropic(
    //        gaussian, test_response, offsets, shells, rotation_angles);
    //test_mutual_convergence_transverse_anisotropic(
    //        aniso_gaussian, test_response, offsets, shells, rotation_angles);
}
