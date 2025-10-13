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
#include <random>

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

void relative_error(
    zest::MDSpan<double, 2> err, zest::MDSpan<const double, 2> a, zest::MDSpan<const double, 2> b)
{
    for (std::size_t i = 0; i < err.extents()[0]; ++i)
    {
        for (std::size_t j = 0; j < err.extents()[1]; ++j)
        {
            err(i, j) = std::fabs(1.0 - a(i, j)/b(i, j));
        }
    }
}

template <typename DistType, typename RespType>
void zebra_evaluate(
    DistType&& dist, RespType&& resp, std::size_t dist_order,
    std::size_t resp_order, std::span<const std::array<double, 3>> offsets,
    std::span<const double> shells, std::span<const double> rotation_angles, zest::MDSpan<double, 2> out)
{
    zest::zt::RealZernikeExpansionNormalGeo distribution
        = zest::zt::ZernikeTransformerNormalGeo<>(dist_order).transform(
                dist, 1.0, dist_order);

    std::vector<std::array<double, 2>> response_buffer(
        shells.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(resp_order));
    zdm::SHExpansionVectorSpan<std::array<double, 2>>
    response(response_buffer.data(), {shells.size()}, resp_order);
    zdm::zebra::ResponseTransformer(resp_order).transform(resp, shells, response);

    zdm::zebra::AnisotropicAngleIntegrator(dist_order, resp_order).integrate(
            distribution, response, offsets, rotation_angles, shells, out);
}

template <typename DistType, typename RespType>
void zebra_convergence(
    DistType&& dist, RespType&& resp, std::span<const std::array<double, 3>> offsets, std::span<const double> shells, std::span<const double> rotation_angles)
{
    constexpr std::size_t dist_order = 200;
    constexpr std::size_t resp_order = 800;
    const std::vector<std::size_t> dist_orders = {20, 50, 100, 150};
    const std::vector<std::size_t> resp_orders = {30, 40, 50, 60, 70, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 400, 480, 640};

    std::vector<double> reference_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> reference(
            reference_buffer.data(), {offsets.size(), shells.size()});
    
    std::printf("Computing reference orders: (%lu, %lu)\n", dist_order, resp_order);
    zebra_evaluate(
            dist, resp, dist_order, resp_order, offsets, shells, rotation_angles, reference);

    std::vector<double> test_buffer(
            dist_orders.size()*resp_orders.size()*offsets.size()*shells.size());
    zest::MDSpan<double, 4> test(
            test_buffer.data(), {dist_orders.size(), resp_orders.size(), offsets.size(), shells.size()});
    
    for (std::size_t i = 0; i < dist_orders.size(); ++i)
    {
        for (std::size_t j = 0; j < resp_orders.size(); ++j)
        {
            std::printf("Computing orders: %lu, %lu\n", dist_orders[i], resp_orders[j]);
            zebra_evaluate(
                    dist, resp, dist_orders[i], resp_orders[j], offsets, shells, rotation_angles, test(i, j));
        }
    }

    std::vector<double> relative_error_buffer(
            dist_orders.size()*resp_orders.size()*offsets.size()*shells.size());
    zest::MDSpan<double, 4> relative_errors(
            relative_error_buffer.data(), {dist_orders.size(), resp_orders.size(), offsets.size(), shells.size()});
    for (std::size_t i = 0; i < dist_orders.size(); ++i)
    {
        for (std::size_t j = 0; j < resp_orders.size(); ++j)
            relative_error(relative_errors(i, j), test(i, j), reference);
    }
    
    for (std::size_t i = 0; i < dist_orders.size(); ++i)
    {
        for (std::size_t j = 0; j < resp_orders.size(); ++j)
        {
            std::printf("\n orders: %lu, %lu\n", dist_orders[i], resp_orders[j]);
            for (std::size_t k = 0; k < offsets.size(); ++k)
            {
                for (std::size_t l = 0; l < shells.size(); ++l)
                    std::printf("%.16e ", relative_errors(i, j, k, l));
                std::printf("\n");
            }
        }
    }
}


inline double smooth_step(double x, double slope)
{
    return 0.5*(1.0 + std::tanh(slope*x));
}

int main()
{
    [[maybe_unused]] auto aniso_gaussian = [](const std::array<double, 3>& v)
    {
        constexpr std::array<std::array<double, 3>, 3> sigma = {
            std::array<double, 3>{3.0, 1.4, 0.5},
            std::array<double, 3>{1.4, 0.3, 2.1},
            std::array<double, 3>{0.5, 2.1, 1.7}
        };
        return std::exp(-0.5*quadratic_form(sigma, v));
    };

    [[maybe_unused]] auto smooth_dots = [](double shell, double longitude, double colatitude)
    {
        constexpr double norm = (4000.0/(3.0*33.0*33.0));

        const double t = std::cos(colatitude);
        const double t2 = t*t;
        const double u2 = (1.0 - t)*(1.0 + t);
        const double u4 = u2*u2;
        const double Y64 = norm*u4*(11.0*t2 - 1.0)*std::cos(4.0*longitude);

        constexpr double rate = 2.0;
        const double surface = 1.0 - std::exp(rate*(Y64 - 1.0));

        constexpr double slope = 10.0;
        return smooth_step(shell*(1.0/1.5) - surface, slope);
    };

    constexpr std::size_t num_offsets = 10;
    constexpr std::size_t num_shells = 10;

    constexpr double offset_len = 0.5;
    constexpr double max_shell = 1.0 + offset_len;

    std::mt19937 gen;
    std::uniform_real_distribution rng_dist{0.0, 1.0};

    std::vector<std::array<double, 3>> offsets(num_offsets);
    std::vector<double> rotation_angles(num_offsets);
    for (std::size_t i = 0; i < num_offsets; ++i)
    {
        const double ct = 2.0*rng_dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*rng_dist(gen);
        offsets[i] = {
            offset_len*st*std::cos(az), offset_len*st*std::sin(az), ct
        };
        rotation_angles[i] = 2.0*std::numbers::pi*rng_dist(gen);
    }

    std::vector<double> shells(num_shells);
    for (std::size_t i = 0; i < num_shells; ++i)
        shells[i] = double(i)*max_shell/double(num_shells - 1);
    
    zebra_convergence(aniso_gaussian, smooth_dots, offsets, shells, rotation_angles);
}
