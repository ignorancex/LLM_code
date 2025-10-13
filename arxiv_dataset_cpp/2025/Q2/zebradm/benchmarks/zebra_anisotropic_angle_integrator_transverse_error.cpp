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
#include <fstream>
#include <iomanip>

#include "zest/zernike_glq_transformer.hpp"

#include "zebra_angle_integrator.hpp"

#include "distributions.hpp"
#include "responses.hpp"

constexpr std::array<double, 2> relative_error(
    std::array<double, 2> test, std::array<double, 2> ref)
{
    return {
        std::fabs(1.0 - test[0]/ref[0]),
        std::fabs(1.0 - test[1]/ref[1])
    };
}

constexpr std::array<double, 2> absolute_error(
    std::array<double, 2> test, std::array<double, 2> ref)
{
    return {
        std::fabs(ref[0] - test[0]),
        std::fabs(ref[1] - test[1])
    };
}

void angle_integrator_error(
    std::span<const std::array<double, 3>> offsets, std::span<const double > rotation_angles, std::span<const double> shells, zest::MDSpan<const std::array<double, 2>, 2> reference, DistributionSpherical dist, const char* dist_name, Response resp, const char* resp_name, std::size_t dist_order, std::size_t resp_order, bool use_relative_error)
{
    zest::zt::RealZernikeExpansion distribution
        = zest::zt::ZernikeTransformerNormalGeo(dist_order).transform(
            dist, 1.0, dist_order);

    std::vector<std::array<double, 2>> response_buffer(
        shells.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(resp_order));
    zdm::SHExpansionVectorSpan<std::array<double, 2>>
    response(response_buffer.data(), {shells.size()}, resp_order);
    zdm::zebra::ResponseTransformer(resp_order).transform(resp, shells, response);

    std::vector<std::array<double, 2>> out_buffer(offsets.size()*shells.size());
    zest::MDSpan<std::array<double, 2>, 2> out(out_buffer.data(), {offsets.size(), shells.size()});

    zdm::zebra::AnisotropicTransverseAngleIntegrator integrator(dist_order, resp_order);
    integrator.integrate(distribution, response, offsets, rotation_angles, shells, out);

    char fname_nt[512] = {};
    char fname_t[512] = {};
    if (use_relative_error)
    {
        std::sprintf(fname_nt, "zebra_angle_integrator_nontransverse_error_relative_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
        std::sprintf(fname_t, "zebra_angle_integrator_transverse_error_relative_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
    }
    else
    {
        std::sprintf(fname_nt, "zebra_angle_integrator_nontransverse_error_absolute_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
        std::sprintf(fname_t, "zebra_angle_integrator_transverse_error_absolute_%s_dist-order_%lu_%s_resp-order_%lu.dat", dist_name, dist_order, resp_name, resp_order);
    }
    std::ofstream output_nt{};
    std::ofstream output_t{};
    output_nt.open(fname_nt);
    output_t.open(fname_t);
    output_nt << std::scientific << std::setprecision(16);
    output_t << std::scientific << std::setprecision(16);
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::array<double, 2> error = {};
            if (reference(i, j)[0] != 0.0 && reference(i, j)[1] != 0.0)
            {
                error = (use_relative_error) ?
                    relative_error(out(i, j), reference(i, j))
                    : absolute_error(out(i, j), reference(i, j));
            }
            output_nt << error[0] << ' ';
            output_t << error[1] << ' ';
        }
        output_nt << '\n';
        output_t << '\n';
    }
    output_nt.close();
    output_t.close();
}

void fill_reference(
    DistributionSpherical dist, Response resp, std::span<const std::array<double, 3>> offsets, std::span<const double > rotation_angles, std::span<const double> shells, zest::MDSpan<std::array<double, 2>, 2> reference)
{
    std::printf("Initalizing reference... ");
    std::fflush(stdout);
    constexpr std::size_t reference_dist_order = 200;
    constexpr std::size_t reference_resp_order = 800;

    zest::zt::RealZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerNormalGeo(reference_dist_order).transform(
            dist, 1.0, reference_dist_order);
    
    std::vector<std::array<double, 2>> reference_response_buffer(
        shells.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(reference_resp_order));
    zdm::SHExpansionVectorSpan<std::array<double, 2>>
    reference_response(reference_response_buffer.data(), {shells.size()}, reference_resp_order);
    zdm::zebra::ResponseTransformer(reference_resp_order).transform(resp, shells, reference_response);
    
    zdm::zebra::AnisotropicTransverseAngleIntegrator integrator(reference_dist_order, reference_resp_order);
    integrator.integrate(reference_distribution, reference_response, offsets, rotation_angles, shells, reference);
    std::printf("Done\n");
}

void angle_integrator_errors(
    DistributionSpherical dist, const char* dist_name, Response resp, const char* resp_name, bool relative_error, double offset_len, std::size_t num_offsets, std::size_t num_shells)
{
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
        shells[i] = double(i)*(offset_len + 1.0)/double(num_shells - 1);

    std::vector<std::array<double, 2>> reference_buffer(offsets.size()*shells.size());
    zest::MDSpan<std::array<double, 2>, 2> reference(
            reference_buffer.data(), {offsets.size(), shells.size()});

    fill_reference(dist, resp, offsets, rotation_angles, shells, reference);

    const std::vector<std::size_t> dist_orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180};
    const std::vector<std::size_t> resp_orders = {2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,50,60,70,80,90,100,120,140,160,180,200,240,280,320,400,480};

    for (std::size_t dist_order : dist_orders)
    {
        for (std::size_t resp_order : resp_orders)
        {
            std::printf("%lu %lu\n", dist_order, resp_order);
            angle_integrator_error(
                    offsets, rotation_angles, shells, reference, dist, dist_name, resp, resp_name, dist_order, resp_order, relative_error);
        }
    }
}

template <typename Object>
struct Labeled
{
    Object object;
    const char* label;
};

int main([[maybe_unused]] int argc, char** argv)
{
    constexpr bool relative_error = true;
    constexpr std::array<Labeled<DistributionSpherical>, 5> distributions = {
        Labeled<DistributionSpherical>{aniso_gaussian, "aniso_gaussian"},
        Labeled<DistributionSpherical>{four_gaussians, "four_gaussians"},
        Labeled<DistributionSpherical>{shm_plus_stream, "shm_plus_stream"},
        Labeled<DistributionSpherical>{shmpp_aniso, "shmpp_aniso"},
        Labeled<DistributionSpherical>{shmpp, "shmpp"}
    };

    constexpr std::array<Labeled<Response>, 2> responses = {
        Labeled<Response>{smooth_exponential, "smooth_exponential"},
        Labeled<Response>{smooth_dots, "smooth_dots"}
    };

    if (argc < 6)
        throw std::runtime_error(
            "Requires arguments:\n"
            "   dist_ind:       index of distribution {0,1,2,3,4}\n"
            "   resp_ind:       index of response {0,1}\n"
            "   offset_len:      length of offset vector (float)\n"
            "   num_offsets:     number of offset vectors (positive integer)\n"
            "   num_shells: number of shell values (positive integer)");

    const std::size_t dist_ind = atoi(argv[1]);
    const std::size_t resp_ind = atoi(argv[2]);
    const double offset_len = atof(argv[3]);
    const std::size_t num_offsets = atoi(argv[4]);
    const std::size_t num_shells = atoi(argv[5]);

    const Labeled<DistributionSpherical> dist = distributions[dist_ind];
    const Labeled<Response> resp = responses[resp_ind];
    std::printf("%s %s\n", dist.label, resp.label);

    angle_integrator_errors(
        dist.object, dist.label, resp.object, resp.label, relative_error, 
        offset_len, num_offsets, num_shells);
}
