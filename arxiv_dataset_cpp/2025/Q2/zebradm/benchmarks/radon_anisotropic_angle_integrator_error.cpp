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

#include "zest/zernike_glq_transformer.hpp"

#include "zebra_angle_integrator.hpp"
#include "radon_integrator.hpp"
#include "nanobench.h"
#include "distributions.hpp"
#include "responses.hpp"

constexpr double relative_error(double test, double ref)
{
    return std::fabs(1.0 - test/ref);
}

void radon_angle_integrator_anisotropic_error(
    DistributionCartesian dist, const char* dist_name, Response resp, const char* resp_name, const std::span<const std::array<double, 3>> offsets, std::span<const double> rotation_angles, std::span<const double> shells, double relerr, std::size_t max_subdiv, zest::MDSpan<const double, 2> reference)
{
    std::vector<double> out_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> out(out_buffer.data(), {offsets.size(), shells.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(
            dist, resp, offsets, rotation_angles, shells, 0.0, relerr, out, max_subdiv);

    char fname[512] = {};
    std::sprintf(fname, "radon_anisotropic_angle_integrator_error_relative_%s_%s_%lu_%.2e.dat", dist_name, resp_name, max_subdiv, relerr);
    std::ofstream output{};
    output.open(fname);
    output << std::scientific << std::setprecision(16);
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            double error = 0.0;
            if (reference(i, j) != 0.0)
            {
                error = relative_error(out(i, j), reference(i, j));
            }
            output << error << ' ';
        }
        output << '\n';
    }
    output.close();
}

void run_errors(
    DistributionCartesian dist, const char* dist_name, Response resp, const char* resp_name, double relerr, std::size_t max_subdiv, double offset_len, std::size_t num_offsets, std::size_t num_shells)
{
    std::printf("Initalizing... ");
    std::mt19937 gen;
    std::uniform_real_distribution rng_dist{0.0, 1.0};

    std::vector<std::array<double, 3>> offsets(num_offsets);
    for (auto& element : offsets)
    {
        const double ct = 2.0*rng_dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*rng_dist(gen);
        element = {
            offset_len*st*std::cos(az), offset_len*st*std::sin(az), ct
        };
    }

    std::vector<double> rotation_angles(num_offsets);
    for (auto& element : rotation_angles)
        element = 2.0*std::numbers::pi*rng_dist(gen);

    std::vector<double> shells(num_shells);
    for (std::size_t i = 0; i < num_shells; ++i)
        shells[i] = double(i)*(offset_len + 1.0)/double(num_shells - 1);

    constexpr std::size_t reference_dist_order = 200;
    constexpr std::size_t reference_resp_order = 800;
    zest::zt::RealZernikeExpansion reference_distribution
        = zest::zt::ZernikeTransformerNormalGeo(reference_dist_order).transform(
            dist, 1.0, reference_dist_order);

    std::vector<std::array<double, 2>> response_buffer(
        shells.size()*zdm::SHExpansionSpan<std::array<double, 2>>::size(reference_resp_order));
    zdm::SHExpansionVectorSpan<std::array<double, 2>>
    reference_response(response_buffer.data(), {shells.size()}, reference_resp_order);
    zdm::zebra::ResponseTransformer(reference_resp_order).transform(resp, shells, reference_response);
    
    std::vector<double> reference_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> reference(
            reference_buffer.data(), {offsets.size(), shells.size()});
    
    zdm::zebra::AnisotropicAngleIntegrator integrator(reference_dist_order, reference_resp_order);
    integrator.integrate(reference_distribution, reference_response, offsets, rotation_angles, shells, reference);
    std::printf("Done\n Integrating... ");

    radon_angle_integrator_anisotropic_error(
            dist, dist_name, resp, resp_name, offsets, rotation_angles, shells, relerr, max_subdiv, reference);
    std::printf("Done\n");
}

template <typename Object>
struct Labeled
{
    Object object;
    const char* label;
};

int main([[maybe_unused]] int argc, char** argv)
{
    constexpr std::array<Labeled<DistributionCartesian>, 5> distributions = {
        Labeled<DistributionCartesian>{aniso_gaussian, "aniso_gaussian"},
        Labeled<DistributionCartesian>{four_gaussians, "four_gaussians"},
        Labeled<DistributionCartesian>{shm_plus_stream, "shm_plus_stream"},
        Labeled<DistributionCartesian>{shmpp_aniso, "shmpp_aniso"},
        Labeled<DistributionCartesian>{shmpp, "shmpp"}
    };

    constexpr std::array<Labeled<Response>, 2> responses = {
        Labeled<Response>{smooth_exponential, "smooth_exponential"},
        Labeled<Response>{smooth_dots, "smooth_dots"}
    };

    if (argc < 8)
        throw std::runtime_error(
            "Requires arguments:\n"
            "   dist_ind:       index of distribution {0,1,2,3,4}\n"
            "   resp_ind:       index of response {0,1}\n"
            "   offset_len:      length of offset vector (float)\n"
            "   num_offsets:     number of offset vectors (positive integer)\n"
            "   num_shells: number of shell values (positive integer)\n"
            "   relerr:         target relative error for integrator (float)\n"
            "   max_subdiv:     maximum number of subdivisions for integrator (positive integer)");

    const std::size_t dist_ind = atoi(argv[1]);
    const std::size_t resp_ind = atoi(argv[2]);
    const double offset_len = atof(argv[3]);
    const std::size_t num_offsets = atoi(argv[4]);
    const std::size_t num_shells = atoi(argv[5]);
    const double relerr = double(atof(argv[6]));
    const std::size_t max_subdiv = atoi(argv[7]);
    std::printf("dist_ind: %lu\nresp_ind: %lu\noffset_len: %.2f\nnum_offsets: %lu\nnum_speeds: %lu\nrelerr: %.0e\nmax_subdiv: %lu\n", dist_ind, resp_ind, offset_len, num_offsets, num_shells, relerr, max_subdiv);

    const Labeled<DistributionCartesian> dist = distributions[dist_ind];
    const Labeled<Response> resp = responses[resp_ind];

    run_errors(
            dist.object, dist.label, resp.object, resp.label, relerr, max_subdiv, offset_len, num_offsets, num_shells);
}
