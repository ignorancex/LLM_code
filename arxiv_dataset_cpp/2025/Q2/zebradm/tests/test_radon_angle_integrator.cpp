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

double angle_integrated_radon_shm(
    const std::array<double, 3>& offset, double shell, double disp_speed)
{
    constexpr double sqrt_pi = 1.0/std::numbers::inv_sqrtpi;
    const double offset_len = zdm::length(offset);
    const double erf_part
        = std::erf(std::min(1.0,shell + offset_len)/disp_speed)
        - std::erf((shell - offset_len)/disp_speed);
    
    const double exp_prefactor
        = (1.0 + offset_len - std::max(1.0 - offset_len, shell));
    
    const double inv_disp = 1.0/disp_speed;
    const double prefactor = std::numbers::pi*disp_speed*disp_speed/offset_len;

    return (2.0*std::numbers::pi)*prefactor*(0.5*sqrt_pi*disp_speed*erf_part - exp_prefactor*std::exp(-inv_disp*inv_disp));
}

void test_radon_integrator_is_accurate_for_shm()
{
    const double disp_speed = 0.4;
    auto shm_dist = [&](const std::array<double, 3>& velocity){
        const double speed = zdm::length(velocity);
        const double ratio = speed/disp_speed;
        return std::exp(-ratio*ratio);
    };

    std::vector<std::array<double, 3>> offsets = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    const std::vector<double> shells = {
        0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35
    };

    std::vector<double> shm_reference_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> shm_reference(
            shm_reference_buffer.data(), {offsets.size(), shells.size()});

    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            shm_reference(i, j) = angle_integrated_radon_shm(
                    offsets[i], shells[j], disp_speed);
        }
    }

    std::vector<double> shm_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> shm_test(
            shm_test_buffer.data(), {offsets.size(), shells.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(shm_dist, offsets, shells, 0.0, 1.0e-9, shm_test);

    std::printf("reference\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", shm_reference(i, j));
        }
        std::printf("\n");
    }

    std::printf("\ntest\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", shm_test(i, j));
        }
        std::printf("\n");
    }
    std::printf("\n");
}

void test_radon_integrator_resp_is_accurate_for_shm()
{
    const double disp_speed = 0.4;
    auto shm_dist = [&](const std::array<double, 3>& velocity){
        const double speed = zdm::length(velocity);
        const double ratio = speed/disp_speed;
        return std::exp(-ratio*ratio);
    };

    auto resp = [&](
        [[maybe_unused]] double shell, [[maybe_unused]] double azimuth, [[maybe_unused]] double colatitude)
    {
        return 1.0;
    };
    const std::vector<std::array<double, 3>> offsets = {
        {0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 0.5},
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };
    const std::vector<double> rotation_angles = {
        0.0, std::numbers::pi, 0.5*std::numbers::pi, 0.0, std::numbers::pi, 0.5*std::numbers::pi
    };

    const std::vector<double> shells = {
        0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35
    };

    std::vector<double> shm_reference_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> shm_reference(
            shm_reference_buffer.data(), {offsets.size(), shells.size()});

    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            shm_reference(i, j) = angle_integrated_radon_shm(
                    offsets[i], shells[j], disp_speed);
        }
    }

    std::vector<double> shm_test_buffer(offsets.size()*shells.size());
    zest::MDSpan<double, 2> shm_test(
            shm_test_buffer.data(), {offsets.size(), shells.size()});

    zdm::integrate::RadonAngleIntegrator integrator{};
    integrator.integrate(
            shm_dist, resp, offsets, rotation_angles, shells, 0.0, 1.0e-9, shm_test);

    std::printf("reference\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", shm_reference(i, j));
        }
        std::printf("\n");
    }

    std::printf("\ntest\n");
    for (std::size_t i = 0; i < offsets.size(); ++i)
    {
        for (std::size_t j = 0; j < shells.size(); ++j)
        {
            std::printf("%.16e ", shm_test(i, j));
        }
        std::printf("\n");
    }
    std::printf("\n");
}

int main()
{
    test_radon_integrator_is_accurate_for_shm();
    test_radon_integrator_resp_is_accurate_for_shm();
}
