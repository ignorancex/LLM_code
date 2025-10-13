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
#include "zebra_parallel_angle_integrator.hpp"

#include <omp.h>

#include "coordinate_transforms.hpp"

#include "zebra_radon.hpp"
#include "radon_util.hpp"

namespace zdm
{
namespace zebra
{
namespace parallel
{

IsotropicAngleIntegrator::IsotropicAngleIntegrator(
    std::size_t num_threads):
    m_contexts(num_threads), m_num_threads(num_threads) {}

IsotropicAngleIntegrator::IsotropicAngleIntegrator(
    std::size_t dist_order, std::size_t num_threads):
    m_wigner_d_pi2(dist_order + 2), m_geg_zernike_exp(dist_order + 2), m_rotated_geg_zernike_exp(num_threads*zernike_exp_size(dist_order + 2)),
    m_contexts(num_threads), m_dist_order(dist_order), m_num_threads(num_threads)
{
    for (auto& context : m_contexts)
        context = {
            zest::Rotor(),
            detail::IsotropicAngleIntegratorCore(dist_order + 2)
        };
}

void IsotropicAngleIntegrator::resize(std::size_t dist_order)
{
    if (m_dist_order == dist_order) return;
    const std::size_t geg_order = dist_order + 2;
    m_wigner_d_pi2.expand(geg_order);
    m_geg_zernike_exp.resize(geg_order);

    m_rotated_geg_zernike_exp.resize(m_num_threads*zernike_exp_size(geg_order));
    for (auto& context : m_contexts)
        context.integrator.resize(geg_order);
    m_dist_order = dist_order;
}

void IsotropicAngleIntegrator::integrate(
    zest::zt::RealZernikeSpanNormalGeo<const std::array<double, 2>> distribution, std::span<const std::array<double, 3>> offsets, std::span<const double> shells, zest::MDSpan<double, 2> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);

    #pragma omp parallel for num_threads(m_num_threads)
    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(omp_get_thread_num(), offsets[i], shells, out[i]);
}

void IsotropicAngleIntegrator::integrate(
    std::size_t thread_id, const std::array<double, 3>& offset,
    std::span<const double> shells, std::span<double> out)
{
    const auto& [offset_len, offset_colat, offset_az]
        = coordinates::cartesian_to_spherical_phys(offset);

    const std::array<double, 3> euler_angles = {
        0.0, -offset_colat, std::numbers::pi - offset_az
    };

    zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>> 
    rotated_geg_zernike_exp = accesss_rotated_geg_zernike_exp(thread_id);

    std::ranges::copy(
        m_geg_zernike_exp.flatten(),
        rotated_geg_zernike_exp.flatten().begin());

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    zest::Rotor& rotor = m_contexts[thread_id].rotor;
    for (std::size_t n = 0; n < m_geg_zernike_exp.order(); ++n)
        rotor.rotate(
                rotated_geg_zernike_exp[n], m_wigner_d_pi2, euler_angles, rotation_type);

    detail::IsotropicAngleIntegratorCore& integrator
        = m_contexts[thread_id].integrator;
    for (std::size_t i = 0; i < shells.size(); ++i)
    {
        out[i] = integrator.integrate(
                rotated_geg_zernike_exp, offset_len, shells[i]);
    }
}

zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>
IsotropicAngleIntegrator::accesss_rotated_geg_zernike_exp(
    std::size_t thread_id) noexcept
{
    const std::size_t geg_order = m_dist_order + 2;
    const std::size_t size = zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>::size(geg_order);

    return zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>(m_rotated_geg_zernike_exp.data() + thread_id*size, geg_order);
}

AnisotropicAngleIntegrator::AnisotropicAngleIntegrator(
    std::size_t num_teams, std::size_t threads_per_team):
    m_rotors(num_teams), m_integrators(num_teams*threads_per_team),
    m_num_teams(num_teams), m_threads_per_team(threads_per_team) {}

AnisotropicAngleIntegrator::AnisotropicAngleIntegrator(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order, 
    std::size_t num_teams, std::size_t threads_per_team):
    m_wigner_d_pi2(std::max(dist_order + 2, resp_order)),
    m_rotors(num_teams),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(num_teams*zernike_exp_size(dist_order + 2)),
    m_rotated_geg_zernike_grids(
        num_teams*(dist_order + 2)*sh_grid_size(std::min(dist_order + 2 + resp_order, trunc_order))),
    m_integrators(num_teams*threads_per_team),
    m_dist_order(dist_order),
    m_resp_order(resp_order),
    m_trunc_order(trunc_order),
    m_num_teams(num_teams),
    m_threads_per_team(threads_per_team)
{
    for (auto& rotor : m_rotors)
        rotor = zest::Rotor(dist_order + 2);
    
    for (auto& integrator : m_integrators)
        integrator = detail::AnisotropicAngleIntegratorCore(
                dist_order + 2, resp_order,
                std::max(dist_order + 2, resp_order));
}

void AnisotropicAngleIntegrator::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
{
    const std::size_t geg_order = dist_order + 2;
    if (dist_order != m_dist_order || resp_order != m_resp_order)
        m_wigner_d_pi2.expand(std::max(geg_order, resp_order));
    
    if (dist_order != m_dist_order)
    {
        m_geg_zernike_exp.resize(geg_order);
        m_rotated_geg_zernike_exp.resize(
                m_num_teams*zernike_exp_size(geg_order));
        for (auto& rotor : m_rotors)
            rotor.expand(geg_order);
    }

    if (dist_order != m_dist_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        const std::size_t top_order
            = std::min(geg_order + resp_order, trunc_order);
        m_rotated_geg_zernike_grids.resize(
                m_num_teams*geg_order*sh_grid_size(top_order));
        for (auto& integrator : m_integrators)
            integrator.resize(geg_order, resp_order, top_order);
    }

    m_dist_order = dist_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void AnisotropicAngleIntegrator::integrate(
    zest::zt::RealZernikeSpanNormalGeo<const std::array<double, 2>> distribution, std::span<const std::array<double, 3>> offsets, std::span<const double> shells, SHExpansionVectorSpan<const std::array<double, 2>> response, std::span<const double> rotation_angles, zest::MDSpan<double, 2> out, std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;

    zebra::radon_transform(distribution, m_geg_zernike_exp);

    #pragma omp teams num_teams(m_num_teams)
    {
        #pragma omp distribute
        for (std::size_t i = 0; i < offsets.size(); ++i)
        {
            using ZernikeSpan = zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>;

            const auto& [offset_len, offset_colat, offset_az]
                = coordinates::cartesian_to_spherical_phys(offsets[i]);

            const std::array<double, 3> euler_angles = {
                0.0, -offset_colat, std::numbers::pi - offset_az
            };

            const std::size_t team_id = omp_get_team_num();
            const std::size_t team_ind = m_threads_per_team*team_id;
            SuperSpan<zest::st::SphereGLQGridSpan<double>>
            rotated_geg_zernike_grids
                = accesss_rotated_geg_zernike_exp_grids(team_id);

            for (std::size_t n = 0; n < geg_order; ++n)
            {
                ZernikeSpan::SubSpan rotated_geg_zernike_exp
                    = accesss_rotated_geg_zernike_exp(team_id, n + 1);
                std::ranges::copy(m_geg_zernike_exp[n].flatten(), m_rotated_geg_zernike_exp.begin());


                constexpr zest::RotationType rotation_type
                    = zest::RotationType::coordinate;
                m_integrators[team_ind].rotor().rotate(
                        rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, rotation_type);
                m_integrators[team_ind].glq_transformer().backward_transform(
                        rotated_geg_zernike_exp, rotated_geg_zernike_grids[n]);
            }

            #pragma omp parallel for num_threads(m_threads_per_team)
            for (std::size_t j = 0; j < shells.size(); ++j)
            {
                const std::size_t thread_ind
                    = team_ind + omp_get_thread_num();
                out(i, j) = m_integrators[thread_ind].integrate(
                        rotated_geg_zernike_grids, response[j], offsets[i], rotation_angles[i], shells[j], m_wigner_d_pi2);
            }
        }
    }
}

zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>::SubSpan
AnisotropicAngleIntegrator::accesss_rotated_geg_zernike_exp(
    std::size_t team_id, std::size_t order) noexcept
{
    const std::size_t geg_order = m_dist_order + 2;
    const std::size_t size = zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>::size(geg_order);

    return zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>::SubSpan(m_rotated_geg_zernike_exp.data() + team_id*size, order);
}

SuperSpan<zest::st::SphereGLQGridSpan<double>>
AnisotropicAngleIntegrator::accesss_rotated_geg_zernike_exp_grids(
    std::size_t team_id)
{
    const std::size_t geg_order = m_dist_order + 2;
    const std::size_t top_order = std::min(
            geg_order + m_resp_order, m_trunc_order);

    const std::size_t grid_size = sh_grid_size(top_order);
    const std::size_t size = geg_order*grid_size;

    return SuperSpan<zest::st::SphereGLQGridSpan<double>>(m_rotated_geg_zernike_grids.data() + team_id*size, {geg_order}, grid_size);
}

} // namespace parallel
} // namespace zebra
} // namespace zdm
