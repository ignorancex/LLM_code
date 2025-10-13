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
#include "zebra_angle_integrator.hpp"

#include <zest/md_span.hpp>
#include <zest/rotor.hpp>
#include <zest/zernike_conventions.hpp>
#include <zest/zernike_expansion.hpp>
#include <zest/sh_glq_transformer.hpp>

#include "coordinate_transforms.hpp"

#include "zebra_radon.hpp"
#include "radon_util.hpp"

namespace zdm
{
namespace zebra
{

IsotropicAngleIntegrator::IsotropicAngleIntegrator(
    std::size_t dist_order):
    m_wigner_d_pi2(dist_order + 2),
    m_rotor(dist_order + 2),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_integrator_core(dist_order + 2),
    m_dist_order(dist_order) {}

void IsotropicAngleIntegrator::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t geg_order = dist_order + 2;
    m_wigner_d_pi2.expand(geg_order);
    m_rotor.expand(geg_order);
    m_geg_zernike_exp.resize(geg_order);
    m_rotated_geg_zernike_exp.resize(geg_order);
    m_integrator_core.resize(geg_order);
    m_dist_order = dist_order;
}

void IsotropicAngleIntegrator::integrate(
    ZernikeExpansionSpan<const std::array<double, 2>> distribution,
    std::span<const std::array<double, 3>> offsets,
    std::span<const double> shells, zest::MDSpan<double, 2> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(offsets[i], shells, out[i]);
}

void IsotropicAngleIntegrator::integrate(
    ZernikeExpansionSpan<const std::array<double, 2>> distribution,
    const std::array<double, 3>& offset, std::span<const double> shells,
    std::span<double> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    integrate(offset, shells, out);
}

void IsotropicAngleIntegrator::integrate(
    const std::array<double, 3>& offset, std::span<const double> shells,
    std::span<double> out)
{
    std::ranges::copy(
        m_geg_zernike_exp.flatten(),
        m_rotated_geg_zernike_exp.flatten().begin());

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);
    
    m_rotor.rotate(
            m_rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                m_rotated_geg_zernike_exp, offset_len, shells[i]);
}

[[nodiscard]] constexpr std::size_t geg_zernike_grids_size(
    std::size_t num_grids, std::size_t grid_order, std::size_t trunc_order
) noexcept
{
    return num_grids*zest::st::SphereGLQGridSpan<double>::size(std::min(grid_order, trunc_order));
}

[[nodiscard]] constexpr std::size_t zernike_expansion_sh_span_size(
    std::size_t order)
{
    return zest::zt::ZernikeSHSpan<std::array<double, 2>, zest::RowSkippingTriangleLayout<zest::IndexingMode::nonnegative>, zest::zt::ZernikeNorm::normed, zest::st::SHNorm::geo, zest::st::SHPhase::none>::size(order);
}

AnisotropicAngleIntegrator::AnisotropicAngleIntegrator(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2(std::max(dist_order + 2, resp_order)),
    m_geg_zernike_exp(dist_order + 2),
    m_rotated_geg_zernike_exp(zernike_expansion_sh_span_size(dist_order + 2)),
    m_rotated_geg_zernike_grids(
        geg_zernike_grids_size(
            dist_order + 2, dist_order + 2 + resp_order, trunc_order)),
    m_integrator_core(
        dist_order + 2, resp_order,
        std::min(dist_order + 2 + resp_order, trunc_order)),
    m_dist_order(dist_order),
    m_resp_order(resp_order),
    m_trunc_order(trunc_order) {}

void AnisotropicAngleIntegrator::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
{
    if (dist_order != m_dist_order || resp_order != m_resp_order)
        m_wigner_d_pi2.expand(std::max(dist_order + 2, resp_order));
    
    if (dist_order != m_dist_order)
    {
        m_geg_zernike_exp.resize(dist_order + 2);
        m_rotated_geg_zernike_exp.resize(
                zernike_expansion_sh_span_size(dist_order + 2));
    }

    if (dist_order != m_dist_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        m_rotated_geg_zernike_grids.resize(
                geg_zernike_grids_size(
                    dist_order + 2, dist_order + 2 + resp_order, trunc_order));
        m_integrator_core.resize(
                dist_order + 2, resp_order,
                std::min(dist_order + 2 + resp_order, trunc_order));
    }

    m_dist_order = dist_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void AnisotropicAngleIntegrator::integrate(
   ZernikeExpansionSpan<const std::array<double, 2>> distribution,
    SHExpansionVectorSpan<const std::array<double, 2>> response,
    std::span<const std::array<double, 3>> offsets,
    std::span<const double> rotation_angles, std::span<const double> shells,
    zest::MDSpan<double, 2> out, std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;
    const std::size_t top_order = std::min(geg_order + resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);

    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(
                response, offsets[i], rotation_angles[i], shells, geg_order,
                top_order, out[i]);
}

void AnisotropicAngleIntegrator::integrate(
    ZernikeExpansionSpan<const std::array<double, 2>> distribution,
    SHExpansionVectorSpan<const std::array<double, 2>> response,
    const std::array<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, zest::MDSpan<double, 2> out,
    std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);
    const std::size_t geg_order = dist_order + 2;
    const std::size_t top_order = std::min(geg_order + resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);

    integrate(response, offset, rotation_angle, shells, geg_order, top_order, out);
}

void AnisotropicAngleIntegrator::integrate(
    SHExpansionVectorSpan<const std::array<double, 2>> response,
    const std::array<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::size_t geg_order,
    std::size_t top_order, std::span<double> out)
{
    using ZernikeSpan = zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>;

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    SuperSpan<zest::st::SphereGLQGridSpan<double>>
    rotated_geg_zernike_grids(
            m_rotated_geg_zernike_grids.data(), geg_order, top_order);

    for (std::size_t n = 0; n < geg_order; ++n)
    {
        std::ranges::copy(
                m_geg_zernike_exp[n].flatten(),
                m_rotated_geg_zernike_exp.begin());
        ZernikeSpan::SubSpan rotated_geg_zernike_exp(
                m_rotated_geg_zernike_exp.data(), n + 1);
        
        m_rotor.rotate(
                rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
                rotation_type);
        m_glq_transformer.backward_transform(
                rotated_geg_zernike_exp, rotated_geg_zernike_grids[n]);
    }

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate(
                rotated_geg_zernike_grids, response[i], offset, rotation_angle, 
                shells[i], m_wigner_d_pi2);
}

IsotropicTransverseAngleIntegrator::IsotropicTransverseAngleIntegrator(
    std::size_t dist_order):
    m_wigner_d_pi2(dist_order + 4),
    m_rotor(dist_order + 4),
    m_geg_zernike_exp(dist_order + 2),
    m_geg_zernike_exp_x(dist_order + 3),
    m_geg_zernike_exp_y(dist_order + 3),
    m_geg_zernike_exp_z(dist_order + 3),
    m_geg_zernike_exp_r2(dist_order + 4),
    m_rotated_geg_zernike_exp(dist_order + 2),
    m_rotated_trans_geg_zernike_exp(dist_order + 4),
    m_multiplier(dist_order + 4),
    m_integrator_core(dist_order + 4),
    m_dist_order(dist_order) {}

void IsotropicTransverseAngleIntegrator::resize(std::size_t dist_order)
{
    if (dist_order == m_dist_order) return;
    const std::size_t const_geg_order = dist_order + 2;
    const std::size_t linear_geg_order = dist_order + 3;
    const std::size_t trans_geg_order = dist_order + 4;
    m_wigner_d_pi2.expand(trans_geg_order);
    m_rotor.expand(trans_geg_order);
    m_geg_zernike_exp.resize(const_geg_order);
    m_geg_zernike_exp_x.resize(linear_geg_order);
    m_geg_zernike_exp_y.resize(linear_geg_order);
    m_geg_zernike_exp_z.resize(linear_geg_order);
    m_geg_zernike_exp_r2.resize(trans_geg_order);
    m_rotated_geg_zernike_exp.resize(const_geg_order);
    m_rotated_trans_geg_zernike_exp.resize(trans_geg_order);
    m_multiplier.expand(dist_order);
    m_integrator_core.resize(trans_geg_order);
    m_dist_order = dist_order;
}

void IsotropicTransverseAngleIntegrator::integrate(
    ZernikeExpansionSpan<const std::array<double, 2>> distribution,
    std::span<const std::array<double, 3>> offsets, std::span<const double> shells,
    zest::MDSpan<std::array<double, 2>, 2> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);
    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(offsets[i], shells, out[i]);
}

void IsotropicTransverseAngleIntegrator::integrate(
    ZernikeExpansionSpan<const std::array<double, 2>> distribution,
    const std::array<double, 3>& offset, std::span<const double> shells,
    std::span<std::array<double, 2>> out)
{
    resize(distribution.order());
    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);
    integrate(offset, shells, out);
}

void IsotropicTransverseAngleIntegrator::integrate(
    const std::array<double, 3>& offset, std::span<const double> shells,
    std::span<std::array<double, 2>> out)
{
    std::ranges::copy(
        m_geg_zernike_exp.flatten(),
        m_rotated_geg_zernike_exp.flatten().begin());

    std::ranges::copy(
        m_geg_zernike_exp_r2.flatten(),
        m_rotated_trans_geg_zernike_exp.flatten().begin());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        dot(offset, offset), m_geg_zernike_exp.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[0], m_geg_zernike_exp_x.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[1], m_geg_zernike_exp_y.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[2], m_geg_zernike_exp_z.flatten());

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    m_rotor.rotate(
            m_rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);
    m_rotor.rotate(
            m_rotated_trans_geg_zernike_exp, m_wigner_d_pi2, euler_angles, rotation_type);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                m_rotated_geg_zernike_exp, m_rotated_trans_geg_zernike_exp, offset_len, shells[i]);
}

AnisotropicTransverseAngleIntegrator::AnisotropicTransverseAngleIntegrator(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order):
    m_wigner_d_pi2(std::max(dist_order + 4, resp_order)),
    m_geg_zernike_exp(dist_order + 2),
    m_geg_zernike_exp_x(dist_order + 3),
    m_geg_zernike_exp_y(dist_order + 3),
    m_geg_zernike_exp_z(dist_order + 3),
    m_geg_zernike_exp_r2(dist_order + 4),
    m_rotated_geg_zernike_exp(zernike_expansion_sh_span_size(dist_order + 2)),
    m_rotated_trans_geg_zernike_exp(dist_order + 4),
    m_rotated_geg_zernike_grids(
        geg_zernike_grids_size(
            dist_order + 2, dist_order + 4 + resp_order, trunc_order)),
    m_rotated_trans_geg_zernike_grids(
        geg_zernike_grids_size(
            dist_order + 4, dist_order + 4 + resp_order, trunc_order)),
    m_multiplier(dist_order + 4),
    m_integrator_core(
        dist_order + 4, resp_order,
        std::min(dist_order + 4 + resp_order, trunc_order)),
    m_dist_order(dist_order),
    m_resp_order(resp_order),
    m_trunc_order(trunc_order) {}

void AnisotropicTransverseAngleIntegrator::resize(
    std::size_t dist_order, std::size_t resp_order, std::size_t trunc_order)
{
    if (dist_order != m_dist_order || resp_order != m_resp_order)
        m_wigner_d_pi2.expand(std::max(dist_order + 4, resp_order));
    
    if (dist_order != m_dist_order)
    {
        m_multiplier.expand(dist_order + 4);
        m_geg_zernike_exp.resize(dist_order + 2);
        m_geg_zernike_exp_x.resize(dist_order + 3);
        m_geg_zernike_exp_y.resize(dist_order + 3);
        m_geg_zernike_exp_z.resize(dist_order + 3);
        m_geg_zernike_exp_r2.resize(dist_order + 4);
        m_rotated_geg_zernike_exp.resize(
                zernike_expansion_sh_span_size(dist_order + 2));
        m_rotated_trans_geg_zernike_exp.resize(dist_order + 4);
    }

    if (dist_order != m_dist_order || resp_order != m_resp_order || trunc_order != m_trunc_order)
    {
        m_rotated_geg_zernike_grids.resize(
                geg_zernike_grids_size(
                    dist_order + 2, dist_order + 4 + resp_order, trunc_order));
        m_rotated_trans_geg_zernike_grids.resize(
                geg_zernike_grids_size(
                    dist_order + 4, dist_order + 4 + resp_order, trunc_order));
        m_integrator_core.resize(
                dist_order + 4, resp_order,
                std::min(dist_order + 4 + resp_order, trunc_order));
    }

    m_dist_order = dist_order;
    m_resp_order = resp_order;
    m_trunc_order = trunc_order;
}

void AnisotropicTransverseAngleIntegrator::integrate(
    ZernikeExpansionSpan<const std::array<double, 2>> distribution,
    SHExpansionVectorSpan<const std::array<double, 2>> response,
    std::span<const std::array<double, 3>> offsets,
    std::span<const double> rotation_angles, std::span<const double> shells,
    zest::MDSpan<std::array<double, 2>, 2> out, std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);

    for (std::size_t i = 0; i < offsets.size(); ++i)
        integrate(response, offsets[i], rotation_angles[i], shells, out[i]);
}

void AnisotropicTransverseAngleIntegrator::integrate(
    ZernikeExpansionSpan<const std::array<double, 2>> distribution,
    SHExpansionVectorSpan<const std::array<double, 2>> response,
    const std::array<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<std::array<double, 2>> out,
    std::size_t trunc_order)
{
    const std::size_t dist_order = distribution.order();
    const std::size_t resp_order = response[0].order();
    resize(dist_order, resp_order, trunc_order);

    zebra::radon_transform(distribution, m_geg_zernike_exp);
    m_multiplier.multiply_by_x_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_x);
    m_multiplier.multiply_by_y_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_y);
    m_multiplier.multiply_by_z_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_z);
    m_multiplier.multiply_by_r2_and_radon_transform_inplace(
            distribution, m_geg_zernike_exp_r2);

    integrate(response, offset, rotation_angle, shells, out);
}

void AnisotropicTransverseAngleIntegrator::integrate(
    SHExpansionVectorSpan<const std::array<double, 2>> response,
    const std::array<double, 3>& offset, double rotation_angle,
    std::span<const double> shells, std::span<std::array<double, 2>> out)
{
    using ZernikeSpan = zest::zt::RealZernikeSpanNormalGeo<std::array<double, 2>>;

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(offset_az, offset_colat);

    const std::size_t geg_order = m_dist_order + 2;
    const std::size_t trans_geg_order = m_dist_order + 4;
    const std::size_t top_order = std::min(trans_geg_order + m_resp_order, m_trunc_order);
    SuperSpan<zest::st::SphereGLQGridSpan<double>>
    rotated_geg_zernike_grids(
            m_rotated_geg_zernike_grids.data(), geg_order, top_order);

    for (std::size_t n = 0; n < geg_order; ++n)
    {
        std::ranges::copy(
                m_geg_zernike_exp[n].flatten(),
                m_rotated_geg_zernike_exp.begin());
        ZernikeSpan::SubSpan rotated_geg_zernike_exp(
                m_rotated_geg_zernike_exp.data(), n + 1);
        m_rotor.rotate(
                rotated_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
                rotation_type);
        m_glq_transformer.backward_transform(
                rotated_geg_zernike_exp, rotated_geg_zernike_grids[n]);
    }

    SuperSpan<zest::st::SphereGLQGridSpan<double>>
    rotated_trans_geg_zernike_grids(
            m_rotated_trans_geg_zernike_grids.data(), trans_geg_order, top_order);

    std::ranges::copy(
            m_geg_zernike_exp_r2.flatten(),
            m_rotated_trans_geg_zernike_exp.flatten().begin());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        offset_len*offset_len, m_geg_zernike_exp.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[0], m_geg_zernike_exp_x.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[1], m_geg_zernike_exp_y.flatten());
    util::fmadd(
        m_rotated_trans_geg_zernike_exp.flatten(),
        -2.0*offset[2], m_geg_zernike_exp_z.flatten());
    
    m_rotor.rotate(
            m_rotated_trans_geg_zernike_exp, m_wigner_d_pi2, euler_angles, 
            rotation_type);

    for (std::size_t n = 0; n < m_rotated_trans_geg_zernike_exp.order(); ++n)
        m_glq_transformer.backward_transform(
                m_rotated_trans_geg_zernike_exp[n], rotated_trans_geg_zernike_grids[n]);

    for (std::size_t i = 0; i < shells.size(); ++i)
        out[i] = m_integrator_core.integrate_transverse(
                rotated_geg_zernike_grids, rotated_trans_geg_zernike_grids, 
                response[i], offset, rotation_angle, shells[i], m_wigner_d_pi2);
}

} // namespace zebra
} // namespace zdm
