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
#include "zebra_angle_integrator_core.hpp"

#include <fstream>

#include <zest/grid_evaluator.hpp>

#include "coordinate_transforms.hpp"

#include "radon_util.hpp"

namespace zdm
{
namespace zebra
{
namespace detail
{

IsotropicAngleIntegratorCore::IsotropicAngleIntegratorCore(
    std::size_t geg_order):
    m_aff_leg_integrals(geg_order, 0),
    m_aff_leg_ylm_integrals(TrapezoidLayout::size(geg_order, 0)),
    m_ylm_integral_norms(geg_order)
{
    for (std::size_t l = 0; l  < geg_order; ++l)
        m_ylm_integral_norms[l]
            = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
}

void IsotropicAngleIntegratorCore::resize(std::size_t geg_order)
{
    if (order() == geg_order) return;
    m_aff_leg_integrals.resize(geg_order, 0);
    m_aff_leg_ylm_integrals.resize(
            TrapezoidLayout::size(geg_order, 0));
    m_ylm_integral_norms.resize(geg_order);
    for (std::size_t l = 0; l  < geg_order; ++l)
        m_ylm_integral_norms[l]
            = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
}

double IsotropicAngleIntegratorCore::integrate(
    ZernikeExpansionSpan<const std::array<double, 2>> rotated_geg_zernike_exp,
    double offset_len, double shell)
{
    if (shell > 1.0 + offset_len) return 0.0;

    TrapezoidSpan<double>
    aff_leg_ylm_integrals = evaluate_aff_leg_ylm_integrals(
            shell, offset_len, rotated_geg_zernike_exp.order());
    
    double res = 0;
    for (std::size_t n = 0; n < rotated_geg_zernike_exp.order(); ++n)
    {
        auto rotated_geg_n = rotated_geg_zernike_exp[n];
        auto aff_leg_ylm_integrals_n = aff_leg_ylm_integrals[n];
        for (std::size_t l = n & 1; l <= n; l += 2)
            res += rotated_geg_n(l, 0)[0]*aff_leg_ylm_integrals_n[l];
    }

    return (2.0*std::numbers::pi)*res;
}

std::array<double, 2> IsotropicAngleIntegratorCore::integrate_transverse(
    ZernikeExpansionSpan<const std::array<double, 2>> rotated_geg_zernike_exp,
    ZernikeExpansionSpan<const std::array<double, 2>> rotated_trans_geg_zernike_exp,
    double offset_len, double shell)
{
    if (shell > 1.0 + offset_len) return {0.0, 0.0};

    const double shell_sq = shell*shell;

    TrapezoidSpan<double>
    aff_leg_ylm_integrals = evaluate_aff_leg_ylm_integrals(
            shell, offset_len, rotated_trans_geg_zernike_exp.order());
    
    std::array<double, 2> res = {0.0, 0.0};

    // transverse contribution
    for (std::size_t n = 0; n < rotated_trans_geg_zernike_exp.order(); ++n)
    {
        auto rotated_trans_geg_n = rotated_trans_geg_zernike_exp[n];
        auto aff_leg_ylm_integrals_n = aff_leg_ylm_integrals[n];
        for (std::size_t l = n & 1; l <= n; l += 2)
            res[1] += rotated_trans_geg_n(l, 0)[0]*aff_leg_ylm_integrals_n[l];
    }

    // nontransverse contribution
    for (std::size_t n = 0; n < rotated_geg_zernike_exp.order(); ++n)
    {
        auto rotated_geg_n = rotated_geg_zernike_exp[n];
        auto aff_leg_ylm_integrals_n = aff_leg_ylm_integrals[n];
        
        for (std::size_t l = n & 1; l <= n; l += 2)
        {
            const double nontrans
                = rotated_geg_n(l, 0)[0]*aff_leg_ylm_integrals_n[l];
            res[0] += nontrans;
            res[1] -= shell_sq*nontrans;
        }
    }

    return {(2.0*std::numbers::pi)*res[0], (2.0*std::numbers::pi)*res[1]};
}

TrapezoidSpan<double> 
IsotropicAngleIntegratorCore::evaluate_aff_leg_ylm_integrals(
    double shell, double offset_len, std::size_t geg_order)
{
    TrapezoidSpan<double> integrals(
            m_aff_leg_ylm_integrals.data(), geg_order, 0);
    m_aff_leg_integrals.integrals(integrals, shell, offset_len);
    
    for (std::size_t n = 0; n < geg_order; ++n)
    {
        auto integrals_n = integrals[n];
        for (std::size_t l = 0; l <= n; ++l)
            integrals_n[l] *= m_ylm_integral_norms[l];
    }

    return integrals;
}

AnisotropicAngleIntegratorCore::AnisotropicAngleIntegratorCore(
    std::size_t geg_order, std::size_t resp_order, std::size_t top_order):
    m_rotor(std::max(geg_order, resp_order)), m_glq_transformer(top_order), 
    m_rotated_response_exp(resp_order), m_rotated_response_grid(top_order), 
    m_aff_leg_integrals(geg_order, resp_order),
    m_aff_leg_ylm_integrals(TrapezoidLayout::size(geg_order, resp_order)), 
    m_ylm_integral_norms(top_order), m_zonal_transformer(top_order), 
    m_rotated_grid(top_order), m_rotated_exp(top_order)
{
    for (std::size_t l = 0; l  < top_order; ++l)
        m_ylm_integral_norms[l]
            = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));
}

void AnisotropicAngleIntegratorCore::resize(
    std::size_t geg_order, std::size_t resp_order, std::size_t top_order)
{
    m_rotor.expand(std::max(geg_order, resp_order));
    m_glq_transformer.resize(top_order);
    m_rotated_response_exp.resize(resp_order);
    m_rotated_response_grid.resize(top_order);

    m_aff_leg_integrals.resize(geg_order, resp_order);
    m_aff_leg_ylm_integrals.resize(
            TrapezoidLayout::size(geg_order, resp_order));
    m_ylm_integral_norms.resize(top_order);
    for (std::size_t l = 0; l  < top_order; ++l)
        m_ylm_integral_norms[l]
            = (2.0*std::numbers::pi)*std::sqrt(double(2*l + 1));

    m_zonal_transformer.resize(top_order);
    m_rotated_grid.resize(top_order);
    m_rotated_exp.resize(top_order);
}

template <std::floating_point T>
std::vector<T> linspace(T start, T stop, std::size_t count)
{
    if (count == 0) return {};
    if (count == 1) return {start};

    std::vector<T> res(count);
    const T step = (stop - start)/T(count - 1);
    for (std::size_t i = 0; i < count - 1; ++i)
        res[i] = start + T(i)*step;
    
    res[count - 1] = stop;

    return res;
}

void save_debug_grid(
    const char* fname_prefix,
    zest::st::RealSHSpanGeo<const std::array<double, 2>> sh_exp, double offset_az, double offset_colat, const std::array<double, 3> euler_angles)
{
    constexpr std::size_t num_lon = 100;
    constexpr std::size_t num_lat = 50;
    std::vector<double> longitudes = linspace(
            0.0, 2.0*std::numbers::pi, num_lon);
    std::vector<double> colatitudes = linspace(
            0.0, std::numbers::pi, num_lat);
    std::vector<double> grid
        = zest::st::GridEvaluator().evaluate(sh_exp, longitudes, colatitudes);
    
    zest::MDSpan<double, 2> values_span(grid.data(), {num_lon, num_lat});

    constexpr double to_deg = 180.0/std::numbers::pi;

    char fname[512];
    std::sprintf(fname, "%s_%.0f_%.0f_rot_%.0f_%.0f_%.0f.dat", fname_prefix, to_deg*offset_az, to_deg*offset_colat, to_deg*euler_angles[0], to_deg*euler_angles[1], to_deg*euler_angles[2]);
    std::ofstream fs(fname);
    for (std::size_t i = 0; i < values_span.extents()[0]; ++i)
    {
        for (std::size_t j = 0; j < values_span.extents()[1]; ++j)
            fs << values_span(i,j) << ' ';
        fs << '\n';
    }
    fs.close();
}

double AnisotropicAngleIntegratorCore::integrate(
    SuperSpan<zest::st::SphereGLQGridSpan<double>> rotated_geg_zernike_grids,
    zest::st::RealSHSpanGeo<const std::array<double, 2>> response_exp,
    const std::array<double, 3>& offset, double rotation_angle, double shell, 
    zest::WignerdPiHalfCollection wigner_d_pi2)
{
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    if (shell > 1.0 + offset_len) return 0.0;

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(
                offset_az - rotation_angle, offset_colat);

    std::ranges::copy(
            response_exp.flatten(), m_rotated_response_exp.flatten().begin());

    m_rotor.rotate(
            m_rotated_response_exp, wigner_d_pi2, euler_angles, rotation_type);
    #ifndef NDEBUG
        save_debug_grid(
            "rotated_response", m_rotated_response_exp, offset_az, offset_colat, euler_angles);
    #endif

    m_glq_transformer.backward_transform(
            m_rotated_response_exp, m_rotated_response_grid);

    const std::size_t geg_order = rotated_geg_zernike_grids.extent();
    const std::size_t resp_order = response_exp.order();
    TrapezoidSpan<double> aff_leg_ylm_integrals
        = evaluate_aff_leg_ylm_integrals(
                shell, offset_len, geg_order, resp_order);

    const std::size_t extra_extent = resp_order - std::min(1UL, resp_order);
    double res = 0.0;
    for (std::size_t n = 0; n < geg_order; ++n)
    {
        std::ranges::copy(
                m_rotated_response_grid.flatten(),
                m_rotated_grid.flatten().begin());
        util::mul(m_rotated_grid.flatten(), rotated_geg_zernike_grids[n].flatten());
        m_zonal_transformer.forward_transform(
                m_rotated_grid, m_rotated_exp);
        for (std::size_t l = 0; l <= n + extra_extent; ++l)
            res += m_rotated_exp[l]*aff_leg_ylm_integrals(n, l);
    }

    return (2.0*std::numbers::pi)*res;
}



std::array<double, 2> AnisotropicAngleIntegratorCore::integrate_transverse(
    SuperSpan<zest::st::SphereGLQGridSpan<double>> rotated_geg_zernike_grids,
    SuperSpan<zest::st::SphereGLQGridSpan<double>> rotated_trans_geg_zernike_grids,
    zest::st::RealSHSpanGeo<const std::array<double, 2>> response_exp,
    const std::array<double, 3>& offset, double rotation_angle, double shell, 
    zest::WignerdPiHalfCollection wigner_d_pi2)
{
    const auto& [offset_az, offset_colat, offset_len]
        = coordinates::cartesian_to_spherical_phys(offset);
    if (shell > 1.0 + offset_len) return {0.0, 0.0};

    constexpr zest::RotationType rotation_type = zest::RotationType::coordinate;
    const std::array<double, 3> euler_angles
        = util::euler_angles_to_align_z<rotation_type>(
                offset_az - rotation_angle, offset_colat);

    std::ranges::copy(
            response_exp.flatten(), m_rotated_response_exp.flatten().begin());

    m_rotor.rotate(
            m_rotated_response_exp, wigner_d_pi2, euler_angles, rotation_type);
    #ifndef NDEBUG
        save_debug_grid(
            "rotated_response_trans", m_rotated_response_exp, offset_az, offset_colat, euler_angles);
    #endif

    m_glq_transformer.backward_transform(
            m_rotated_response_exp, m_rotated_response_grid);

    const std::size_t geg_order = rotated_geg_zernike_grids.extent();
    const std::size_t trans_geg_order = rotated_trans_geg_zernike_grids.extent();
    const std::size_t resp_order = response_exp.order();
    TrapezoidSpan<double> aff_leg_ylm_integrals
        = evaluate_aff_leg_ylm_integrals(
                shell, offset_len, trans_geg_order, resp_order);

    const std::size_t extra_extent = resp_order - std::min(1UL, resp_order);
    std::array<double, 2> res = {0.0, 0.0};

    //transverse contribution
    for (std::size_t n = 0; n < trans_geg_order; ++n)
    {
        std::ranges::copy(
                m_rotated_response_grid.flatten(),
                m_rotated_grid.flatten().begin());
        assert(m_rotated_grid.flatten().size() == rotated_geg_zernike_grids[n].flatten().size());
        util::mul(m_rotated_grid.flatten(), rotated_trans_geg_zernike_grids[n].flatten());
        m_zonal_transformer.forward_transform(
                m_rotated_grid, m_rotated_exp);
        for (std::size_t l = 0; l <= n + extra_extent; ++l)
            res[1] += m_rotated_exp[l]*aff_leg_ylm_integrals(n, l);
    }

    const double shell_sq = shell*shell;

    // nontransverse contribution
    for (std::size_t n = 0; n < geg_order; ++n)
    {
        std::ranges::copy(
                m_rotated_response_grid.flatten(),
                m_rotated_grid.flatten().begin());
        assert(m_rotated_grid.flatten().size() == rotated_geg_zernike_grids[n].flatten().size());
        util::mul(m_rotated_grid.flatten(), rotated_geg_zernike_grids[n].flatten());
        m_zonal_transformer.forward_transform(
                m_rotated_grid, m_rotated_exp);
        for (std::size_t l = 0; l <= n + extra_extent; ++l)
        {
            const double nontrans
                = m_rotated_exp[l]*aff_leg_ylm_integrals(n, l);
            res[0] += nontrans;
            res[1] -= shell_sq*nontrans;
        }
    }

    return {2.0*std::numbers::pi*res[0], 2.0*std::numbers::pi*res[1]};
}

TrapezoidSpan<double> 
AnisotropicAngleIntegratorCore::evaluate_aff_leg_ylm_integrals(
    double shell, double offset_len, std::size_t geg_order,
    std::size_t resp_order)
{
    const std::size_t extra_extent = resp_order - std::min(1UL, resp_order);
    TrapezoidSpan<double> integrals(
            m_aff_leg_ylm_integrals.data(), geg_order, extra_extent);
    m_aff_leg_integrals.integrals(integrals, shell, offset_len);
    
    for (std::size_t n = 0; n < geg_order; ++n)
    {
        auto integrals_n = integrals[n];
        for (std::size_t l = 0; l <= n + extra_extent; ++l)
            integrals_n[l] *= m_ylm_integral_norms[l];
    }

    return integrals;
}

} // namespace detail
} // namespace zebra
} // namespace zdm
