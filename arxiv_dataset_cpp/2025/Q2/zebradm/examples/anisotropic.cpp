#include <array>
#include <vector>
#include <cmath>
#include <numbers>
#include <random>
#include <cstdio>

#include <zest/zernike_glq_transformer.hpp>
#include <zest/md_array.hpp>
#include <zest/rotor.hpp>

#include <zebradm/zebra_angle_integrator.hpp>
#include <zebradm/linalg.hpp>

std::vector<std::array<double, 3>>
generate_offsets(std::size_t count, double offset_len)
{
    std::mt19937 gen;
    std::uniform_real_distribution rng_dist{0.0, 1.0};

    std::vector<std::array<double, 3>> offsets(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const double ct = 2.0*rng_dist(gen) - 1.0;
        const double st = std::sqrt((1.0 - ct)*(1.0 + ct));
        const double az = 2.0*std::numbers::pi*rng_dist(gen);
        offsets[i] = {offset_len*st*std::cos(az), offset_len*st*std::sin(az), ct};
    }

    return offsets;
}

std::vector<double> generate_rotation_angles(std::size_t offset_count)
{
    std::vector<double> rotation_angles(offset_count);
    for (std::size_t i = 0; i < offset_count; ++i)
        rotation_angles[i]
            = 2.0*std::numbers::pi*double(i)/double(offset_count - 1);
    return rotation_angles;
}


std::vector<double> generate_shells(std::size_t count, double offset_len)
{
    const double max_shell = 1.0 + offset_len;

    std::vector<double> shells(count);
    for (std::size_t i = 0; i < count; ++i)
        shells[i] = max_shell*double(i)/double(count - 1);

    return shells;
}

int main()
{
    constexpr std::array<double, 3> var = {0.5, 0.6, 0.7};
    auto dist_func = [&](double lon, double colat, double r)
    {
        const std::array<double, 3> x = {
            r*std::sin(colat)*std::cos(lon),
            r*std::sin(colat)*std::sin(lon),
            r*std::cos(colat)
        };

        return std::exp(-(x[0]*x[0]/var[0]) + x[1]*x[1]/var[1] + x[2]*x[2]/var[2]);
    };

    constexpr std::array<double, 3> a = {0.5, 0.5, 0.5};
    auto resp_func = [&](double shell, double lon, double colat)
    {
        const std::array<double, 3> dir = {
            std::sin(colat)*std::cos(lon),
            std::sin(colat)*std::sin(lon),
            std::cos(colat)
        };

        return std::exp(-shell*(zdm::dot(dir, a)));
    };

    constexpr double offset_len = 0.5;
    constexpr std::size_t offset_count = 10;
    constexpr std::size_t shell_count = 50;

    std::vector<std::array<double, 3>> offsets
        = generate_offsets(offset_count, offset_len);
    std::vector<double> rotation_angles = generate_rotation_angles(offset_count);
    std::vector<double> shells = generate_shells(shell_count, offset_len);

    constexpr double radius = 2.0;
    constexpr std::size_t dist_order = 30;
    zest::zt::ZernikeTransformerNormalGeo zernike_transformer{};
    zest::zt::RealZernikeExpansionNormalGeo distribution
        = zernike_transformer.transform(dist_func, radius, dist_order);

    constexpr std::size_t resp_order = 60;
    zdm::zebra::ResponseTransformer response_transformer{};
    zdm::SHExpansionVector response 
        = response_transformer.transform(resp_func, shells, resp_order);

    constexpr std::array<double, 3> euler_angles = {
        std::numbers::pi/2, std::numbers::pi/3, std::numbers::pi/4
    };

    zest::WignerdPiHalfCollection wigner(resp_order);
    zest::Rotor rotor(resp_order);
    for (std::size_t i = 0; i < response.extent(); ++i)
        rotor.rotate(response[i], wigner, euler_angles, zest::RotationType::coordinate);

    zdm::zebra::AnisotropicAngleIntegrator integrator(dist_order, resp_order);

    zest::MDArray<double, 2> out({offset_count, shell_count});
    integrator.integrate(
            distribution, response, offsets, rotation_angles, shells, out);

    for (auto& element : out.flatten())
        element *= radius*radius;

    for (std::size_t i = 0; i < out.extent(0); ++i)
    {
        for (std::size_t j = 0; j < out.extent(0); ++j)
            std::printf("%.7e ", out(i,j));
        std::printf("\n");
    }
}
