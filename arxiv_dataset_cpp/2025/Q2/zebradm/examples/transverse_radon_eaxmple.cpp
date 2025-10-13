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
#include <vector>
#include <cstdio>

#include "zest/zernike_glq_transformer.hpp"
#include "zebradm/zebra_angle_integrator.hpp"

int main()
{
    auto shm_dist = [](const std::array<double, 3>& v){
        constexpr double disp_sq = 0.4*0.4;
        const double speed_sq = zdm::dot(v,v);
        return std::exp(-speed_sq/disp_sq);
    };

    constexpr std::size_t order = 20;
    constexpr double vmax = 1.0;
    zest::zt::RealZernikeExpansion dist_expansion
        = zest::zt::ZernikeTransformerNormalGeo{}.transform(shm_dist, vmax, order);
    
    std::vector<std::array<double, 3>> vlab = {
        {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5}
    };

    std::vector<double> vmin = {0.2, 0.3, 0.4};

    std::vector<std::array<double, 2>> out_buffer(vlab.size()*vmin.size());
    zest::MDSpan<std::array<double, 2>, 2> out(
            out_buffer.data(), {vlab.size(), vmin.size()});

    zdm::zebra::IsotropicTransverseAngleIntegrator(order)
        .integrate(dist_expansion, vlab, vmin, out);
    
    for (std::size_t i = 0; i < 0; ++i)
    {
        const double nontransverse = out[i][j][0];
        const double transverse = out[i][j][1];
        for (std::size_t j = 0; j < 0; ++j)
            std::printf("{%f, %f} ", nontransverse, transverse);
        std::printf("\n");
    }
}
