/* GRTresna
 * Copyright 2024 The GRTL Collaboration.
 * Please refer to LICENSE in GRTresna's root directory.
 */

#include "DerivativeOperators.hpp"
#include "Interval.H"
#include "REAL.H"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"

void DerivativeOperators::get_d1(Tensor<1, Real, SpaceDim> &d1,
                                 const IntVect &a_iv,
                                 const FArrayBox &a_vars_box, const int icomp)
{
    FOR1(idir)
    {
        IntVect iv_offset1 = a_iv;
        IntVect iv_offset2 = a_iv;
        iv_offset1[idir] -= 1;
        iv_offset2[idir] += 1;

        d1[idir] =
            0.5 *
            (a_vars_box(iv_offset2, icomp) - a_vars_box(iv_offset1, icomp)) /
            m_dx[idir];
    }
}

void DerivativeOperators::get_d2(Tensor<2, Real, SpaceDim> &d2,
                                 const IntVect &a_iv,
                                 const FArrayBox &a_vars_box, const int icomp)
{
    FOR2(idir1, idir2)
    {
        if (idir1 != idir2)
        {
            IntVect iv_offset1 = a_iv;
            IntVect iv_offset2 = a_iv;
            IntVect iv_offset3 = a_iv;
            IntVect iv_offset4 = a_iv;
            iv_offset1[idir1] -= 1;
            iv_offset1[idir2] -= 1;
            iv_offset2[idir1] += 1;
            iv_offset2[idir2] += 1;
            iv_offset3[idir1] += 1;
            iv_offset3[idir2] -= 1;
            iv_offset4[idir1] -= 1;
            iv_offset4[idir2] += 1;

            d2[idir1][idir2] =
                (a_vars_box(iv_offset1, icomp) + a_vars_box(iv_offset2, icomp) -
                 a_vars_box(iv_offset3, icomp) -
                 a_vars_box(iv_offset4, icomp)) /
                (4.0 * m_dx[idir1] * m_dx[idir2]);
        }
        else
        {
            IntVect iv_offset1 = a_iv;
            IntVect iv_offset2 = a_iv;
            iv_offset1[idir1] -= 1;
            iv_offset2[idir1] += 1;

            d2[idir1][idir1] =
                (a_vars_box(iv_offset1, icomp) - 2.0 * a_vars_box(a_iv, icomp) +
                 a_vars_box(iv_offset2, icomp)) /
                (m_dx[idir1] * m_dx[idir1]);
        }
    }
}

void DerivativeOperators::get_d1_vector(Tensor<2, Real, SpaceDim> &d1,
                                        const IntVect &a_iv,
                                        const FArrayBox &a_vars_box,
                                        const Interval &a_interval)
{
    // get the derivs for each component
    Tensor<1, Real, SpaceDim> d1_comp1, d1_comp2, d1_comp3;
    get_d1(d1_comp1, a_iv, a_vars_box, a_interval.begin());
    get_d1(d1_comp2, a_iv, a_vars_box, a_interval.begin() + 1);
    get_d1(d1_comp3, a_iv, a_vars_box, a_interval.end());

    using namespace TensorAlgebra;
    FOR2(i, j)
    {
        d1[i][j] = delta(i, 0) * d1_comp1[j] + delta(i, 1) * d1_comp2[j] +
                   delta(i, 2) * d1_comp3[j];
    }
}

void DerivativeOperators::get_d2_vector(Tensor<3, Real, SpaceDim> &d2,
                                        const IntVect &a_iv,
                                        const FArrayBox &a_vars_box,
                                        const Interval &a_interval)
{
    // get the derivs for each component
    Tensor<2, Real, SpaceDim> d2_comp1, d2_comp2, d2_comp3;
    get_d2(d2_comp1, a_iv, a_vars_box, a_interval.begin());
    get_d2(d2_comp2, a_iv, a_vars_box, a_interval.begin() + 1);
    get_d2(d2_comp3, a_iv, a_vars_box, a_interval.end());

    using namespace TensorAlgebra;
    FOR3(i, j, k)
    {
        d2[i][j][k] = delta(i, 0) * d2_comp1[j][k] +
                      delta(i, 1) * d2_comp2[j][k] +
                      delta(i, 2) * d2_comp3[j][k];
    }
}

void DerivativeOperators::scalar_Laplacian(Real &laplacian, const IntVect &a_iv,
                                           const FArrayBox &a_vars_box,
                                           const int a_comp)
{
    laplacian = 0.0;
    FOR1(idir)
    {
        IntVect iv_offset1 = a_iv;
        IntVect iv_offset2 = a_iv;
        iv_offset1[idir] -= 1;
        iv_offset2[idir] += 1;

        // 2nd order stencil
        Real d2comp_dxdx = 1.0 / (m_dx[idir] * m_dx[idir]) *
                           (1.0 * a_vars_box(iv_offset2, a_comp) -
                            2.0 * a_vars_box(a_iv, a_comp) +
                            1.0 * a_vars_box(iv_offset1, a_comp));
        laplacian += d2comp_dxdx;
    }
}

void DerivativeOperators::vector_Laplacian(Tensor<1, Real, SpaceDim> &laplacian,
                                           const IntVect &a_iv,
                                           const FArrayBox &a_vars_box,
                                           const Interval &a_interval)
{
    Real laplacian1, laplacian2, laplacian3;
    scalar_Laplacian(laplacian1, a_iv, a_vars_box, a_interval.begin());
    scalar_Laplacian(laplacian2, a_iv, a_vars_box, a_interval.begin() + 1);
    scalar_Laplacian(laplacian3, a_iv, a_vars_box, a_interval.end());
    laplacian[0] = laplacian1;
    laplacian[1] = laplacian2;
    laplacian[2] = laplacian3;
}