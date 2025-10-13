#include "specOrthMat.hpp"

void SpecOrthMat::SchurAngular_SpecOrth()
{
    // The Schur decomposition of n x n matrices require n^2 workspace to operate, which is overwritten by the block upper triangular T
    // and the additional 2 n workspace to store the complex eigenvalues. In the case of special orthogonal matrix, there are not stored
    // as the equivalvent information, the angles, is stored in SchurAngularFactor::angles.
    REAL_TYPE *wr = saf.w;
    REAL_TYPE *wi = wr + d;
    REAL_TYPE *wS = wr + 2 * d;
    memcpy(wS, v, sizeof(REAL_TYPE) * d * d);

    saf.svec.fast_col_access();

    my_dgees('V', 'S', _dgees_select_nonreal, d, wS, d, &(saf.nzsize), wr, wi, saf.v, d);
    // auto view = View_ColMat<REAL_TYPE>(wS, d, d, d);
    // std::printf("Schur decomposition, quasi upper triangular matrix:\n");
    // view.printf();
    // std::printf("Schur decomposition, Schur vectors:\n");
    // saf.svec.printf();

    saf.nzsize = saf.nzsize / 2;
    REAL_TYPE *top_left_ptr = wS;
    for (INTE_TYPE i = 0; i < saf.nzsize; i++, top_left_ptr += (2 * d + 2))
        saf.a[i] = atan2(*(top_left_ptr + 1), *top_left_ptr);
}

void SchurAngular_SpecOrth(SchurAngularFactor &saf, const SpecOrthMat &MatQ)
{
    auto wr = saf.w;
    auto wi = wr + MatQ.d;
    auto wS = wr + 2 * MatQ.d;
    memcpy(wS, MatQ.v, sizeof(REAL_TYPE) * MatQ.d * MatQ.d);

    saf.svec.fast_col_access();

    my_dgees('V', 'S', _dgees_select_nonreal, MatQ.d, wS, MatQ.d, &(saf.nzsize), wr, wi, saf.v, MatQ.d);
    saf.nzsize = saf.nzsize / 2;
    REAL_TYPE *top_left_ptr = wS;
    for (INTE_TYPE i = 0; i < saf.nzsize; i++, top_left_ptr += (2 * MatQ.d + 2))
        saf.a[i] = atan2(*(top_left_ptr + 1), *top_left_ptr);
}
