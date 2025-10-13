#include "spectralFac.hpp"

void SpectralFactor::setSPF(const SchurAngularFactor &saf)
{
    auto a = saf.a;
    auto v = saf.v;

    spv = ColMat<CMPX_TYPE>(d, d);
    auto scv = saf.svec;

    REAL_TYPE scale = sqrt(2.0) * 0.5;
    // Complex part
    for (INTE_TYPE ang_ind = 0; ang_ind < saf.nzsize; ang_ind++)
    {
        // Eigenvalues 
        assignCMPX(val[2 * ang_ind], 0.0, a[ang_ind]);
        assignCMPX(val[2 * ang_ind + 1], 0.0, -a[ang_ind]);

        // Eigenvectors
        for (INTE_TYPE row_ind = 0; row_ind < d; row_ind++)
        {
            assignCMPX(spv(row_ind, 2 * ang_ind), scale * scv(row_ind, 2 * ang_ind), -scale * scv(row_ind, 2 * ang_ind + 1));
            assignCMPX(spv(row_ind, 2 * ang_ind + 1), scale * scv(row_ind, 2 * ang_ind), scale * scv(row_ind, 2 * ang_ind + 1));
        }
    }
    // Real part
    for (INTE_TYPE col_ind = 2 * saf.nzsize; col_ind < d; col_ind++)
    {
        assignCMPX(val[col_ind], 0.0, 0.0);
        for (INTE_TYPE row_ind = 0; row_ind < d; row_ind++)
            assignCMPX(spv(row_ind, col_ind), scv(row_ind, col_ind), 0.0);
    }
}


SpectralFactor::SpectralFactor(const SchurAngularFactor &saf) : SpectralFactor::SpectralFactor(saf.d, true)
{
    this->setSPF(saf);
}
