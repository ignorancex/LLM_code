#include "dscal.hpp"

void my_dscal(INTE_TYPE n, REAL_TYPE s, REAL_TYPE *x, INTE_TYPE incx)
{
    cblas_dscal(n, s, x, incx);
}
void my_dscal(INTE_TYPE n, REAL_TYPE s, REAL_TYPE *x)
{
    cblas_dscal(n, s, x, 1);
}