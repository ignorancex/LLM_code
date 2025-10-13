#include "daxpy.hpp"

void my_daxpy(INTE_TYPE n, REAL_TYPE a, REAL_TYPE *x, INTE_TYPE incx, REAL_TYPE *y, INTE_TYPE incy)
{
    cblas_daxpy(n, a, x, incx, y, incy);
}
void my_daxpy(INTE_TYPE n, REAL_TYPE a, REAL_TYPE *x, REAL_TYPE *y)
{
    cblas_daxpy(n, a, x, 1, y, 1);
}