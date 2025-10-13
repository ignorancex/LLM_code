#include "ddot.hpp"

REAL_TYPE my_ddot(INTE_TYPE n, const REAL_TYPE *x, INTE_TYPE incx, const REAL_TYPE *y, INTE_TYPE incy)
{
    return cblas_ddot(n, x, incx, y, incy);
}
REAL_TYPE my_ddot(INTE_TYPE n, const REAL_TYPE *x, const REAL_TYPE *y)
{
    return cblas_ddot(n, x, 1, y, 1);
}