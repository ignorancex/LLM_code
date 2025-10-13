#include "daxpby.hpp"

void my_daxpby(INTE_TYPE n, REAL_TYPE a, REAL_TYPE *x, INTE_TYPE incx, REAL_TYPE b, REAL_TYPE *y, INTE_TYPE incy)
{
	cblas_daxpby(n, a, x, incx, b, y, incy);
}
void my_daxpby(INTE_TYPE n, REAL_TYPE a, REAL_TYPE *x, REAL_TYPE b, REAL_TYPE *y)
{
	cblas_daxpby(n, a, x, 1, b, y, 1);
}
