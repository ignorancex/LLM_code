#include "dnrm2.hpp"

REAL_TYPE my_dnrm2(INTE_TYPE n, const REAL_TYPE *v, INTE_TYPE incv)
{
    return cblas_dnrm2(n, v, incv);
}
REAL_TYPE my_dnrm2(INTE_TYPE n, const REAL_TYPE *v)
{
    return cblas_dnrm2(n, v, 1);
}