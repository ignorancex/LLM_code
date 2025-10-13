#include "vecOp.hpp"

REAL_TYPE normsql2(INTE_TYPE n, const REAL_TYPE *v, INTE_TYPE incv)
{
    return my_ddot(n, v, incv, v, incv);
}
REAL_TYPE normsql2(INTE_TYPE n, const REAL_TYPE *v)
{
    return my_ddot(n, v, v);
}

REAL_TYPE norml2(INTE_TYPE n, const REAL_TYPE *v, INTE_TYPE incv)
{
    return sqrt(normsql2(n, v, incv));
}
REAL_TYPE norml2(INTE_TYPE n, const REAL_TYPE *v)
{
    return sqrt(normsql2(n, v));
}

void scal(INTE_TYPE n, REAL_TYPE scale, REAL_TYPE *v)
{
    my_dscal(n, scale, v, 1);
}