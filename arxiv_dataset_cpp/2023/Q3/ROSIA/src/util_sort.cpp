/////////////////////////////////////////////////////////////////////////////
//
//                          ROSIA
//
// This package contains the source code which implements the
// rotation-search-based star identification algorithm proposed in
//
// @article{chng2022rosia,
//  title={ROSIA: Rotation-Search-Based Star Identification Algorithm},
//  author={Chng, Chee-Kheng and Bustos, Alvaro Parra and McCarthy, Benjamin and Chin, Tat-Jun},
//  journal={arXiv preprint arXiv:2210.00429},
//  year={2022}
//}
//
// Copyright (c) 2023 Chee-Kheng Chng
// Australian Institute for Machine Learning (AIML)
// School of Computer Science, The University of Adelaide, Australia
// Please acknowledge the authors by citing the above paper in any academic
// publications that have made use of this package or part of it.
//
/////////////////////////////////////////////////////////////////////////////
#include "util_sort.h"

int reg::util::compare_value_index(const void * a, const void * b)
{
    struct value_index *vi1, *vi2;

    vi1 = (struct value_index*)a;
    vi2 = (struct value_index*)b;

    if (vi1->value > vi2->value)
    {
        return 1;
    }
    if (vi1->value < vi2->value)
    {
        return -1;
    }
    return 0;
}


int reg::util::compare_value_index_uint(const void * a, const void * b)
{
    struct value_index_uint *vi1, *vi2;

    vi1 = (struct value_index_uint*)a;
    vi2 = (struct value_index_uint*)b;

    if (vi1->value > vi2->value)
    {
        return 1;
    }
    if (vi1->value < vi2->value)
    {
        return -1;
    }
    return 0;
}


unsigned int* reg::util::sort_index(double *a, unsigned int len)
{
    unsigned int i;
    struct value_index *vi;
    unsigned int *res;

    vi = (struct value_index *)malloc(len*sizeof(struct value_index));
    for(i=0; i<len; i++)
    {
        vi[i].value=a[i];
        vi[i].index=i;
    }

    qsort((void *)(vi), len, sizeof(struct value_index), compare_value_index);
    res = (unsigned int *)malloc(len*sizeof(unsigned int));

    for(i=0; i<len; i++)
    {
        res[i] = vi[i].index;
        a[i]   = vi[i].value;
    }

    free(vi);
    return res;
}


unsigned int* reg::util::sort_index(unsigned int *a, unsigned int len)
{
//    mxAssert(len>=0,"");
    unsigned int i;
    struct value_index_uint *vi;
    unsigned int *res;

    vi = (struct value_index_uint *)malloc(len*sizeof(struct value_index_uint));
    for(i=0; i<len; i++)
    {
        vi[i].value=a[i];
        vi[i].index=i;
    }

    qsort((void *)(vi), len, sizeof(struct value_index_uint), compare_value_index_uint);
    res = (unsigned int *)malloc(len*sizeof(unsigned int));

    for(i=0; i<len; i++)
    {
        res[i] = vi[i].index;
        a[i]   = vi[i].value;
    }

    free(vi);
    return res;
}



void reg::util::sorted_by_index(double *a, unsigned int* idx, unsigned int len)
{
//    mxAssert(len>=0,"");

    unsigned int i;
    double *tmp;

    tmp = (double *)malloc(len*sizeof(double));
    for(i=0;i<len;i++)
    {
        tmp[i]=a[idx[i]];
    }
    memcpy(a, tmp, len*sizeof(double));
    free(tmp);
}


void reg::util::sorted_by_index(int *a, unsigned int* idx, unsigned int len)
{
//    mxAssert(len>=0, "");

    unsigned int i;
    int *tmp;

    tmp = (int *)malloc(len*sizeof(int));
    for(i=0;i<len;i++)
    {
        tmp[i]=a[idx[i]];
    }
    memcpy(a, tmp, len*sizeof(int));
    free(tmp);
}

void reg::util::sorted_by_index2(
        double *a, unsigned int* idx, unsigned int len, double *tmp)
{
//    mxAssert(len>=0,"");
    unsigned int i;
    for(i=0;i<len;i++)
    {
        tmp[i]=a[idx[i]];
    }
    memcpy(a, tmp, len*sizeof(double));
}

