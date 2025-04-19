#include "test_all.h"

int test_all(int result){
  //result |= test_blas(result);
  result |= test_my_zgemm(result);
  result |= test_inv2by2(result);
  result |= test_inv2by2inv2by2_overwrite(result);

  result |= test_outer(result);
  result |= test_my_zaxpy(result);
  result |= test_my_zdot(result);

  return result;
}
