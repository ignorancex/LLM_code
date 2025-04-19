/*Copyright (c) 2017 Guillermo Navas-Palencia

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.*/

/*
** Source modified from original by ghcollin.
**  - Relevant expint functions extracted and modified to provide improved
**    numerical stability when subtracting a constant from the result.
**  - Functions placed in namespace.
*/

#include <cmath>

#include "expint.h"

namespace expint {

/**
 * Special cases for generalized exponential integral with integer nu.
 * iflag return 1 if special case is detected, otherwise 0.
 */
void expint_n_special_cases(const int n, const double x, int *iflag,
    double *result) 
{
  // x == 0
  if (x == 0.0) 
  {
    if (n <= 1) // exponential integral is not defined
      *result = EXPINT_INF;
    else
      *result =  1.0 / (n - 1);
    *iflag = 1;
  }

  // case n == 0
  if (n == 0) 
  {
    *result = exp(-x) / x;
    *iflag = 1;
  }

  // case n == 1
  if (n == 1) 
  {
    if (x > 0.9 && x < 10.0) 
    {
      *result = expinte1(x);
      *iflag = 1;
    }
  }
}

/**
 * Special cases for generalized exponential integral with double nu.
 * iflag return 1 if special case is detected, otherwise 0.
 */
void expint_v_special_cases(const double v, const double x, int *iflag,
    double *result) 
{
  double sqrt_x;

  // x == 0
  if (x == 0.0) 
  {
    if (v <= 1.0)   // exponential integral is not defined
      *result = EXPINT_INF;
    else
      *result = 1.0 / (v - 1.0);
    *iflag = 1;
  }

  // case v = 1/2
  if (v == 0.5 && x < 10.0) 
  {
    // x >= 10 some loss of digits is observed
    sqrt_x = sqrt(x);
    *result = EXPINT_SQRT_PI * erfc(sqrt_x) / sqrt_x;
    *iflag = 1;
  }
}


/**
 * Series expansion for small x (<= 1.5) and double v
 */
double expint_small_x(const double v, const double x, const double c) 
{
  if (v / x > 10.0) // fast convergence
    return expint_series_a(v, x, c);

  if (v > 1.5 && x > 0.5) // slow but accurate
    return expint_laguerre_series(v, x, c);
  else 
  {
    if (v < 0.9) // all terms of the expansion are positive
      return expint_series_b(v, x, c);
    else
      return expint_series_a(v, x, c);
  }
}

/**
 * Asymptotic expansion or Laguerre series for large v
 */
double expint_large_v(const double v, const double x, const double c) 
{
  int iter;

  // fixed and small x and large v
  if (x < 5.0) 
  {
    iter = use_expint_asymp_v(v, x);
    if (iter)
      return expint_asymp_v(v, x, c, iter);
  } 
  return expint_laguerre_series(v, x, c);
}

/**
 * Asymptotic expansion or Laguerre series for large x
 */
double expint_large_x(const double v, const double x) 
{
  int iter;

  // fixed v and large v
  if (x / v > 100.0) {
    iter = use_expint_asymp_x(v, x);
    if (iter)
      return expint_asymp_x(v, x, iter);
    else
      return expint_laguerre_series(v, x, 0);
  } 
  else
    return expint_laguerre_series(v, x, 0);
}

/**
 * Algorithm for the generalized exponential integral with integer nu
 */
double expint_n(const int n, const double x, const double c)
{
  int iflag = 0;
  double result = 0.0;

  // special cases
  expint_n_special_cases(n, x, &iflag, &result);
  if (iflag) return result + c;

  // small x
  if (x <= 1.5 && n < 20)
    return expint_series_n(n, x, c);
  else if (x <= 2.0 && n <= 10)
    // use series expansion in terms of E1(x) to avoid many iterations
    // laguerre series
    return expint_series_n_x_le_2(n, x, c);
  else if (n >= x)
    return expint_large_v(n, x, c);
  else
    return expint_large_x(n, x) + c;
}

/**
 * Algorithm for the generalized exponential integral with double nu
 */
double expint_v(const double v, const double x, const double c)
{
  int iflag = 0, vint;
  double result = 0.0;

  // v is integer
  vint = int(v);
  if (v == vint)
    return expint_n(vint, x, c);

  // special cases
  expint_v_special_cases(v, x, &iflag, &result);
  if (iflag) return result + c;

  // general cases
  if (x <= 1.0)
    return expint_small_x(v, x, c);
  else if (v >= x)
    return expint_large_v(v, x, c);
  else
    return expint_large_x(v, x) + c;
}


}