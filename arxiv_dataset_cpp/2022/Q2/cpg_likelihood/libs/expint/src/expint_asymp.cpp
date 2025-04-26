/*Copyright (c) 2016 Guillermo Navas-Palencia

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


/**
 * Asymptotic expansions for large x and v
 */

#include <cmath>

#include "expint.h"

namespace expint {

int use_expint_asymp_x(const double v, const double x)
{
  int i, n_max;
  double r;

  n_max = (int) ceil(x - v);

  r = 1.0;
  for (i = 1; i <= n_max; i++)
  {
    r *= (v + i - 1) / x;
    if (r < EXPINT_EPSILON)
      return i;
  }
  return 0;
}

double expint_asymp_x(const double v, const double x, const int n) 
{
  int i;
  double d, s, u;

  double v1 = v - 1.0;

  // series
  u = 1.0;
  d = x;
  s = 1.0 / x;

  for (i = 1; i <= n; i++) 
  {
    u *= -(v1 + i);
    d *= x;
    s += u / d;
  }
  return exp(-x) * s;
}


int use_expint_asymp_v(const double v, const double x) 
{
  int i, n_max;
  double r;

  n_max = (int) ceil(v - x - 1);

  r = 1.0 / (v - 1.0);
  for (i = 1; i <= n_max; i++) {
    r *= x / (v - 1.0 - i);
    if (r < EXPINT_EPSILON)
      return i;
  }
  return 0;
}


double expint_asymp_v(const double v, const double x, const double c, const int n) 
{
  int i;
  double d, s, u;
  double exp_mx = exp(-x);

  // series
  u = 1.0 / (v - 1.0);
  s = c/exp_mx + u;
  d = 1.0;

  for (i = 1; i <= n; i++) 
  {
    u /= (v - 1.0 - i);
    d *= -x;
    s += u * d;
  }
  return s * exp_mx;
}

}