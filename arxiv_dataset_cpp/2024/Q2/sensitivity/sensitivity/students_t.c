#include <stdio.h>
#include <math.h>
#include "students_t.h"

// Function prototypes
double betacf(double a, double b, double x);
double betai(double a, double b, double x);
double t_cdf(double t, int df);

// Continued fraction for the incomplete beta function
double betacf(double a, double b, double x) {
    int m, m2;
    double aa, c, d, del, h, qab, qam, qap;

    qab = a + b;
    qap = a + 1.0;
    qam = a - 1.0;
    c = 1.0;
    d = 1.0 - qab * x / qap;
    if (fabs(d) < FPMIN) d = FPMIN;
    d = 1.0 / d;
    h = d;
    for (m = 1; m <= MAXIT; m++) {
        m2 = 2 * m;
        aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (fabs(d) < FPMIN) d = FPMIN;
        c = 1.0 + aa / c;
        if (fabs(c) < FPMIN) c = FPMIN;
        d = 1.0 / d;
        del = d * c;
        h *= del;
        if (fabs(del - 1.0) < EPS) break;
    }
    return h;
}

// Incomplete beta function
double betai(double a, double b, double x) {
    double bt;
    if (x == 0.0 || x == 1.0) bt = 0.0;
    else
        bt = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1.0 - x));
    if (x < (a + 1.0) / (a + b + 2.0))
        return bt * betacf(a, b, x) / a;
    else
        return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

// Student's t-distribution CDF
double t_cdf(double t, int df) {
    double x = df / (df + t * t);
    return 1.0 - 0.5 * betai(0.5 * df, 0.5, x);
}

// Function to compute the p-value from Pearson's correlation coefficient r
double students_t(double r, int n) {
    int df = n - 2;
    double t = r * sqrt(df / (1 - r * r));
    double p_value = 2 * (1 - t_cdf(fabs(t), df));
    return p_value;
}
