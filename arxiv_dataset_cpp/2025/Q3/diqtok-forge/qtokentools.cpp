#include "qtokentools.h"

double binomial(const int n, int k) {
    //calculates the biniomial coefficient n over k
	int i{ 0 };
	double result{ 1. };
    if (k > n - k) {
        k = n - k;
    }
    for (i = 1; i <= k; ++i) {
        result = result * (n - i + 1) / i;
    }
	return (result);
}


inline double pq(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return p1 + 0.5 * (p0 - p1) * (1. + cos(theta1) * cos(theta2) + cos(phi1 - phi2) * sin(theta1) * sin(theta2));
}

inline double dpqdtheta1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * (cos(theta2) * sin(theta1) - cos(phi1 - phi2) * cos(theta1) * sin(theta2));
}

inline double dpqdphi1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * sin(phi1 - phi2) * sin(theta1) * sin(theta2);
}

inline double dpqdtheta2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * (cos(phi1 - phi2) * cos(theta2) * sin(theta1) - cos(theta1) * sin(theta2));
}

inline double dpqdphi2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * sin(phi1 - phi2) * sin(theta1) * sin(theta2);
}

inline double d2pqdtheta1dtheta1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * (cos(theta1) * cos(theta2) + cos(phi1 - phi2) * sin(theta1) * sin(theta2));
}

inline double d2pqdtheta1dphi1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * cos(theta1) * sin(phi1 - phi2) * sin(theta2);
}

inline double d2pqdtheta1dtheta2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * (cos(phi1 - phi2) * cos(theta1) * cos(theta2) + sin(theta1) * sin(theta2));
}

inline double d2pqdtheta1dphi2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * cos(theta1) * sin(phi1 - phi2) * sin(theta2);
}

inline double d2pqdphi1dtheta1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * cos(theta1) * sin(phi1 - phi2) * sin(theta2);
}

inline double d2pqdphi1dphi1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * cos(phi1 - phi2) * sin(theta1) * sin(theta2);
}

inline double d2pqdphi1dtheta2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * cos(theta2) * sin(phi1 - phi2) * sin(theta1);
}

inline double d2pqdphi1dphi2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * cos(phi1 - phi2) * sin(theta1) * sin(theta2);
}

inline double d2pqdtheta2dtheta1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * (cos(phi1 - phi2) * cos(theta1) * cos(theta2) + sin(theta1) * sin(theta2));
}

inline double d2pqdtheta2dphi1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * cos(theta2) * sin(phi1 - phi2) * sin(theta1);
}

inline double d2pqdtheta2dtheta2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * (cos(theta1) * cos(theta2) + cos(phi1 - phi2) * sin(theta1) * sin(theta2));
}

inline double d2pqdtheta2dphi2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * cos(theta2) * sin(phi1 - phi2) * sin(theta1);
}

inline double d2pqdphi2dtheta1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * cos(theta1) * sin(phi1 - phi2) * sin(theta2);
}

inline double d2pqdphi2dphi1(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * cos(phi1 - phi2) * sin(theta1) * sin(theta2);
}

inline double d2pqdphi2dtheta2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return 0.5 * (p0 - p1) * cos(theta2) * sin(phi1 - phi2) * sin(theta1);
}

inline double d2pqdphi2dphi2(double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    return -0.5 * (p0 - p1) * cos(phi1 - phi2) * sin(theta1) * sin(theta2);
}

double pt(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    if (k == 0 && nq == 1) {
        return 1 - p;
    }
    if (k == 0 && nq == 2) {
        return pow(-1 + p, 2);
    }
    if (k == 0 && nq > 2) {
        return pow(1 - p, nq);
    }
    if (k == 1 && nq == 1) {
        return p;
    }
    if (k == 1 && nq == 2) {
        return -2 * (-1 + p) * p;
    }
    if (k == 1 && nq > 2) {
        return nq * pow(1 - p, -1 + nq) * p;
    }
    if (k == -1 + nq && nq == 1) {
        return 1 - p;
    }
    if (k == -1 + nq && nq == 2) {
        return -2 * (-1 + p) * p;
    }
    if (k == -1 + nq && nq > 2) {
        return -(nq * (-1 + p) * pow(p, -1 + nq));
    }
    if (k == nq && nq == 1) {
        return p;
    }
    if (k == nq && nq == 2) {
        return pow(p, 2);
    }
    if (k == nq && nq > 2) {
        return pow(p, nq);
    }
    return pow(1 - p, -k + nq) * pow(p, k) * binomial(nq, k);
}

double dptdtheta1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(1 - p, -1 + nq) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(1 - p, -2 + nq) * (-1 + nq * p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -2 + nq) * (1 - nq + nq * p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq == 1) {
        return dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * p * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -1 + nq) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return -(pow(1 - p, -1 - k + nq) * pow(p, -1 + k) * (-k + nq * p) * binomial(nq, k) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
}

double dptdphi1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(1 - p, -1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(1 - p, -2 + nq) * (-1 + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -2 + nq) * (1 - nq + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq == 1) {
        return dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * p * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return -(pow(1 - p, -1 - k + nq) * pow(p, -1 + k) * (-k + nq * p) * binomial(nq, k) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2));
}

double dptdtheta2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(1 - p, -1 + nq) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(1 - p, -2 + nq) * (-1 + nq * p) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -2 + nq) * (1 - nq + nq * p) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq == 1) {
        return dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * p * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -1 + nq) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return -(pow(1 - p, -1 - k + nq) * pow(p, -1 + k) * (-k + nq * p) * binomial(nq, k) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
}

double dptdphi2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(1 - p, -1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(1 - p, -2 + nq) * (-1 + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -2 + nq) * (1 - nq + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq == 1) {
        return dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * p * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return -(pow(1 - p, -1 - k + nq) * pow(p, -1 + k) * (-k + nq * p) * binomial(nq, k) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdtheta1dtheta1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * ((-1 + p) * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) + pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) - 4 * pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) - 4 * pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) + pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta1dtheta1(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * pow(dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2), 2));
}

double d2ptdtheta1dphi1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdtheta1dtheta2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdtheta1dphi2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdphi1dtheta1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta1dphi1(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdphi1dphi1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * ((-1 + p) * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) + pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) - 4 * pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) - 4 * pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) + pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdphi1dphi1(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * pow(dpqdphi1(p0, p1, theta1, phi1, theta2, phi2), 2));
}

double d2ptdphi1dtheta2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdphi1dphi2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdtheta2dtheta1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdtheta2dphi1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdphi1dtheta2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdtheta2dtheta2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * ((-1 + p) * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) + pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) - 4 * pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) + pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta2dtheta2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * pow(dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2), 2));
}

double d2ptdtheta2dphi2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdphi2dtheta1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta1dphi2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta1(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdphi2dphi1(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdphi1dphi2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi1(p0, p1, theta1, phi1, theta2, phi2) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdphi2dtheta2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (-1 + p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + 2 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdtheta2dphi2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * dpqdphi2(p0, p1, theta1, phi1, theta2, phi2) * dpqdtheta2(p0, p1, theta1, phi1, theta2, phi2));
}

double d2ptdphi2dphi2(int const nq, int const k, double const p0, double const p1, double const theta1, double const phi1, double const theta2, double const phi2) {
    if (k == 0 && nq == 1) {
        return -d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 0 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * ((-1 + p) * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) + pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == 0 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -2 + nq) * ((-1 + p) * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == 1 && nq == 1) {
        return d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == 1 && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2);
    }
    if (k == 1 && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(1 - p, -3 + nq) * ((-1 + p) * (-1 + nq * p) * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (-2 + nq * p) * pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == -1 + nq && nq == 1) {
        return -d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == -1 + nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (1 - 2 * p) * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) - 4 * pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2);
    }
    if (k == -1 + nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return -(nq * pow(p, -3 + nq) * (p * (1 - nq + nq * p) * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * (2 - nq + nq * p) * pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2)));
    }
    if (k == nq && nq == 1) {
        return d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2);
    }
    if (k == nq && nq == 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return 2 * (p * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) + pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    if (k == nq && nq > 2) {
        double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
        return nq * pow(p, -2 + nq) * (p * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) + (-1 + nq) * pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2));
    }
    double p{ pq(p0, p1, theta1, phi1, theta2, phi2) };
    return pow(1 - p, -2 - k + nq) * pow(p, -2 + k) * binomial(nq, k) * ((-1 + p) * p * (-k + nq * p) * d2pqdphi2dphi2(p0, p1, theta1, phi1, theta2, phi2) + ((-1 + k) * k - 2 * k * (-1 + nq) * p + (-1 + nq) * nq * pow(p, 2)) * pow(dpqdphi2(p0, p1, theta1, phi1, theta2, phi2), 2));
}


void adjust_into_interval(double const a, double const b, double& v) {
	// adjusts the number v into the interval [a,b]
	if (v > b)
		v = b;
	else if (v < a)
		v = a;
}


void ml_estimator_1m(double const p0, double const p1, int const N1, int const nf1, double& thetaf) {
	// maxium likelihood estimator for 1 measurement using thetaf1 = 0, phif1 = 0
	// in order to determine thetaf
	double af1{ (2. * nf1 / N1 - p0 - p1) / (p0 - p1) };
	adjust_into_interval(-1., 1., af1);
	thetaf = acos(af1);
}

void ml_estimator_2m(double const p0, double const p1, int const N1, int const N2, int const nf1, int const nf2, double const thetaf2, double& thetaf, double& phif) {
	// maximum likelihood estimator for 2 measurements, 1. measurement using thetaf1 = 0, phif1 = 0
	// 2. measurement at thetaf2 and phif2=0
	// in order to determine thetaf, phif
	double af1{ (2. * nf1 / N1 - p0 - p1) / (p0 - p1) };
	double af2{ (2. * nf2 / N2 - p0 - p1) / (p0 - p1) };
	adjust_into_interval(-1., 1., af1);
	thetaf = acos(af1);
	if (af1 > 1. - 1.0E-14 || af1 < -1 + 1.0E-14) {
		phif = 0.;
	} 
	else {
        double cosphif{ (af2 - cos(thetaf2) * af1) / (sin(thetaf2) * sqrt(1 - af1*af1)) };
        adjust_into_interval(-1., 1., cosphif);
        phif = acos(cosphif);
	}
}

void ml_estimator_3m(double const p0, double const p1, int const N1, int const N2, int const N3, int const nf1, int const nf2, int const nf3,
    double const thetaf2, double const thetaf3, double const phif3, double& thetaf, double& phif) {
    // maximum likelihood estimator for 3 measurements, 1. measurement using thetaf1 = 0, phif1 = 0
    // 2. measurement at thetaf2 and phif2=0
    // 3. measurement at thetaf3 and phif3
    // in order to determine thetaf, phif
    double af1{ (2. * nf1 / N1 - p0 - p1) / (p0 - p1) };
    double af2{ (2. * nf2 / N2 - p0 - p1) / (p0 - p1) };
    double af3{ (2. * nf3 / N3 - p0 - p1) / (p0 - p1) };
    adjust_into_interval(-1., 1., af1);
    thetaf = acos(af1);
    if (af1 > 1. - 1.0E-14 || af1 < -1 + 1.0E-14) {
        phif = 0.;
    }
    else {
        double cosphif{ (af2 - cos(thetaf2) * af1) / (sin(thetaf2) * sqrt(1 - af1 * af1)) };
        double termp, termm;
        adjust_into_interval(-1., 1., cosphif);
        phif = acos(cosphif);
        termp = (af3 - cos(thetaf3) * af1) / (sin(thetaf3) * sin(phif3) * sqrt(1 - af1 * af1)) - cos(phif3) * cos(phif) / sin(phif3) - sin(phif);
        termm = (af3 - cos(thetaf3) * af1) / (sin(thetaf3) * sin(phif3) * sqrt(1 - af1 * af1)) - cos(phif3) * cos(-phif) / sin(phif3) - sin(-phif);
        if (std::abs(termm) < std::abs(termp)) {
            phif = -phif;
        }
    }
}

void direct_inversion_tomography(int const N1, int const N2, int const N3, int const nf1, int const nf2, int const nf3, double& thetaf, double& phif) {
    //direct inversion tomography for 3 measurements: 1. measurement using thetaf1=0, phif1=0 (z-axis)
    // 2. measurement thetaf2 = pi/2, phif1 = 0 (x-axis)
    // 3. measurement thetaf3 = pi/2, phif3 = pi/2 (y-axis)
    // in order to determin thetaf, phif
    double rsdx{ (-2.*nf2 + N2) / N2 };
    double rsdy{ (-2.*nf3 + N3) / N3 };
    double rsdz{ (-2.*nf1 + N1) / N1 };
    double rsd{ sqrt(rsdx * rsdx + rsdy * rsdy + rsdz * rsdz) };
    if (rsd > 0.) {
        rsd = 1. / rsd;
        rsdx *= rsd;
        rsdy *= rsd;
        rsdz *= rsd;

        thetaf = acos(rsdz);
        if (std::abs(thetaf) < 1.0E-14 || std::abs(thetaf - M_PI) < 1.0E-14) {
            phif = 0.;
        }
        else
        {
            double Isinthetaf{ 1. / sin(thetaf) };
            phif = atan2(rsdx * Isinthetaf, rsdy * Isinthetaf);
        }
    }
    else {
        thetaf = 0.;
        phif = 0.;
    }
}