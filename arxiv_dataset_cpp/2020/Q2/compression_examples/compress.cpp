#include <math.h>
#include <iostream>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdlib.h>

using namespace std;

template<int E, int M, int T, int P, int B = (1 << (E - 1)) - 1>
static inline uint64_t compress(float* xyz)
{
    static_assert(E + M + P + T == 64, "Invalid number of bits");
    
    const float rpis = 1.0/static_cast<float>(M_PI);
    const double told = 1e-13;

    float ts = atan2(static_cast<float>(xyz[1]),static_cast<float>(xyz[0]));
    uint64_t nt = round(static_cast<float>((1 << (T-1)) - 1)*(abs(ts)*rpis));
    if(ts >= 0) nt = nt | (1 << (T-1));

    double rd = sqrt((double)xyz[0]*xyz[0] + (double)xyz[1]*xyz[1] + (double)xyz[2]*xyz[2]);
    float rs = (float)rd;
    float pd = (rd < told) ? 0 : static_cast<float>(acos(static_cast<double>(xyz[2])/rd));
    uint64_t np = round(static_cast<float>((1 << P)-1)*pd*rpis);

    uint32_t* ptr = reinterpret_cast<uint32_t*>(&rs);
    uint64_t fre = (*ptr & 0x7f800000) >> 23;
    uint64_t frm = (*ptr & 0x007fffff) >> 23-M ;
    
    return nt | (np << (T)) | (frm << (T + P)) | ((fre - 127 + B) << (T + P + M));
    
}

template<int E, int M, int T, int P, int B = (1 << (E - 1)) - 1>
static inline void decompress(const uint64_t& C, float& x, float& y, float& z)
{
    static_assert(E + M + P + T == 64, "Invalid number of bits");
    
    const float pi = static_cast<float>(M_PI);
    
    uint32_t tm = (1 << T) - 1;
    uint32_t pm = (1 << P) - 1;
    uint32_t mm = (1 << M) - 1;
    uint32_t em = (1 << E) - 1;
    
    uint32_t td = C & tm;
    uint32_t pd = (C >> T) & pm;
    uint32_t frm = (C >> (T+P)) & mm;
    uint32_t fre = (C >> (T+P+M)) & em;
    uint32_t ri = ((fre - B + 127) << 23) | (frm << (23-M));
    
    float r = *reinterpret_cast<float*>(&ri);    
    float t = pi*(-1.0 + 2.0*(td >> (T-1))) *(td & (tm >> 1))/static_cast<float>((1 << (T-1))-1);    
    float p = pi*pd/static_cast<float>((1 << P)-1);
    
    x = r*cosf(t)*sinf(p);
    y = r*sinf(t)*sinf(p);
    z = r*cosf(p);
    
}
