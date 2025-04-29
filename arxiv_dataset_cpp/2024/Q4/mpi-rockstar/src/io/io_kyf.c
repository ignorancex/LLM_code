#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <string.h>
#include "../check_syscalls.h"
#include "../universal_constants.h"
#include "stringparse.h"
#include "../config_vars.h"
#include "../config.h"
#include "../particle.h"

#define IO_CACHE_SIZE (2097152)

inline float swapFloat(const float f) {

    union {
        float fval;
        char  cval[4];
    } u;

    u.fval = f;

    char tmp;
    tmp       = u.cval[0];
    u.cval[0] = u.cval[3];
    u.cval[3] = tmp;
    tmp       = u.cval[1];
    u.cval[1] = u.cval[2];
    u.cval[2] = tmp;

    return u.fval;
}

#include <arpa/inet.h>
inline float swapFloat2(const float f) {
    union {
        float f;
        int   i;
    } pack;

    pack.f = f;
    pack.i = ntohl(pack.i);
    return pack.f;
}

inline int swapInt(const int i) {

    union {
        int  ival;
        char cval[4];
    } u;

    u.ival = i;

    char tmp;
    tmp       = u.cval[0];
    u.cval[0] = u.cval[3];
    u.cval[3] = tmp;
    tmp       = u.cval[1];
    u.cval[1] = u.cval[2];
    u.cval[2] = tmp;

    return u.ival;
}

inline double swapDouble(const double d) {

    union ud {
        double dval;
        char   cval[8];
    } u;

    u.dval = d;

    char tmp;
    tmp       = u.cval[0];
    u.cval[0] = u.cval[7];
    u.cval[7] = tmp;
    tmp       = u.cval[1];
    u.cval[1] = u.cval[6];
    u.cval[6] = tmp;
    tmp       = u.cval[2];
    u.cval[2] = u.cval[5];
    u.cval[5] = tmp;
    tmp       = u.cval[3];
    u.cval[3] = u.cval[4];
    u.cval[4] = tmp;

    return u.dval;
}

inline long long int swapLLInt(const long long int ll) {

    union ud {
        char          cval[8];
        long long int llval;
    } u;

    u.llval = ll;

    char tmp;
    tmp       = u.cval[0];
    u.cval[0] = u.cval[7];
    u.cval[7] = tmp;
    tmp       = u.cval[1];
    u.cval[1] = u.cval[6];
    u.cval[6] = tmp;
    tmp       = u.cval[2];
    u.cval[2] = u.cval[5];
    u.cval[5] = tmp;
    tmp       = u.cval[3];
    u.cval[3] = u.cval[4];
    u.cval[4] = tmp;

    return u.llval;
}

inline int swapShort(const short s) {

    union {
        short sval;
        char  cval[2];
    } u;

    u.sval = s;

    char tmp;
    tmp       = u.cval[0];
    u.cval[0] = u.cval[1];
    u.cval[1] = tmp;

    return u.sval;
}

void load_particles_kyf(char *filename, struct particle **p, int64_t *num_p) {

    FILE *fin = check_fopen(filename, "rb");

    int    io_ver;
    float  omega0, omegab, lambda0, hubble;
    float  astart, anow, tnow;
    double lunit, munit, tunit;

    check_fread(&io_ver, sizeof(int), 1, fin);
#ifdef REVERSE_ENDIAN_INPUT
    io_ver = swapInt(io_ver);
#endif

    int ntmp1, ntmp2;
    check_fread(&ntmp1, sizeof(int), 1, fin);
    check_fread(&ntmp2, sizeof(int), 1, fin);
#ifdef REVERSE_ENDIAN_INPUT
    int npart = (long long int)swapInt(ntmp1);
    //int ngas  = (long long int)swapInt(ntmp2);
#else
    int npart = (long long int)ntmp1;
    //int ngas  = (long long int)ntmp2;
#endif

    check_fread(&omega0, sizeof(float), 1, fin);
    check_fread(&omegab, sizeof(float), 1, fin);
    check_fread(&lambda0, sizeof(float), 1, fin);
    check_fread(&hubble, sizeof(float), 1, fin);
    check_fread(&astart, sizeof(float), 1, fin);
    check_fread(&anow, sizeof(float), 1, fin);
    check_fread(&tnow, sizeof(float), 1, fin);
    check_fread(&lunit, sizeof(double), 1, fin);
    check_fread(&munit, sizeof(double), 1, fin);
    check_fread(&tunit, sizeof(double), 1, fin);
#ifdef REVERSE_ENDIAN_INPUT
    omega0  = swapFloat(omega0);
    omegab  = swapFloat(omegab);
    lambda0 = swapFloat(lambda0);
    hubble  = swapFloat(hubble);
    astart  = swapFloat(astart);
    anow    = swapFloat(anow);
    tnow    = swapFloat(tnow);
    lunit   = swapDouble(lunit);
    munit   = swapDouble(munit);
    tunit   = swapDouble(tunit);
#endif

    // lunit *= LUNIT_SCALE;

    double lunitm = lunit * 1.0e6 * 3.08568025e13;
    double km_s   = tunit / lunitm;

#if 0
    double znow   = (float)(1.0 / anow - 1.0);
    static int first_call = 0;
    if( first_call == 0){
      fprintf( stderr, "omega0:%e omegab:%e lambda0:%e hubble:%e\n",
	       omega0, omegab, lambda0, hubble);
      fprintf( stderr, "astart:%e anow:%e tnow:%e\n",
	       astart, anow, tnow);
      fprintf( stderr, "lunit:%e munit:%e tunit:%e znow:%e\n",
	       lunit, munit, tunit, znow);
      first_call = 1;
    }
#endif

    Ol = lambda0;
    Om = omega0;
    ;
    h0            = hubble;
    BOX_SIZE      = lunit * hubble;
    SCALE_NOW     = anow;
    PARTICLE_MASS = Om * CRITICAL_DENSITY * pow(BOX_SIZE, 3) / TOTAL_PARTICLES;
    AVG_PARTICLE_SPACING = cbrt(PARTICLE_MASS / (Om * CRITICAL_DENSITY));

#if 0
  fprintf( stderr, "-npart= %d\n", npart);
  fprintf( stderr, "-Ol= %e\n", Ol);
  fprintf( stderr, "-Om= %e\n", Om);
  fprintf( stderr, "-h0= %e\n", h0);
  fprintf( stderr, "-BOX_SIZE= %e\n", BOX_SIZE);
  fprintf( stderr, "-SCALE_NOW= %e\n", SCALE_NOW);
  fprintf( stderr, "-PARTICLE_MASS= %e\n", PARTICLE_MASS);
  fprintf( stderr, "-TOTAL_PARTICLES= %lld\n", TOTAL_PARTICLES);
  fprintf( stderr, "-AVG_PARTICLE_SPACING= %e\n", AVG_PARTICLE_SPACING);
#endif

    double lunit_gadget = lunit * hubble;
    // double munit_gadget = munit * hubble / 1.0e10;
    // double vunit_gadget = sqrt(anow) / km_s;
    double vunit_gadget = anow / km_s;

    *p = (struct particle *)check_realloc(
        *p, ((*num_p) + npart) * sizeof(struct particle),
        "Allocating particles.");

    int nmemory = IO_CACHE_SIZE;

    float *fcache = (float *)malloc(sizeof(float) * nmemory * 3);
    for (int ii = 0; ii < npart; ii += nmemory) {
        int nread = nmemory;
        if ((ii + nmemory) > npart)
            nread = npart - ii;
        check_fread(fcache, sizeof(float), nread * 3, fin);
        for (int iii = 0; iii < nread; iii++) {
            // int ip = ii + iii;
            int ip = ii + iii + *num_p;
#ifdef REVERSE_ENDIAN_INPUT
            (*p)[ip].pos[0] = swapFloat(fcache[3 * iii + 0]);
            (*p)[ip].pos[1] = swapFloat(fcache[3 * iii + 1]);
            (*p)[ip].pos[2] = swapFloat(fcache[3 * iii + 2]);
#else
            (*p)[ip].pos[0] = fcache[3 * iii + 0];
            (*p)[ip].pos[1] = fcache[3 * iii + 1];
            (*p)[ip].pos[2] = fcache[3 * iii + 2];
#endif
            (*p)[ip].pos[0] *= lunit_gadget;
            (*p)[ip].pos[1] *= lunit_gadget;
            (*p)[ip].pos[2] *= lunit_gadget;
        }
    }

    for (int ii = 0; ii < npart; ii += nmemory) {
        int nread = nmemory;
        if ((ii + nmemory) > npart)
            nread = npart - ii;
        check_fread(fcache, sizeof(float), nread * 3, fin);
        for (int iii = 0; iii < nread; iii++) {
            // int ip = ii + iii;
            int ip = ii + iii + *num_p;
#ifdef REVERSE_ENDIAN_INPUT
            (*p)[ip].pos[3] = swapFloat(fcache[3 * iii + 0]);
            (*p)[ip].pos[4] = swapFloat(fcache[3 * iii + 1]);
            (*p)[ip].pos[5] = swapFloat(fcache[3 * iii + 2]);
#else
            (*p)[ip].pos[3] = fcache[3 * iii + 0];
            (*p)[ip].pos[4] = fcache[3 * iii + 1];
            (*p)[ip].pos[5] = fcache[3 * iii + 2];
#endif
            (*p)[ip].pos[3] *= vunit_gadget;
            (*p)[ip].pos[4] *= vunit_gadget;
            (*p)[ip].pos[5] *= vunit_gadget;
        }
    }

    free(fcache);
    int64_t *llicache = (int64_t *)malloc(sizeof(int64_t) * nmemory);

    for (int ii = 0; ii < npart; ii += nmemory) {
        int nread = nmemory;
        if ((ii + nmemory) > npart)
            nread = npart - ii;
        check_fread(llicache, sizeof(long long int), nread, fin);
        for (int iii = 0; iii < nread; iii++) {
            // int ip = ii + iii;
            int ip = ii + iii + *num_p;
#ifdef REVERSE_ENDIAN_INPUT
            (*p)[ip].id = swapLLInt(llicache[iii]);
#else
            (*p)[ip].id     = llicache[iii];
#endif
        }
    }

    free(llicache);

    //(*num_p) = npart;
    (*num_p) += npart;

    fclose(fin);
}
