// #include <immintrin.h>
// #include <math.h>
// #include <stdint.h>
// #include <assert.h>

// #if defined(__GNUC__) || defined(__clang__)
// #   define RESTRICT   __restrict
// #   define ALIGNED(x) __attribute__((aligned(x)))
// #else
// #   define RESTRICT   restrict
// #   define ALIGNED(x)
// #endif

// static inline float sigmoidf_opt2(float x)
// {
//     return 1.0f / (1.0f + expf(-x));
// }

// static inline void scale_row(const float * RESTRICT in,
//     float       * RESTRICT out,
//     int NB, float s)
// {
//     if (NB == 4) {
//         /* one 128‑bit vector – fastest for NB=4 */
//         __m128 vs4 = _mm_set1_ps(s);
//         __m128 v   = _mm_loadu_ps(in);
//         _mm_storeu_ps(out, _mm_mul_ps(v, vs4));
//         return;
//     }

//     /* NB ≥ 8, fallback to AVX2 + scalar tail */
//     __m256 vs8 = _mm256_set1_ps(s);
//     int i = 0;
//     for (; i + 7 < NB; i += 8) {
//         __m256 v  = _mm256_loadu_ps(in + i);
//         v = _mm256_mul_ps(v, vs8);
//         _mm256_storeu_ps(out + i, v);
//     }
//     for (; i < NB; ++i) out[i] = in[i] * s;
// }

// /* build byte offsets for gathering one blade from 8 channels */
// #define V_OFFSETS(blade, NB, c0)  \
//     _mm256_set_epi32( (((c0)+7)*(NB)+(blade))<<2, (((c0)+6)*(NB)+(blade))<<2, \
//                       (((c0)+5)*(NB)+(blade))<<2, (((c0)+4)*(NB)+(blade))<<2, \
//                       (((c0)+3)*(NB)+(blade))<<2, (((c0)+2)*(NB)+(blade))<<2, \
//                       (((c0)+1)*(NB)+(blade))<<2, (((c0)+0)*(NB)+(blade))<<2 )

// #define W_OFFSETS(k, K, c0)       \
//     _mm256_set_epi32( (((c0)+7)*(K)+(k))<<2, (((c0)+6)*(K)+(k))<<2, \
//                       (((c0)+5)*(K)+(k))<<2, (((c0)+4)*(K)+(k))<<2, \
//                       (((c0)+3)*(K)+(k))<<2, (((c0)+2)*(K)+(k))<<2, \
//                       (((c0)+1)*(K)+(k))<<2, (((c0)+0)*(K)+(k))<<2 )

// /*  multivector_act_opt2.c
//  *
//  *  – Tuned for num_kernel_blades K ∈ {4, 8}
//  *  – 8 channels processed in parallel so that every AVX2 lane is useful
//  *  – Unrolled, branch-free dot/sum loops for K=4 and K=8
//  *  – Scalar sigmoid (avoids heavy AVX poly and clock throttle)
//  *  – Specialised scale‑out paths:
//  *        NB == 4 => single SSE (128‑bit) vector
//  *        NB >= 8 => AVX2 stream with scalar tail
//  */
// void multivector_act_forward(
//         const float *      RESTRICT v,
//         const float *      RESTRICT w,
//         const float *      RESTRICT bias,
//         int B,
//         int C,
//         int NB,
//         int K,
//         const int *        RESTRICT kidx,
//         int agg_mode,
//         float *            RESTRICT out)
// {
//     assert((NB == 4) || ((NB & 7) == 0));          /* 4 or multiple of 8 */
//     assert((K==4 || K==8));                        /* small‑K special */

//     const float inv_K = 1.0f / (float)K;

// for (int b = 0; b < B; ++b) {
//         const float *vB = v   + (size_t)b*C*NB;
//         float       *oB = out + (size_t)b*C*NB;

//         /* 8‑channel groups */
//         int c0 = 0;
//         for (; c0 + 7 < C; c0 += 8) {
//             __m256 acc0 = _mm256_setzero_ps();
//             __m256 acc1 = _mm256_setzero_ps();

//             if (K == 4) {
//                 /* blades 0‑3 unrolled */
//                 #define GATHER_BLADE(idx, acc)                                 \
//                 {                                                              \
//                     __m256 u = _mm256_i32gather_ps(vB,                         \
//                                   V_OFFSETS(kidx[idx], NB, c0), 1);            \
//                     if (agg_mode==0) {                                         \
//                         __m256 wv = _mm256_i32gather_ps(w,                     \
//                                      W_OFFSETS(idx, K, c0), 1);                \
//                         acc = _mm256_fmadd_ps(u, wv, acc);                     \
//                     } else {                                                   \
//                         acc = _mm256_add_ps(acc, u);                           \
//                     }                                                          \
//                 }
//                 GATHER_BLADE(0, acc0); GATHER_BLADE(1, acc0);
//                 GATHER_BLADE(2, acc0); GATHER_BLADE(3, acc0);
//                 #undef GATHER_BLADE
//             } else { /* K==8 */
//                 #define GATHER_BLADE(idx, tgt)                                 \
//                 {                                                              \
//                     __m256 u = _mm256_i32gather_ps(vB,                         \
//                                   V_OFFSETS(kidx[idx], NB, c0), 1);            \
//                     if (agg_mode==0) {                                         \
//                         __m256 wv = _mm256_i32gather_ps(w,                     \
//                                      W_OFFSETS(idx, K, c0), 1);                \
//                         tgt = _mm256_fmadd_ps(u, wv, tgt);                     \
//                     } else {                                                   \
//                         tgt = _mm256_add_ps(tgt, u);                           \
//                     }                                                          \
//                 }
//                 /* first half → acc0 */
//                 GATHER_BLADE(0, acc0); GATHER_BLADE(1, acc0);
//                 GATHER_BLADE(2, acc0); GATHER_BLADE(3, acc0);
//                 /* second half → acc1 */
//                 GATHER_BLADE(4, acc1); GATHER_BLADE(5, acc1);
//                 GATHER_BLADE(6, acc1); GATHER_BLADE(7, acc1);
//                 acc0 = _mm256_add_ps(acc0, acc1);
//                 #undef GATHER_BLADE
//             }

//             /* bias / mean */
//             if (agg_mode==0) {
//                 __m256 bvec = _mm256_loadu_ps(bias + c0);
//                 acc0 = _mm256_add_ps(acc0, bvec);
//             } else if (agg_mode==2) {
//                 acc0 = _mm256_mul_ps(acc0, _mm256_set1_ps(inv_K));
//             }

//             /* per‑lane sigmoid and broadcast scale */
//             float acts[8] ALIGNED(32); _mm256_storeu_ps(acts, acc0);
//             for (int i=0;i<8;++i) acts[i] = sigmoidf_opt2(acts[i]);
//             for (int j=0;j<8;++j) {
//                 scale_row(vB + (size_t)(c0+j)*NB,
//                           oB  + (size_t)(c0+j)*NB,
//                           NB, acts[j]);
//             }
//         }

//         /* ---- leftover channels (<8) scalar path ------------------------ */
//         for (int c=c0; c<C; ++c) {
//             float acc = 0.0f;
//             if (K==4) {
//                 float v0=vB[(size_t)c*NB + kidx[0]];
//                 float v1=vB[(size_t)c*NB + kidx[1]];
//                 float v2=vB[(size_t)c*NB + kidx[2]];
//                 float v3=vB[(size_t)c*NB + kidx[3]];
//                 if (agg_mode==0) {
//                     const float *wRow = w + (size_t)c*K;
//                     acc = v0*wRow[0] + v1*wRow[1] + v2*wRow[2] + v3*wRow[3];
//                 } else {
//                     acc = v0+v1+v2+v3;
//                 }
//             } else {
//                 /* K==8, still unrolled */
//                 const float *row = vB + (size_t)c*NB;
//                 float v0=row[kidx[0]], v1=row[kidx[1]], v2=row[kidx[2]], v3=row[kidx[3]],
//                       v4=row[kidx[4]], v5=row[kidx[5]], v6=row[kidx[6]], v7=row[kidx[7]];
//                 if (agg_mode==0) {
//                     const float *wRow = w + (size_t)c*K;
//                     acc = v0*wRow[0] + v1*wRow[1] + v2*wRow[2] + v3*wRow[3] +
//                           v4*wRow[4] + v5*wRow[5] + v6*wRow[6] + v7*wRow[7];
//                 } else {
//                     acc = v0+v1+v2+v3 + v4+v5+v6+v7;
//                 }
//             }
//             if (agg_mode==0)      acc += bias[c];
//             else if (agg_mode==2) acc *= inv_K;
//             float act = sigmoidf_opt2(acc);
//             scale_row(vB + (size_t)c*NB,
//                       oB  + (size_t)c*NB,
//                       NB, act);
//         }
//     }
// }

/*
 * multivector_act_intel.c  – Intel‐tuned, K==4 SSE/FMA + K==8 AVX2/FMA, no gathers in K==4
 * Builds on Ice Lake/U: -mavx2 -mfma -march=icelake -O3
 * Agg modes: 0=LINEAR (dot + bias), 1=SUM, 2=MEAN
 */

 #include <immintrin.h>
 #include <math.h>
 #include <stdint.h>
 #include <assert.h>
 
 #if defined(__GNUC__) || defined(__clang__)
 #   define RESTRICT   __restrict
 #   define ALIGNED(x) __attribute__((aligned(x)))
 #else
 #   define RESTRICT   restrict
 #   define ALIGNED(x)
 #endif
 
 // Scalar sigmoid
 static inline float sigmoidf(float x) {
     return 1.0f / (1.0f + expf(-x));
 }
 
 // Scale one row of NB floats by scalar s
 static inline void scale_row(
     const float * RESTRICT in,
     float       * RESTRICT out,
     int NB,
     float s
 ) {
     if (NB == 4) {
         __m128 vs = _mm_set1_ps(s);
         __m128 v  = _mm_loadu_ps(in);
         _mm_storeu_ps(out, _mm_mul_ps(v, vs));
         return;
     }
     // NB >= 8: AVX2
     __m256 vs8 = _mm256_set1_ps(s);
     int i = 0;
     for (; i + 7 < NB; i += 8) {
         __m256 v = _mm256_loadu_ps(in + i);
         v = _mm256_mul_ps(v, vs8);
         _mm256_storeu_ps(out + i, v);
     }
     for (; i < NB; ++i) out[i] = in[i] * s;
 }
 
 // Byte-offset macro for gather
 #define V_OFF(blade, NB, c0)                                         \
 _mm256_set_epi32(                                                     \
   (((c0)+7)*(NB)+(blade))<<2,                                          \
   (((c0)+6)*(NB)+(blade))<<2,                                          \
   (((c0)+5)*(NB)+(blade))<<2,                                          \
   (((c0)+4)*(NB)+(blade))<<2,                                          \
   (((c0)+3)*(NB)+(blade))<<2,                                          \
   (((c0)+2)*(NB)+(blade))<<2,                                          \
   (((c0)+1)*(NB)+(blade))<<2,                                          \
   (((c0)+0)*(NB)+(blade))<<2                                           \
 )
 
 void multivector_act_forward(
     const float *      RESTRICT v,     // [B][C][NB]
     const float *      RESTRICT w,     // [C][K]
     const float *      RESTRICT bias,  // [C]
     int B, int C, int NB, int K,
     const int *        RESTRICT kidx,  // [K]
     int agg_mode,                     // 0=LINEAR,1=SUM,2=MEAN
     float *            RESTRICT out    // [B][C][NB]
 ) {
     assert((K==4)||(K==8));
     assert((NB==4)||((NB & 7)==0));
     const float invK = (K==4 ? 1.0f/4.0f : 1.0f/8.0f);
 
     for (int b = 0; b < B; ++b) {
         const float *vB = v   + (size_t)b * C * NB;
         float       *oB = out + (size_t)b * C * NB;
 
         if (K == 4) {
             // SSE/FMA path, 4 channels at a time
             int c0 = 0;
             for (; c0 + 3 < C; c0 += 4) {
                 __m128 acc = _mm_setzero_ps();
                 if (agg_mode == 0) {
                     // dot-product: load weights + values
                     for (int k = 0; k < 4; ++k) {
                         float w0 = w[(c0+0)*4 + k];
                         float w1 = w[(c0+1)*4 + k];
                         float w2 = w[(c0+2)*4 + k];
                         float w3 = w[(c0+3)*4 + k];
                         __m128 wv = _mm_set_ps(w3, w2, w1, w0);
                         float v0 = vB[(c0+0)*NB + kidx[k]];
                         float v1 = vB[(c0+1)*NB + kidx[k]];
                         float v2 = vB[(c0+2)*NB + kidx[k]];
                         float v3 = vB[(c0+3)*NB + kidx[k]];
                         __m128 vv = _mm_set_ps(v3, v2, v1, v0);
                         acc = _mm_fmadd_ps(wv, vv, acc);
                     }
                 } else {
                     // SUM or MEAN: accumulate values only
                     for (int k = 0; k < 4; ++k) {
                         float v0 = vB[(c0+0)*NB + kidx[k]];
                         float v1 = vB[(c0+1)*NB + kidx[k]];
                         float v2 = vB[(c0+2)*NB + kidx[k]];
                         float v3 = vB[(c0+3)*NB + kidx[k]];
                         __m128 vv = _mm_set_ps(v3, v2, v1, v0);
                         acc = _mm_add_ps(acc, vv);
                     }
                 }
                 // add bias or scale for MEAN
                 if (agg_mode == 0) {
                     __m128 bv = _mm_loadu_ps(bias + c0);
                     acc = _mm_add_ps(acc, bv);
                 } else if (agg_mode == 2) {
                     __m128 sf = _mm_set1_ps(invK);
                     acc = _mm_mul_ps(acc, sf);
                 }
                 // sigmoid + write-out
                 float acts[4] ALIGNED(16);
                 _mm_storeu_ps(acts, acc);
                 for (int i = 0; i < 4; ++i)
                     acts[i] = sigmoidf(acts[i]);
                 for (int j = 0; j < 4; ++j)
                     scale_row(
                         vB + (c0+j)*NB,
                         oB + (c0+j)*NB,
                         NB, acts[j]
                     );
             }
             // scalar tail
             for (int c = c0; c < C; ++c) {
                 float acc = 0.0f;
                 if (agg_mode == 0) {
                     // weighted dot
                     for (int k = 0; k < 4; ++k)
                         acc += vB[c*NB + kidx[k]] * w[c*4 + k];
                     acc += bias[c];
                 } else {
                     // SUM / MEAN
                     for (int k = 0; k < 4; ++k)
                         acc += vB[c*NB + kidx[k]];
                     if (agg_mode == 2) acc *= invK;
                 }
                 float act = sigmoidf(acc);
                 scale_row(
                     vB + c*NB,
                     oB + c*NB,
                     NB, act
                 );
             }
 
         } else {
             // K == 8: AVX2/FMA path
             int c0 = 0;
             for (; c0 + 7 < C; c0 += 8) {
                 __m256 acc = _mm256_setzero_ps();
                 if (agg_mode == 0) {
                     // dot-product: gather v + gather w
                     for (int k = 0; k < 8; ++k) {
                         __m256 vv = _mm256_i32gather_ps(
                             vB,
                             V_OFF(kidx[k], NB, c0),
                             1
                         );
                         __m256 wv = _mm256_i32gather_ps(
                             w + c0*8 + k,  // pointer to w[c0][k]
                             _mm256_set_epi32(
                                 7,6,5,4,3,2,1,0
                             ),
                             8
                         );
                         acc = _mm256_fmadd_ps(vv, wv, acc);
                     }
                 } else {
                     // SUM / MEAN: gather v only
                     for (int k = 0; k < 8; ++k) {
                         __m256 vv = _mm256_i32gather_ps(
                             vB,
                             V_OFF(kidx[k], NB, c0),
                             1
                         );
                         acc = _mm256_add_ps(acc, vv);
                     }
                 }
                 // add bias or scale for MEAN
                 if (agg_mode == 0) {
                     __m256 bv = _mm256_loadu_ps(bias + c0);
                     acc = _mm256_add_ps(acc, bv);
                 } else if (agg_mode == 2) {
                     acc = _mm256_mul_ps(
                         acc,
                         _mm256_set1_ps(invK)
                     );
                 }
                 // sigmoid + store-out
                 float acts8[8] ALIGNED(32);
                 _mm256_storeu_ps(acts8, acc);
                 for (int i = 0; i < 8; ++i)
                     acts8[i] = sigmoidf(acts8[i]);
                 for (int j = 0; j < 8; ++j)
                     scale_row(
                         vB + (c0+j)*NB,
                         oB + (c0+j)*NB,
                         NB, acts8[j]
                     );
             }
             // tail
             for (int c = c0; c < C; ++c) {
                 float acc = 0.0f;
                 if (agg_mode == 0) {
                     for (int k = 0; k < 8; ++k)
                         acc += vB[c*NB + kidx[k]] * w[c*8 + k];
                     acc += bias[c];
                 } else {
                     for (int k = 0; k < 8; ++k)
                         acc += vB[c*NB + kidx[k]];
                     if (agg_mode == 2) acc *= invK;
                 }
                 float act = sigmoidf(acc);
                 scale_row(
                     vB + c*NB,
                     oB + c*NB,
                     NB, act
                 );
             }
         }
     }
 }
 