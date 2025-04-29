#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpointer-sign"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include "Expression_Kernels.h"

static int CoreCountDynamic = 1;
static int ActiveCore = gap_ncore();

static inline unsigned int __attribute__((always_inline)) ChunkSize(unsigned int X)

{
	unsigned int NCore;
	unsigned int Log2Core;
	unsigned int Chunk;

	if (CoreCountDynamic) NCore = ActiveCore; else NCore = gap_ncore();
	Log2Core = gap_fl1(NCore);
	Chunk = (X>>Log2Core) + ((X&(NCore-1))!=0);
	return Chunk;
}

#ifndef AT_NORM
#define AT_NORM(x, n)   gap_roundnorm_reg((x), (n))
#endif
#define ATLShift(x, n)  ((x) << (n))

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s373_kernel(s373_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_0_in_0 = Args->expr_0_in_0; // (1, 1, 2574) int8 0.004 Q7
    signed char *__restrict__  expr_0_in_1 = Args->expr_0_in_1; // (1, 1, 2574) int8 322.520 Q7
    signed char *__restrict__  expr_0_in_2 = Args->expr_0_in_2; // (1, 1, 2574) int8 314.457 Q7
    signed char *__restrict__  expr_0_in_3 = Args->expr_0_in_3; // (1, 1, 2574) int8 258.016 Q7
    signed char *__restrict__  expr_0_in_4 = Args->expr_0_in_4; // (1, 1, 2574) int8 0.004 Q7
    signed char *__restrict__  expr_0_out_0 = Args->expr_0_out_0; // (1, 1, 2574) int8 299.696 Q7
    signed char *__restrict__  expr_0_out_1 = Args->expr_0_out_1; // (1, 1, 2574) int8 402.094 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 1, 2574) var shapes:
    // inter__Add_2: (1, 1, 2574) expr_0_in_0: (1, 1, 2574) expr_0_in_1: (1, 1,
    // 2574) expr_0_in_2: (1, 1, 2574) inter__Mul_7: (1, 1, 2574) expr_0_in_4:
    // (1, 1, 2574) expr_0_in_3: (1, 1, 2574) expr_0_out_0: (1, 1, 2574)
    // expr_0_out_1: (1, 1, 2574)
    // Iteration reduced to spaces ((2,),)
    // Fixed spaces ()
    // Parameteric spaces ((2,),)
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_0_in_0: int8 0.004 Q7 expr_0_in_1: int8 322.520 Q7
        // expr_0_in_2: int8 314.457 Q7
        // inter__Add_2 = Add(Norm(Mul(Mul(Cast(expr_0_in_0, int32), Cast(expr_0_in_1, int32)), [136]), [22]), Cast(expr_0_in_2, int32))
        int inter__Add_2 = (gap_roundnorm_reg(((((int)expr_0_in_0[i0])*((int)expr_0_in_1[i0]))*(136)), (22))+((int)expr_0_in_2[i0]));
        // inputs expr_0_in_4: int8 0.004 Q7 expr_0_in_3: int8 258.016 Q7
        // inter__Mul_7 = Mul([1], Norm(Mul(Norm(Mul(Cast(Exp1715(Norm(Mul(GapMin([131072], GapMax([-31719424], Cast(expr_0_in_4, int32))), [132]), [7])), int32), [255]), [8]), Cast(expr_0_in_3, int32)), [7]))
        // Min0 scale arg 0 to 1 - int32 4.135 Q0 -> int32 0.004 Q7
        int _SYMBOL_Constant2201 = (131072);
        // Max0 scale arg 0 to 1 - int32 -1000.000 Q0 -> int32 0.004 Q7
        int _SYMBOL_Constant2199 = (-31719424);
        int inter__Mul_7 = ((1)*gap_roundnorm_reg((gap_roundnorm_reg((((int)exp_17_15(gap_roundnorm_reg((gap_min(_SYMBOL_Constant2201, gap_max(_SYMBOL_Constant2199, ((int)expr_0_in_4[i0])))*(132)), (7))))*(255)), (8))*((int)expr_0_in_3[i0])), (7)));
        // inputs inter__Add_2: int32 314.457 Q7 inter__Mul_7: int32 129.492
        // Q15
        // expr_0_out_0 = Cast(Clip(Norm(Mul(Sub(inter__Add_2, Norm(Mul(inter__Mul_7, [211]), [17])), [134]), [7]), -128, 127), int8)
        expr_0_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((inter__Add_2-gap_roundnorm_reg((inter__Mul_7*(211)), (17)))*(134)), (7))), ((7))));
        // inputs inter__Add_2: int32 314.457 Q7 inter__Mul_7: int32 129.492
        // Q15
        // expr_0_out_1 = Cast(Clip(Norm(Mul(Add(inter__Add_2, Norm(Mul(inter__Mul_7, [211]), [17])), [200]), [8]), -128, 127), int8)
        expr_0_out_1[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((inter__Add_2+gap_roundnorm_reg((inter__Mul_7*(211)), (17)))*(200)), (8))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s367_kernel(s367_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_1_in_0 = Args->expr_1_in_0; // (1, 1, 2574) int8 0.004 Q7
    signed char *__restrict__  expr_1_in_1 = Args->expr_1_in_1; // (1, 1, 2574) int8 258.016 Q7
    signed char *__restrict__  expr_1_in_2 = Args->expr_1_in_2; // (1, 1, 2574) int8 249.953 Q7
    signed char *__restrict__  expr_1_in_3 = Args->expr_1_in_3; // (1, 1, 2574) int8 258.016 Q7
    signed char *__restrict__  expr_1_in_4 = Args->expr_1_in_4; // (1, 1, 2574) int8 0.004 Q7
    signed char *__restrict__  expr_1_out_0 = Args->expr_1_out_0; // (1, 1, 2574) int8 235.112 Q7
    signed char *__restrict__  expr_1_out_1 = Args->expr_1_out_1; // (1, 1, 2574) int8 337.627 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 1, 2574) var shapes:
    // inter__Add_3: (1, 1, 2574) expr_1_in_0: (1, 1, 2574) expr_1_in_1: (1, 1,
    // 2574) expr_1_in_2: (1, 1, 2574) inter__Mul_6: (1, 1, 2574) expr_1_in_4:
    // (1, 1, 2574) expr_1_in_3: (1, 1, 2574) expr_1_out_0: (1, 1, 2574)
    // expr_1_out_1: (1, 1, 2574)
    // Iteration reduced to spaces ((2,),)
    // Fixed spaces ()
    // Parameteric spaces ((2,),)
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_1_in_0: int8 0.004 Q7 expr_1_in_1: int8 258.016 Q7
        // expr_1_in_2: int8 249.953 Q7
        // inter__Add_3 = Add(Norm(Mul(Mul(Cast(expr_1_in_0, int32), Cast(expr_1_in_1, int32)), [137]), [22]), Cast(expr_1_in_2, int32))
        int inter__Add_3 = (gap_roundnorm_reg(((((int)expr_1_in_0[i0])*((int)expr_1_in_1[i0]))*(137)), (22))+((int)expr_1_in_2[i0]));
        // inputs expr_1_in_4: int8 0.004 Q7 expr_1_in_3: int8 258.016 Q7
        // inter__Mul_6 = Mul([1], Norm(Mul(Norm(Mul(Cast(Exp1715(Norm(Mul(GapMin([131072], GapMax([-31719424], Cast(expr_1_in_4, int32))), [132]), [7])), int32), [256]), [8]), Cast(expr_1_in_3, int32)), [7]))
        // Min1 scale arg 0 to 1 - int32 4.135 Q0 -> int32 0.004 Q7
        int _SYMBOL_Constant2197 = (131072);
        // Max1 scale arg 0 to 1 - int32 -1000.000 Q0 -> int32 0.004 Q7
        int _SYMBOL_Constant2195 = (-31719424);
        int inter__Mul_6 = ((1)*gap_roundnorm_reg((gap_roundnorm_reg((((int)exp_17_15(gap_roundnorm_reg((gap_min(_SYMBOL_Constant2197, gap_max(_SYMBOL_Constant2195, ((int)expr_1_in_4[i0])))*(132)), (7))))*(256)), (8))*((int)expr_1_in_3[i0])), (7)));
        // inputs inter__Add_3: int32 249.953 Q7 inter__Mul_6: int32 129.256
        // Q15
        // expr_1_out_0 = Cast(Clip(Norm(Mul(Sub(inter__Add_3, Norm(Mul(inter__Mul_6, [132]), [16])), [136]), [7]), -128, 127), int8)
        expr_1_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((inter__Add_3-gap_roundnorm_reg((inter__Mul_6*(132)), (16)))*(136)), (7))), ((7))));
        // inputs inter__Add_3: int32 249.953 Q7 inter__Mul_6: int32 129.256
        // Q15
        // expr_1_out_1 = Cast(Clip(Norm(Mul(Add(inter__Add_3, Norm(Mul(inter__Mul_6, [132]), [16])), [190]), [8]), -128, 127), int8)
        expr_1_out_1[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((inter__Add_3+gap_roundnorm_reg((inter__Mul_6*(132)), (16)))*(190)), (8))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s144_kernel(s144_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_11_in_0 = Args->expr_11_in_0; // (1, 1, 480)   int8 1.008 Q7
    signed char *__restrict__  expr_11_in_1 = Args->expr_11_in_1; // (16, 20, 480) int8 5.888 Q7
    signed char *__restrict__  expr_11_out_0 = Args->expr_11_out_0; // (16, 20, 480) int8 0.488 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (16, 20, 480) var shapes:
    // expr_11_out_0: (16, 20, 480) expr_11_in_0: (1, 1, 480) expr_11_in_1:
    // (16, 20, 480)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_11_in_0: int8 1.008 Q7 expr_11_in_1: int8 5.888 Q7
            // expr_11_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_11_in_0, int32), Cast(expr_11_in_1, int32)), [195]), [11]), -128, 127), int8)
            expr_11_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_11_in_0[i1])*((int)expr_11_in_1[(i0*I1)+i1]))*(195)), (11))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s164_kernel(s164_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_14_in_0 = Args->expr_14_in_0; // (1, 1, 672)   int8 1.008 Q7
    signed char *__restrict__  expr_14_in_1 = Args->expr_14_in_1; // (16, 20, 672) int8 5.241 Q7
    signed char *__restrict__  expr_14_out_0 = Args->expr_14_out_0; // (16, 20, 672) int8 0.470 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (16, 20, 672) var shapes:
    // expr_14_out_0: (16, 20, 672) expr_14_in_0: (1, 1, 672) expr_14_in_1:
    // (16, 20, 672)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_14_in_0: int8 1.008 Q7 expr_14_in_1: int8 5.241 Q7
            // expr_14_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_14_in_0, int32), Cast(expr_14_in_1, int32)), [180]), [11]), -128, 127), int8)
            expr_14_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_14_in_0[i1])*((int)expr_14_in_1[(i0*I1)+i1]))*(180)), (11))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s45_kernel(s45_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_16_in_0 = Args->expr_16_in_0; // (32, 40, 72) int8 5.729 Q7
    signed char *__restrict__  expr_16_in_1 = Args->expr_16_in_1; // (1, 1, 72)   int8 1.008 Q7
    signed char *__restrict__  expr_16_out_0 = Args->expr_16_out_0; // (32, 40, 72) int8 1.503 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (32, 40, 72) var shapes:
    // expr_16_out_0: (32, 40, 72) expr_16_in_1: (1, 1, 72) expr_16_in_0: (32,
    // 40, 72)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_16_in_1: int8 1.008 Q7 expr_16_in_0: int8 5.729 Q7
            // expr_16_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_16_in_1, int32), Cast(expr_16_in_0, int32)), [246]), [13]), -128, 127), int8)
            expr_16_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_16_in_1[i1])*((int)expr_16_in_0[(i0*I1)+i1]))*(246)), (13))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s64_kernel(s64_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_17_in_0 = Args->expr_17_in_0; // (32, 40, 120) int8 4.801 Q7
    signed char *__restrict__  expr_17_in_1 = Args->expr_17_in_1; // (1, 1, 120)   int8 1.008 Q7
    signed char *__restrict__  expr_17_out_0 = Args->expr_17_out_0; // (32, 40, 120) int8 1.469 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (32, 40, 120) var shapes:
    // expr_17_out_0: (32, 40, 120) expr_17_in_1: (1, 1, 120) expr_17_in_0:
    // (32, 40, 120)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_17_in_1: int8 1.008 Q7 expr_17_in_0: int8 4.801 Q7
            // expr_17_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_17_in_1, int32), Cast(expr_17_in_0, int32)), [211]), [13]), -128, 127), int8)
            expr_17_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_17_in_1[i1])*((int)expr_17_in_0[(i0*I1)+i1]))*(211)), (13))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s84_kernel(s84_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_18_in_0 = Args->expr_18_in_0; // (32, 40, 120) int8 5.256 Q7
    signed char *__restrict__  expr_18_in_1 = Args->expr_18_in_1; // (1, 1, 120)   int8 1.008 Q7
    signed char *__restrict__  expr_18_out_0 = Args->expr_18_out_0; // (32, 40, 120) int8 1.155 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (32, 40, 120) var shapes:
    // expr_18_out_0: (32, 40, 120) expr_18_in_1: (1, 1, 120) expr_18_in_0:
    // (32, 40, 120)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_18_in_1: int8 1.008 Q7 expr_18_in_0: int8 5.256 Q7
            // expr_18_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_18_in_1, int32), Cast(expr_18_in_0, int32)), [147]), [12]), -128, 127), int8)
            expr_18_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_18_in_1[i1])*((int)expr_18_in_0[(i0*I1)+i1]))*(147)), (12))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s184_kernel(s184_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_26_in_0 = Args->expr_26_in_0; // (1, 1, 672)  int8 1.008 Q7
    signed char *__restrict__  expr_26_in_1 = Args->expr_26_in_1; // (8, 10, 672) int8 3.806 Q7
    signed char *__restrict__  expr_26_out_0 = Args->expr_26_out_0; // (8, 10, 672) int8 0.516 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (8, 10, 672) var shapes:
    // expr_26_out_0: (8, 10, 672) expr_26_in_0: (1, 1, 672) expr_26_in_1: (8,
    // 10, 672)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_26_in_0: int8 1.008 Q7 expr_26_in_1: int8 3.806 Q7
            // expr_26_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_26_in_0, int32), Cast(expr_26_in_1, int32)), [238]), [12]), -128, 127), int8)
            expr_26_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_26_in_0[i1])*((int)expr_26_in_1[(i0*I1)+i1]))*(238)), (12))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s204_kernel(s204_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_29_in_0 = Args->expr_29_in_0; // (1, 1, 480)  int8 1.008 Q7
    signed char *__restrict__  expr_29_in_1 = Args->expr_29_in_1; // (8, 10, 480) int8 1.215 Q7
    signed char *__restrict__  expr_29_out_0 = Args->expr_29_out_0; // (8, 10, 480) int8 0.659 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (8, 10, 480) var shapes:
    // expr_29_out_0: (8, 10, 480) expr_29_in_0: (1, 1, 480) expr_29_in_1: (8,
    // 10, 480)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_29_in_0: int8 1.008 Q7 expr_29_in_1: int8 1.215 Q7
            // expr_29_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_29_in_0, int32), Cast(expr_29_in_1, int32)), [238]), [14]), -128, 127), int8)
            expr_29_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_29_in_0[i1])*((int)expr_29_in_1[(i0*I1)+i1]))*(238)), (14))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s224_kernel(s224_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_32_in_0 = Args->expr_32_in_0; // (1, 1, 480)  int8 1.008 Q7
    signed char *__restrict__  expr_32_in_1 = Args->expr_32_in_1; // (8, 10, 480) int8 0.951 Q7
    signed char *__restrict__  expr_32_out_0 = Args->expr_32_out_0; // (8, 10, 480) int8 0.473 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I1);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I1);
    // Max shape: (8, 10, 480) var shapes:
    // expr_32_out_0: (8, 10, 480) expr_32_in_0: (1, 1, 480) expr_32_in_1: (8,
    // 10, 480)
    // Iteration reduced to spaces ((0, 1), (2,))
    // Fixed spaces ()
    // Parameteric spaces ((0, 1), (2,))
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=0; i0<I0; i0++) {
        for (int i1=First; i1<Last; i1++) {
            // inputs expr_32_in_0: int8 1.008 Q7 expr_32_in_1: int8 0.951 Q7
            // expr_32_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_32_in_0, int32), Cast(expr_32_in_1, int32)), [130]), [13]), -128, 127), int8)
            expr_32_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_32_in_0[i1])*((int)expr_32_in_1[(i0*I1)+i1]))*(130)), (13))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s336_kernel(s336_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0; // (2574, 4) int8 0.039 Q7
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (2574, 4) int8 0.004 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (2574, 4) var shapes:
    // expr_2_out_0: (2574, 4) expr_2_in_0: (2574, 4)
    // Iteration reduced to spaces ((0, 1),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1),)
    // Paralelized space (0, 1)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: int8 0.039 Q7
        // expr_2_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [-5]), [1]), [246]), [8]), -128, 127), int8)
        expr_2_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(-5))*(1))*(246)), (8))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_6(expr_6_args_t *Args) {
    signed char *__restrict__  expr_6_in_0 = Args->expr_6_in_0;
    signed char *__restrict__  expr_6_out_0 = Args->expr_6_out_0; // (128, 160, 16) int8 9.453 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (128, 160, 16) var shapes:
    // expr_6_out_0: (128, 160, 16) expr_6_in_0: (128, 160, 16)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_6_in_0: int8 9.453 Q7
        // expr_6_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_6_in_0, int32), [-38]), Mul(GapMin([81], GapMax([0], Add(Sub(Cast(expr_6_in_0, int32), [-38]), [41]))), [1])), [156]), [14]), -128, 127), int8)
        // Min2 scale arg 0 to 1 - int32 6.000 Q0 -> int32 9.453 Q7
        int _SYMBOL_Constant2095 = (81);
        // Max2 scale arg 0 to 1 - int32 1.000 Q15 -> int32 9.453 Q7
        int _SYMBOL_QuantizedConstant14 = (0);
        // Add0 scale arg 1 to 0 - int32 3.000 Q0 -> int32 9.453 Q7
        int _SYMBOL_Constant2091 = (41);
        expr_6_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_6_in_0[i0])-(-38))*(gap_min(_SYMBOL_Constant2095, gap_max(_SYMBOL_QuantizedConstant14, ((((int)expr_6_in_0[i0])-(-38))+_SYMBOL_Constant2091)))*(1)))*(156)), (14))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_19(expr_19_args_t *Args) {
    signed char *__restrict__  expr_19_in_0 = Args->expr_19_in_0;
    signed char *__restrict__  expr_19_out_0 = Args->expr_19_out_0; // (32, 40, 240) int8 6.719 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (32, 40, 240) var shapes:
    // expr_19_out_0: (32, 40, 240) expr_19_in_0: (32, 40, 240)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_19_in_0: int8 6.719 Q7
        // expr_19_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_19_in_0, int32), [16]), Mul(GapMin([115], GapMax([0], Add(Sub(Cast(expr_19_in_0, int32), [16]), [57]))), [1])), [164]), [14]), -128, 127), int8)
        // Min10 scale arg 0 to 1 - int32 6.000 Q0 -> int32 6.719 Q7
        int _SYMBOL_Constant2101 = (115);
        // Max10 scale arg 0 to 1 - int32 1.000 Q15 -> int32 6.719 Q7
        int _SYMBOL_QuantizedConstant50 = (0);
        // Add8 scale arg 1 to 0 - int32 3.000 Q0 -> int32 6.719 Q7
        int _SYMBOL_Constant2097 = (57);
        expr_19_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_19_in_0[i0])-(16))*(gap_min(_SYMBOL_Constant2101, gap_max(_SYMBOL_QuantizedConstant50, ((((int)expr_19_in_0[i0])-(16))+_SYMBOL_Constant2097)))*(1)))*(164)), (14))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_20(expr_20_args_t *Args) {
    signed char *__restrict__  expr_20_in_0 = Args->expr_20_in_0;
    signed char *__restrict__  expr_20_out_0 = Args->expr_20_out_0; // (16, 20, 240) int8 4.160 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 240) var shapes:
    // expr_20_out_0: (16, 20, 240) expr_20_in_0: (16, 20, 240)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_20_in_0: int8 4.160 Q7
        // expr_20_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_20_in_0, int32), [-24]), Mul(GapMin([185], GapMax([0], Add(Sub(Cast(expr_20_in_0, int32), [-24]), [93]))), [1])), [150]), [15]), -128, 127), int8)
        // Min11 scale arg 0 to 1 - int32 6.000 Q0 -> int32 4.160 Q7
        int _SYMBOL_Constant2106 = (185);
        // Max11 scale arg 0 to 1 - int32 1.000 Q15 -> int32 4.160 Q7
        int _SYMBOL_QuantizedConstant67 = (0);
        // Add9 scale arg 1 to 0 - int32 3.000 Q0 -> int32 4.160 Q7
        int _SYMBOL_Constant2103 = (93);
        expr_20_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_20_in_0[i0])-(-24))*(gap_min(_SYMBOL_Constant2106, gap_max(_SYMBOL_QuantizedConstant67, ((((int)expr_20_in_0[i0])-(-24))+_SYMBOL_Constant2103)))*(1)))*(150)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_21(expr_21_args_t *Args) {
    signed char *__restrict__  expr_21_in_0 = Args->expr_21_in_0;
    signed char *__restrict__  expr_21_out_0 = Args->expr_21_out_0; // (16, 20, 200) int8 5.411 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 200) var shapes:
    // expr_21_out_0: (16, 20, 200) expr_21_in_0: (16, 20, 200)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_21_in_0: int8 5.411 Q7
        // expr_21_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_21_in_0, int32), [-47]), Mul(GapMin([142], GapMax([0], Add(Sub(Cast(expr_21_in_0, int32), [-47]), [71]))), [1])), [168]), [15]), -128, 127), int8)
        // Min12 scale arg 0 to 1 - int32 6.000 Q0 -> int32 5.411 Q7
        int _SYMBOL_Constant2111 = (142);
        // Max12 scale arg 0 to 1 - int32 1.000 Q15 -> int32 5.411 Q7
        int _SYMBOL_QuantizedConstant84 = (0);
        // Add10 scale arg 1 to 0 - int32 3.000 Q0 -> int32 5.411 Q7
        int _SYMBOL_Constant2108 = (71);
        expr_21_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_21_in_0[i0])-(-47))*(gap_min(_SYMBOL_Constant2111, gap_max(_SYMBOL_QuantizedConstant84, ((((int)expr_21_in_0[i0])-(-47))+_SYMBOL_Constant2108)))*(1)))*(168)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_22(expr_22_args_t *Args) {
    signed char *__restrict__  expr_22_in_0 = Args->expr_22_in_0;
    signed char *__restrict__  expr_22_out_0 = Args->expr_22_out_0; // (16, 20, 200) int8 3.335 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 200) var shapes:
    // expr_22_out_0: (16, 20, 200) expr_22_in_0: (16, 20, 200)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_22_in_0: int8 3.335 Q7
        // expr_22_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_22_in_0, int32), [26]), Mul(GapMin([230], GapMax([0], Add(Sub(Cast(expr_22_in_0, int32), [26]), [115]))), [1])), [190]), [15]), -128, 127), int8)
        // Min13 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.335 Q7
        int _SYMBOL_Constant2116 = (230);
        // Max13 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.335 Q7
        int _SYMBOL_QuantizedConstant101 = (0);
        // Add11 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.335 Q7
        int _SYMBOL_Constant2113 = (115);
        expr_22_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_22_in_0[i0])-(26))*(gap_min(_SYMBOL_Constant2116, gap_max(_SYMBOL_QuantizedConstant101, ((((int)expr_22_in_0[i0])-(26))+_SYMBOL_Constant2113)))*(1)))*(190)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_23(expr_23_args_t *Args) {
    signed char *__restrict__  expr_23_in_0 = Args->expr_23_in_0;
    signed char *__restrict__  expr_23_out_0 = Args->expr_23_out_0; // (16, 20, 184) int8 4.046 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 184) var shapes:
    // expr_23_out_0: (16, 20, 184) expr_23_in_0: (16, 20, 184)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_23_in_0: int8 4.046 Q7
        // expr_23_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_23_in_0, int32), [-23]), Mul(GapMin([190], GapMax([0], Add(Sub(Cast(expr_23_in_0, int32), [-23]), [95]))), [1])), [146]), [15]), -128, 127), int8)
        // Min14 scale arg 0 to 1 - int32 6.000 Q0 -> int32 4.046 Q7
        int _SYMBOL_Constant2121 = (190);
        // Max14 scale arg 0 to 1 - int32 1.000 Q15 -> int32 4.046 Q7
        int _SYMBOL_QuantizedConstant118 = (0);
        // Add12 scale arg 1 to 0 - int32 3.000 Q0 -> int32 4.046 Q7
        int _SYMBOL_Constant2118 = (95);
        expr_23_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_23_in_0[i0])-(-23))*(gap_min(_SYMBOL_Constant2121, gap_max(_SYMBOL_QuantizedConstant118, ((((int)expr_23_in_0[i0])-(-23))+_SYMBOL_Constant2118)))*(1)))*(146)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_24(expr_24_args_t *Args) {
    signed char *__restrict__  expr_24_in_0 = Args->expr_24_in_0;
    signed char *__restrict__  expr_24_out_0 = Args->expr_24_out_0; // (16, 20, 184) int8 3.619 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 184) var shapes:
    // expr_24_out_0: (16, 20, 184) expr_24_in_0: (16, 20, 184)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_24_in_0: int8 3.619 Q7
        // expr_24_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_24_in_0, int32), [6]), Mul(GapMin([212], GapMax([0], Add(Sub(Cast(expr_24_in_0, int32), [6]), [106]))), [1])), [162]), [15]), -128, 127), int8)
        // Min15 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.619 Q7
        int _SYMBOL_Constant2126 = (212);
        // Max15 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.619 Q7
        int _SYMBOL_QuantizedConstant135 = (0);
        // Add13 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.619 Q7
        int _SYMBOL_Constant2123 = (106);
        expr_24_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_24_in_0[i0])-(6))*(gap_min(_SYMBOL_Constant2126, gap_max(_SYMBOL_QuantizedConstant135, ((((int)expr_24_in_0[i0])-(6))+_SYMBOL_Constant2123)))*(1)))*(162)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_7(expr_7_args_t *Args) {
    signed char *__restrict__  expr_7_in_0 = Args->expr_7_in_0;
    signed char *__restrict__  expr_7_out_0 = Args->expr_7_out_0; // (16, 20, 184) int8 3.474 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 184) var shapes:
    // expr_7_out_0: (16, 20, 184) expr_7_in_0: (16, 20, 184)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_7_in_0: int8 3.474 Q7
        // expr_7_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_7_in_0, int32), [-16]), Mul(GapMin([221], GapMax([0], Add(Sub(Cast(expr_7_in_0, int32), [-16]), [111]))), [1])), [132]), [15]), -128, 127), int8)
        // Min3 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.474 Q7
        int _SYMBOL_Constant2131 = (221);
        // Max3 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.474 Q7
        int _SYMBOL_QuantizedConstant152 = (0);
        // Add1 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.474 Q7
        int _SYMBOL_Constant2128 = (111);
        expr_7_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_7_in_0[i0])-(-16))*(gap_min(_SYMBOL_Constant2131, gap_max(_SYMBOL_QuantizedConstant152, ((((int)expr_7_in_0[i0])-(-16))+_SYMBOL_Constant2128)))*(1)))*(132)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_8(expr_8_args_t *Args) {
    signed char *__restrict__  expr_8_in_0 = Args->expr_8_in_0;
    signed char *__restrict__  expr_8_out_0 = Args->expr_8_out_0; // (16, 20, 184) int8 2.863 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 184) var shapes:
    // expr_8_out_0: (16, 20, 184) expr_8_in_0: (16, 20, 184)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_8_in_0: int8 2.863 Q7
        // expr_8_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_8_in_0, int32), [-8]), Mul(GapMin([268], GapMax([0], Add(Sub(Cast(expr_8_in_0, int32), [-8]), [134]))), [1])), [231]), [16]), -128, 127), int8)
        // Min4 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.863 Q7
        int _SYMBOL_Constant2136 = (268);
        // Max4 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.863 Q7
        int _SYMBOL_QuantizedConstant169 = (0);
        // Add2 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.863 Q7
        int _SYMBOL_Constant2132 = (134);
        expr_8_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_8_in_0[i0])-(-8))*(gap_min(_SYMBOL_Constant2136, gap_max(_SYMBOL_QuantizedConstant169, ((((int)expr_8_in_0[i0])-(-8))+_SYMBOL_Constant2132)))*(1)))*(231)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_9(expr_9_args_t *Args) {
    signed char *__restrict__  expr_9_in_0 = Args->expr_9_in_0;
    signed char *__restrict__  expr_9_out_0 = Args->expr_9_out_0; // (16, 20, 480) int8 4.203 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 480) var shapes:
    // expr_9_out_0: (16, 20, 480) expr_9_in_0: (16, 20, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_9_in_0: int8 4.203 Q7
        // expr_9_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_9_in_0, int32), [-8]), Mul(GapMin([183], GapMax([0], Add(Sub(Cast(expr_9_in_0, int32), [-8]), [92]))), [1])), [169]), [15]), -128, 127), int8)
        // Min5 scale arg 0 to 1 - int32 6.000 Q0 -> int32 4.203 Q7
        int _SYMBOL_Constant2141 = (183);
        // Max5 scale arg 0 to 1 - int32 1.000 Q15 -> int32 4.203 Q7
        int _SYMBOL_QuantizedConstant186 = (0);
        // Add3 scale arg 1 to 0 - int32 3.000 Q0 -> int32 4.203 Q7
        int _SYMBOL_Constant2138 = (92);
        expr_9_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_9_in_0[i0])-(-8))*(gap_min(_SYMBOL_Constant2141, gap_max(_SYMBOL_QuantizedConstant186, ((((int)expr_9_in_0[i0])-(-8))+_SYMBOL_Constant2138)))*(1)))*(169)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_10(expr_10_args_t *Args) {
    signed char *__restrict__  expr_10_in_0 = Args->expr_10_in_0;
    signed char *__restrict__  expr_10_out_0 = Args->expr_10_out_0; // (16, 20, 480) int8 6.045 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 480) var shapes:
    // expr_10_out_0: (16, 20, 480) expr_10_in_0: (16, 20, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_10_in_0: int8 6.045 Q7
        // expr_10_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_10_in_0, int32), [3]), Mul(GapMin([127], GapMax([0], Add(Sub(Cast(expr_10_in_0, int32), [3]), [64]))), [1])), [132]), [14]), -128, 127), int8)
        // Min6 scale arg 0 to 1 - int32 6.000 Q0 -> int32 6.045 Q7
        int _SYMBOL_Constant2147 = (127);
        // Max6 scale arg 0 to 1 - int32 1.000 Q15 -> int32 6.045 Q7
        int _SYMBOL_QuantizedConstant204 = (0);
        // Add4 scale arg 1 to 0 - int32 3.000 Q0 -> int32 6.045 Q7
        int _SYMBOL_Constant2143 = (64);
        expr_10_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_10_in_0[i0])-(3))*(gap_min(_SYMBOL_Constant2147, gap_max(_SYMBOL_QuantizedConstant204, ((((int)expr_10_in_0[i0])-(3))+_SYMBOL_Constant2143)))*(1)))*(132)), (14))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_12(expr_12_args_t *Args) {
    signed char *__restrict__  expr_12_in_0 = Args->expr_12_in_0;
    signed char *__restrict__  expr_12_out_0 = Args->expr_12_out_0; // (16, 20, 672) int8 2.977 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 672) var shapes:
    // expr_12_out_0: (16, 20, 672) expr_12_in_0: (16, 20, 672)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_12_in_0: int8 2.977 Q7
        // expr_12_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_12_in_0, int32), Mul(GapMin([258], GapMax([0], Add(Cast(expr_12_in_0, int32), [129]))), [1])), [255]), [16]), -128, 127), int8)
        // Min7 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.977 Q7
        int _SYMBOL_Constant2152 = (258);
        // Max7 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.977 Q7
        int _SYMBOL_QuantizedConstant224 = (0);
        // Add5 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.977 Q7
        int _SYMBOL_Constant2148 = (129);
        expr_12_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_12_in_0[i0])*(gap_min(_SYMBOL_Constant2152, gap_max(_SYMBOL_QuantizedConstant224, (((int)expr_12_in_0[i0])+_SYMBOL_Constant2148)))*(1)))*(255)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_13(expr_13_args_t *Args) {
    signed char *__restrict__  expr_13_in_0 = Args->expr_13_in_0;
    signed char *__restrict__  expr_13_out_0 = Args->expr_13_out_0; // (16, 20, 672) int8 4.732 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 672) var shapes:
    // expr_13_out_0: (16, 20, 672) expr_13_in_0: (16, 20, 672)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_13_in_0: int8 4.732 Q7
        // expr_13_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_13_in_0, int32), [-14]), Mul(GapMin([162], GapMax([0], Add(Sub(Cast(expr_13_in_0, int32), [-14]), [81]))), [1])), [182]), [15]), -128, 127), int8)
        // Min8 scale arg 0 to 1 - int32 6.000 Q0 -> int32 4.732 Q7
        int _SYMBOL_Constant2157 = (162);
        // Max8 scale arg 0 to 1 - int32 1.000 Q15 -> int32 4.732 Q7
        int _SYMBOL_QuantizedConstant241 = (0);
        // Add6 scale arg 1 to 0 - int32 3.000 Q0 -> int32 4.732 Q7
        int _SYMBOL_Constant2154 = (81);
        expr_13_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_13_in_0[i0])-(-14))*(gap_min(_SYMBOL_Constant2157, gap_max(_SYMBOL_QuantizedConstant241, ((((int)expr_13_in_0[i0])-(-14))+_SYMBOL_Constant2154)))*(1)))*(182)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_15(expr_15_args_t *Args) {
    signed char *__restrict__  expr_15_in_0 = Args->expr_15_in_0;
    signed char *__restrict__  expr_15_out_0 = Args->expr_15_out_0; // (16, 20, 672) int8 3.153 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 672) var shapes:
    // expr_15_out_0: (16, 20, 672) expr_15_in_0: (16, 20, 672)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_15_in_0: int8 3.153 Q7
        // expr_15_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_15_in_0, int32), [9]), Mul(GapMin([244], GapMax([0], Add(Sub(Cast(expr_15_in_0, int32), [9]), [122]))), [1])), [147]), [15]), -128, 127), int8)
        // Min9 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.153 Q7
        int _SYMBOL_Constant2162 = (244);
        // Max9 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.153 Q7
        int _SYMBOL_QuantizedConstant263 = (0);
        // Add7 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.153 Q7
        int _SYMBOL_Constant2159 = (122);
        expr_15_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_15_in_0[i0])-(9))*(gap_min(_SYMBOL_Constant2162, gap_max(_SYMBOL_QuantizedConstant263, ((((int)expr_15_in_0[i0])-(9))+_SYMBOL_Constant2159)))*(1)))*(147)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_25(expr_25_args_t *Args) {
    signed char *__restrict__  expr_25_in_0 = Args->expr_25_in_0;
    signed char *__restrict__  expr_25_out_0 = Args->expr_25_out_0; // (8, 10, 672) int8 3.197 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 672) var shapes:
    // expr_25_out_0: (8, 10, 672) expr_25_in_0: (8, 10, 672)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_25_in_0: int8 3.197 Q7
        // expr_25_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_25_in_0, int32), [-24]), Mul(GapMin([240], GapMax([0], Add(Sub(Cast(expr_25_in_0, int32), [-24]), [120]))), [1])), [229]), [16]), -128, 127), int8)
        // Min16 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.197 Q7
        int _SYMBOL_Constant2167 = (240);
        // Max16 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.197 Q7
        int _SYMBOL_QuantizedConstant280 = (0);
        // Add14 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.197 Q7
        int _SYMBOL_Constant2164 = (120);
        expr_25_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_25_in_0[i0])-(-24))*(gap_min(_SYMBOL_Constant2167, gap_max(_SYMBOL_QuantizedConstant280, ((((int)expr_25_in_0[i0])-(-24))+_SYMBOL_Constant2164)))*(1)))*(229)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_27(expr_27_args_t *Args) {
    signed char *__restrict__  expr_27_in_0 = Args->expr_27_in_0;
    signed char *__restrict__  expr_27_out_0 = Args->expr_27_out_0; // (8, 10, 480) int8 1.759 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_27_out_0: (8, 10, 480) expr_27_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_27_in_0: int8 1.759 Q7
        // expr_27_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_27_in_0, int32), [12]), Mul(GapMin([436], GapMax([0], Add(Sub(Cast(expr_27_in_0, int32), [12]), [218]))), [1])), [218]), [16]), -128, 127), int8)
        // Min17 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.759 Q7
        int _SYMBOL_Constant2172 = (436);
        // Max17 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.759 Q7
        int _SYMBOL_QuantizedConstant302 = (0);
        // Add15 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.759 Q7
        int _SYMBOL_Constant2168 = (218);
        expr_27_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_27_in_0[i0])-(12))*(gap_min(_SYMBOL_Constant2172, gap_max(_SYMBOL_QuantizedConstant302, ((((int)expr_27_in_0[i0])-(12))+_SYMBOL_Constant2168)))*(1)))*(218)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_28(expr_28_args_t *Args) {
    signed char *__restrict__  expr_28_in_0 = Args->expr_28_in_0;
    signed char *__restrict__  expr_28_out_0 = Args->expr_28_out_0; // (8, 10, 480) int8 1.994 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_28_out_0: (8, 10, 480) expr_28_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_28_in_0: int8 1.994 Q7
        // expr_28_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_28_in_0, int32), [26]), Mul(GapMin([386], GapMax([0], Add(Sub(Cast(expr_28_in_0, int32), [26]), [193]))), [1])), [140]), [15]), -128, 127), int8)
        // Min18 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.994 Q7
        int _SYMBOL_Constant2177 = (386);
        // Max18 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.994 Q7
        int _SYMBOL_QuantizedConstant319 = (0);
        // Add16 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.994 Q7
        int _SYMBOL_Constant2173 = (193);
        expr_28_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_28_in_0[i0])-(26))*(gap_min(_SYMBOL_Constant2177, gap_max(_SYMBOL_QuantizedConstant319, ((((int)expr_28_in_0[i0])-(26))+_SYMBOL_Constant2173)))*(1)))*(140)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_30(expr_30_args_t *Args) {
    signed char *__restrict__  expr_30_in_0 = Args->expr_30_in_0;
    signed char *__restrict__  expr_30_out_0 = Args->expr_30_out_0; // (8, 10, 480) int8 1.590 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_30_out_0: (8, 10, 480) expr_30_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_30_in_0: int8 1.590 Q7
        // expr_30_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_30_in_0, int32), [8]), Mul(GapMin([482], GapMax([0], Add(Sub(Cast(expr_30_in_0, int32), [8]), [241]))), [1])), [193]), [16]), -128, 127), int8)
        // Min19 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.590 Q7
        int _SYMBOL_Constant2182 = (482);
        // Max19 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.590 Q7
        int _SYMBOL_QuantizedConstant341 = (0);
        // Add17 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.590 Q7
        int _SYMBOL_Constant2178 = (241);
        expr_30_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_30_in_0[i0])-(8))*(gap_min(_SYMBOL_Constant2182, gap_max(_SYMBOL_QuantizedConstant341, ((((int)expr_30_in_0[i0])-(8))+_SYMBOL_Constant2178)))*(1)))*(193)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_31(expr_31_args_t *Args) {
    signed char *__restrict__  expr_31_in_0 = Args->expr_31_in_0;
    signed char *__restrict__  expr_31_out_0 = Args->expr_31_out_0; // (8, 10, 480) int8 1.741 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_31_out_0: (8, 10, 480) expr_31_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_31_in_0: int8 1.741 Q7
        // expr_31_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_31_in_0, int32), [31]), Mul(GapMin([442], GapMax([0], Add(Sub(Cast(expr_31_in_0, int32), [31]), [221]))), [1])), [136]), [15]), -128, 127), int8)
        // Min20 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.741 Q7
        int _SYMBOL_Constant2187 = (442);
        // Max20 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.741 Q7
        int _SYMBOL_QuantizedConstant358 = (0);
        // Add18 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.741 Q7
        int _SYMBOL_Constant2183 = (221);
        expr_31_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_31_in_0[i0])-(31))*(gap_min(_SYMBOL_Constant2187, gap_max(_SYMBOL_QuantizedConstant358, ((((int)expr_31_in_0[i0])-(31))+_SYMBOL_Constant2183)))*(1)))*(136)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_33(expr_33_args_t *Args) {
    signed char *__restrict__  expr_33_in_0 = Args->expr_33_in_0;
    signed char *__restrict__  expr_33_out_0 = Args->expr_33_out_0; // (8, 10, 480) int8 2.122 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 480) var shapes:
    // expr_33_out_0: (8, 10, 480) expr_33_in_0: (8, 10, 480)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_33_in_0: int8 2.122 Q7
        // expr_33_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_33_in_0, int32), [-18]), Mul(GapMin([362], GapMax([0], Add(Sub(Cast(expr_33_in_0, int32), [-18]), [181]))), [1])), [175]), [16]), -128, 127), int8)
        // Min21 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.122 Q7
        int _SYMBOL_Constant2192 = (362);
        // Max21 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.122 Q7
        int _SYMBOL_QuantizedConstant380 = (0);
        // Add19 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.122 Q7
        int _SYMBOL_Constant2188 = (181);
        expr_33_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_33_in_0[i0])-(-18))*(gap_min(_SYMBOL_Constant2192, gap_max(_SYMBOL_QuantizedConstant380, ((((int)expr_33_in_0[i0])-(-18))+_SYMBOL_Constant2188)))*(1)))*(175)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}


#pragma GCC diagnostic pop