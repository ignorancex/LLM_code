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
void s372_kernel(s372_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_0_in_0 = Args->expr_0_in_0; // (1, 1, 2574) int8 0.003 Q7
    signed char *__restrict__  expr_0_in_1 = Args->expr_0_in_1; // (1, 1, 2574) int8 322.520 Q7
    signed char *__restrict__  expr_0_in_2 = Args->expr_0_in_2; // (1, 1, 2574) int8 314.457 Q7
    signed char *__restrict__  expr_0_in_3 = Args->expr_0_in_3; // (1, 1, 2574) int8 258.016 Q7
    signed char *__restrict__  expr_0_in_4 = Args->expr_0_in_4; // (1, 1, 2574) int8 0.003 Q7
    signed char *__restrict__  expr_0_out_0 = Args->expr_0_out_0; // (1, 1, 2574) int8 299.663 Q7
    signed char *__restrict__  expr_0_out_1 = Args->expr_0_out_1; // (1, 1, 2574) int8 402.061 Q7
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
        // inputs expr_0_in_0: int8 0.003 Q7 expr_0_in_1: int8 322.520 Q7
        // expr_0_in_2: int8 314.457 Q7
        // inter__Add_2 = Add(Norm(Mul(Mul(Cast(expr_0_in_0, int32), Cast(expr_0_in_1, int32)), [234]), [23]), Cast(expr_0_in_2, int32))
        int inter__Add_2 = (gap_roundnorm_reg(((((int)expr_0_in_0[i0])*((int)expr_0_in_1[i0]))*(234)), (23))+((int)expr_0_in_2[i0]));
        // inputs expr_0_in_4: int8 0.003 Q7 expr_0_in_3: int8 258.016 Q7
        // inter__Mul_7 = Mul([1], Norm(Mul(Norm(Mul(Cast(Exp1715(Norm(Mul(GapMin([152576], GapMax([-36700160], Cast(expr_0_in_4, int32))), [228]), [8])), int32), [255]), [8]), Cast(expr_0_in_3, int32)), [7]))
        // Min0 scale arg 0 to 1 - int32 4.135 Q0 -> int32 0.003 Q7
        int _SYMBOL_Constant403 = (152576);
        // Max0 scale arg 0 to 1 - int32 -1000.000 Q0 -> int32 0.003 Q7
        int _SYMBOL_Constant401 = (-36700160);
        int inter__Mul_7 = ((1)*gap_roundnorm_reg((gap_roundnorm_reg((((int)exp_17_15(gap_roundnorm_reg((gap_min(_SYMBOL_Constant403, gap_max(_SYMBOL_Constant401, ((int)expr_0_in_4[i0])))*(228)), (8))))*(255)), (8))*((int)expr_0_in_3[i0])), (7)));
        // inputs inter__Add_2: int32 314.457 Q7 inter__Mul_7: int32 129.290
        // Q15
        // expr_0_out_0 = Cast(Clip(Norm(Mul(Sub(inter__Add_2, Norm(Mul(inter__Mul_7, [211]), [17])), [134]), [7]), -128, 127), int8)
        expr_0_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((inter__Add_2-gap_roundnorm_reg((inter__Mul_7*(211)), (17)))*(134)), (7))), ((7))));
        // inputs inter__Add_2: int32 314.457 Q7 inter__Mul_7: int32 129.290
        // Q15
        // expr_0_out_1 = Cast(Clip(Norm(Mul(Add(inter__Add_2, Norm(Mul(inter__Mul_7, [211]), [17])), [200]), [8]), -128, 127), int8)
        expr_0_out_1[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((inter__Add_2+gap_roundnorm_reg((inter__Mul_7*(211)), (17)))*(200)), (8))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void s366_kernel(s366_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    signed char *__restrict__  expr_1_in_0 = Args->expr_1_in_0; // (1, 1, 2574) int8 0.003 Q7
    signed char *__restrict__  expr_1_in_1 = Args->expr_1_in_1; // (1, 1, 2574) int8 258.016 Q7
    signed char *__restrict__  expr_1_in_2 = Args->expr_1_in_2; // (1, 1, 2574) int8 249.953 Q7
    signed char *__restrict__  expr_1_in_3 = Args->expr_1_in_3; // (1, 1, 2574) int8 258.016 Q7
    signed char *__restrict__  expr_1_in_4 = Args->expr_1_in_4; // (1, 1, 2574) int8 0.003 Q7
    signed char *__restrict__  expr_1_out_0 = Args->expr_1_out_0; // (1, 1, 2574) int8 235.077 Q7
    signed char *__restrict__  expr_1_out_1 = Args->expr_1_out_1; // (1, 1, 2574) int8 337.518 Q7
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
        // inputs expr_1_in_0: int8 0.003 Q7 expr_1_in_1: int8 258.016 Q7
        // expr_1_in_2: int8 249.953 Q7
        // inter__Add_3 = Add(Norm(Mul(Mul(Cast(expr_1_in_0, int32), Cast(expr_1_in_1, int32)), [235]), [23]), Cast(expr_1_in_2, int32))
        int inter__Add_3 = (gap_roundnorm_reg(((((int)expr_1_in_0[i0])*((int)expr_1_in_1[i0]))*(235)), (23))+((int)expr_1_in_2[i0]));
        // inputs expr_1_in_4: int8 0.003 Q7 expr_1_in_3: int8 258.016 Q7
        // inter__Mul_6 = Mul([1], Norm(Mul(Norm(Mul(Cast(Exp1715(Norm(Mul(GapMin([152576], GapMax([-36700160], Cast(expr_1_in_4, int32))), [228]), [8])), int32), [256]), [8]), Cast(expr_1_in_3, int32)), [7]))
        // Min1 scale arg 0 to 1 - int32 4.135 Q0 -> int32 0.003 Q7
        int _SYMBOL_Constant399 = (152576);
        // Max1 scale arg 0 to 1 - int32 -1000.000 Q0 -> int32 0.003 Q7
        int _SYMBOL_Constant397 = (-36700160);
        int inter__Mul_6 = ((1)*gap_roundnorm_reg((gap_roundnorm_reg((((int)exp_17_15(gap_roundnorm_reg((gap_min(_SYMBOL_Constant399, gap_max(_SYMBOL_Constant397, ((int)expr_1_in_4[i0])))*(228)), (8))))*(256)), (8))*((int)expr_1_in_3[i0])), (7)));
        // inputs inter__Add_3: int32 249.953 Q7 inter__Mul_6: int32 129.197
        // Q15
        // expr_1_out_0 = Cast(Clip(Norm(Mul(Sub(inter__Add_3, Norm(Mul(inter__Mul_6, [132]), [16])), [136]), [7]), -128, 127), int8)
        expr_1_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg(((inter__Add_3-gap_roundnorm_reg((inter__Mul_6*(132)), (16)))*(136)), (7))), ((7))));
        // inputs inter__Add_3: int32 249.953 Q7 inter__Mul_6: int32 129.197
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
    signed char *__restrict__  expr_11_in_1 = Args->expr_11_in_1; // (16, 20, 480) int8 3.827 Q7
    signed char *__restrict__  expr_11_out_0 = Args->expr_11_out_0; // (16, 20, 480) int8 0.486 Q7
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
            // inputs expr_11_in_0: int8 1.008 Q7 expr_11_in_1: int8 3.827 Q7
            // expr_11_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_11_in_0, int32), Cast(expr_11_in_1, int32)), [254]), [12]), -128, 127), int8)
            expr_11_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_11_in_0[i1])*((int)expr_11_in_1[(i0*I1)+i1]))*(254)), (12))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s164_kernel(s164_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_14_in_0 = Args->expr_14_in_0; // (1, 1, 672)   int8 1.008 Q7
    signed char *__restrict__  expr_14_in_1 = Args->expr_14_in_1; // (16, 20, 672) int8 3.393 Q7
    signed char *__restrict__  expr_14_out_0 = Args->expr_14_out_0; // (16, 20, 672) int8 0.346 Q7
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
            // inputs expr_14_in_0: int8 1.008 Q7 expr_14_in_1: int8 3.393 Q7
            // expr_14_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_14_in_0, int32), Cast(expr_14_in_1, int32)), [158]), [11]), -128, 127), int8)
            expr_14_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_14_in_0[i1])*((int)expr_14_in_1[(i0*I1)+i1]))*(158)), (11))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s45_kernel(s45_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_16_in_0 = Args->expr_16_in_0; // (32, 40, 72) int8 11.855 Q7
    signed char *__restrict__  expr_16_in_1 = Args->expr_16_in_1; // (1, 1, 72)   int8 1.008 Q7
    signed char *__restrict__  expr_16_out_0 = Args->expr_16_out_0; // (32, 40, 72) int8 1.487 Q7
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
            // inputs expr_16_in_1: int8 1.008 Q7 expr_16_in_0: int8 11.855 Q7
            // expr_16_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_16_in_1, int32), Cast(expr_16_in_0, int32)), [129]), [11]), -128, 127), int8)
            expr_16_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_16_in_1[i1])*((int)expr_16_in_0[(i0*I1)+i1]))*(129)), (11))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s64_kernel(s64_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_17_in_0 = Args->expr_17_in_0; // (32, 40, 120) int8 3.358 Q7
    signed char *__restrict__  expr_17_in_1 = Args->expr_17_in_1; // (1, 1, 120)   int8 1.008 Q7
    signed char *__restrict__  expr_17_out_0 = Args->expr_17_out_0; // (32, 40, 120) int8 1.003 Q7
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
            // inputs expr_17_in_1: int8 1.008 Q7 expr_17_in_0: int8 3.358 Q7
            // expr_17_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_17_in_1, int32), Cast(expr_17_in_0, int32)), [216]), [13]), -128, 127), int8)
            expr_17_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_17_in_1[i1])*((int)expr_17_in_0[(i0*I1)+i1]))*(216)), (13))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s84_kernel(s84_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_18_in_0 = Args->expr_18_in_0; // (32, 40, 120) int8 3.778 Q7
    signed char *__restrict__  expr_18_in_1 = Args->expr_18_in_1; // (1, 1, 120)   int8 1.008 Q7
    signed char *__restrict__  expr_18_out_0 = Args->expr_18_out_0; // (32, 40, 120) int8 0.875 Q7
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
            // inputs expr_18_in_1: int8 1.008 Q7 expr_18_in_0: int8 3.778 Q7
            // expr_18_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_18_in_1, int32), Cast(expr_18_in_0, int32)), [139]), [12]), -128, 127), int8)
            expr_18_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_18_in_1[i1])*((int)expr_18_in_0[(i0*I1)+i1]))*(139)), (12))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s184_kernel(s184_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_26_in_0 = Args->expr_26_in_0; // (1, 1, 672)  int8 1.008 Q7
    signed char *__restrict__  expr_26_in_1 = Args->expr_26_in_1; // (8, 10, 672) int8 2.666 Q7
    signed char *__restrict__  expr_26_out_0 = Args->expr_26_out_0; // (8, 10, 672) int8 0.492 Q7
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
            // inputs expr_26_in_0: int8 1.008 Q7 expr_26_in_1: int8 2.666 Q7
            // expr_26_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_26_in_0, int32), Cast(expr_26_in_1, int32)), [175]), [12]), -128, 127), int8)
            expr_26_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_26_in_0[i1])*((int)expr_26_in_1[(i0*I1)+i1]))*(175)), (12))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s204_kernel(s204_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_29_in_0 = Args->expr_29_in_0; // (1, 1, 480)  int8 1.008 Q7
    signed char *__restrict__  expr_29_in_1 = Args->expr_29_in_1; // (8, 10, 480) int8 0.805 Q7
    signed char *__restrict__  expr_29_out_0 = Args->expr_29_out_0; // (8, 10, 480) int8 0.433 Q7
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
            // inputs expr_29_in_0: int8 1.008 Q7 expr_29_in_1: int8 0.805 Q7
            // expr_29_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_29_in_0, int32), Cast(expr_29_in_1, int32)), [240]), [14]), -128, 127), int8)
            expr_29_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_29_in_0[i1])*((int)expr_29_in_1[(i0*I1)+i1]))*(240)), (14))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 2 external iteration spaces
void s224_kernel(s224_kernel_args_t *Args) {
    unsigned int I0 = Args->I0;
    unsigned int I1 = Args->I1;
    signed char *__restrict__  expr_32_in_0 = Args->expr_32_in_0; // (1, 1, 480)  int8 1.008 Q7
    signed char *__restrict__  expr_32_in_1 = Args->expr_32_in_1; // (8, 10, 480) int8 0.683 Q7
    signed char *__restrict__  expr_32_out_0 = Args->expr_32_out_0; // (8, 10, 480) int8 0.344 Q7
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
            // inputs expr_32_in_0: int8 1.008 Q7 expr_32_in_1: int8 0.683 Q7
            // expr_32_out_0 = Cast(Clip(Norm(Mul(Mul(Cast(expr_32_in_0, int32), Cast(expr_32_in_1, int32)), [256]), [14]), -128, 127), int8)
            expr_32_out_0[(i0*I1)+i1] = ((signed char)gap_clip((gap_roundnorm_reg(((((int)expr_32_in_0[i1])*((int)expr_32_in_1[(i0*I1)+i1]))*(256)), (14))), ((7))));
        }
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_6(expr_6_args_t *Args) {
    signed char *__restrict__  expr_6_in_0 = Args->expr_6_in_0;
    signed char *__restrict__  expr_6_out_0 = Args->expr_6_out_0; // (128, 160, 16) int8 11.217 Q7
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
        // inputs expr_6_in_0: int8 11.217 Q7
        // expr_6_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_6_in_0, int32), [5]), Mul(GapMin([69], GapMax([0], Add(Sub(Cast(expr_6_in_0, int32), [5]), [34]))), [1])), [249]), [14]), -128, 127), int8)
        // Min2 scale arg 0 to 1 - int32 6.000 Q0 -> int32 11.217 Q7
        int _SYMBOL_Constant292 = (69);
        // Max2 scale arg 0 to 1 - int32 1.000 Q15 -> int32 11.217 Q7
        int _SYMBOL_QuantizedConstant14 = (0);
        // Add0 scale arg 1 to 0 - int32 3.000 Q0 -> int32 11.217 Q7
        int _SYMBOL_Constant288 = (34);
        expr_6_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_6_in_0[i0])-(5))*(gap_min(_SYMBOL_Constant292, gap_max(_SYMBOL_QuantizedConstant14, ((((int)expr_6_in_0[i0])-(5))+_SYMBOL_Constant288)))*(1)))*(249)), (14))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_19(expr_19_args_t *Args) {
    signed char *__restrict__  expr_19_in_0 = Args->expr_19_in_0;
    signed char *__restrict__  expr_19_out_0 = Args->expr_19_out_0; // (32, 40, 240) int8 6.316 Q7
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
        // inputs expr_19_in_0: int8 6.316 Q7
        // expr_19_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_19_in_0, int32), [20]), Mul(GapMin([122], GapMax([0], Add(Sub(Cast(expr_19_in_0, int32), [20]), [61]))), [1])), [160]), [14]), -128, 127), int8)
        // Min10 scale arg 0 to 1 - int32 6.000 Q0 -> int32 6.316 Q7
        int _SYMBOL_Constant298 = (122);
        // Max10 scale arg 0 to 1 - int32 1.000 Q15 -> int32 6.316 Q7
        int _SYMBOL_QuantizedConstant50 = (0);
        // Add8 scale arg 1 to 0 - int32 3.000 Q0 -> int32 6.316 Q7
        int _SYMBOL_Constant294 = (61);
        expr_19_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_19_in_0[i0])-(20))*(gap_min(_SYMBOL_Constant298, gap_max(_SYMBOL_QuantizedConstant50, ((((int)expr_19_in_0[i0])-(20))+_SYMBOL_Constant294)))*(1)))*(160)), (14))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_20(expr_20_args_t *Args) {
    signed char *__restrict__  expr_20_in_0 = Args->expr_20_in_0;
    signed char *__restrict__  expr_20_out_0 = Args->expr_20_out_0; // (16, 20, 240) int8 4.152 Q7
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
        // inputs expr_20_in_0: int8 4.152 Q7
        // expr_20_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_20_in_0, int32), [-23]), Mul(GapMin([185], GapMax([0], Add(Sub(Cast(expr_20_in_0, int32), [-23]), [93]))), [1])), [150]), [15]), -128, 127), int8)
        // Min11 scale arg 0 to 1 - int32 6.000 Q0 -> int32 4.152 Q7
        int _SYMBOL_Constant303 = (185);
        // Max11 scale arg 0 to 1 - int32 1.000 Q15 -> int32 4.152 Q7
        int _SYMBOL_QuantizedConstant67 = (0);
        // Add9 scale arg 1 to 0 - int32 3.000 Q0 -> int32 4.152 Q7
        int _SYMBOL_Constant300 = (93);
        expr_20_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_20_in_0[i0])-(-23))*(gap_min(_SYMBOL_Constant303, gap_max(_SYMBOL_QuantizedConstant67, ((((int)expr_20_in_0[i0])-(-23))+_SYMBOL_Constant300)))*(1)))*(150)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_21(expr_21_args_t *Args) {
    signed char *__restrict__  expr_21_in_0 = Args->expr_21_in_0;
    signed char *__restrict__  expr_21_out_0 = Args->expr_21_out_0; // (16, 20, 200) int8 5.028 Q7
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
        // inputs expr_21_in_0: int8 5.028 Q7
        // expr_21_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_21_in_0, int32), [-58]), Mul(GapMin([153], GapMax([0], Add(Sub(Cast(expr_21_in_0, int32), [-58]), [77]))), [1])), [147]), [15]), -128, 127), int8)
        // Min12 scale arg 0 to 1 - int32 6.000 Q0 -> int32 5.028 Q7
        int _SYMBOL_Constant308 = (153);
        // Max12 scale arg 0 to 1 - int32 1.000 Q15 -> int32 5.028 Q7
        int _SYMBOL_QuantizedConstant84 = (0);
        // Add10 scale arg 1 to 0 - int32 3.000 Q0 -> int32 5.028 Q7
        int _SYMBOL_Constant305 = (77);
        expr_21_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_21_in_0[i0])-(-58))*(gap_min(_SYMBOL_Constant308, gap_max(_SYMBOL_QuantizedConstant84, ((((int)expr_21_in_0[i0])-(-58))+_SYMBOL_Constant305)))*(1)))*(147)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_22(expr_22_args_t *Args) {
    signed char *__restrict__  expr_22_in_0 = Args->expr_22_in_0;
    signed char *__restrict__  expr_22_out_0 = Args->expr_22_out_0; // (16, 20, 200) int8 3.066 Q7
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
        // inputs expr_22_in_0: int8 3.066 Q7
        // expr_22_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_22_in_0, int32), [32]), Mul(GapMin([251], GapMax([0], Add(Sub(Cast(expr_22_in_0, int32), [32]), [126]))), [1])), [198]), [15]), -128, 127), int8)
        // Min13 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.066 Q7
        int _SYMBOL_Constant313 = (251);
        // Max13 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.066 Q7
        int _SYMBOL_QuantizedConstant101 = (0);
        // Add11 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.066 Q7
        int _SYMBOL_Constant310 = (126);
        expr_22_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_22_in_0[i0])-(32))*(gap_min(_SYMBOL_Constant313, gap_max(_SYMBOL_QuantizedConstant101, ((((int)expr_22_in_0[i0])-(32))+_SYMBOL_Constant310)))*(1)))*(198)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_23(expr_23_args_t *Args) {
    signed char *__restrict__  expr_23_in_0 = Args->expr_23_in_0;
    signed char *__restrict__  expr_23_out_0 = Args->expr_23_out_0; // (16, 20, 184) int8 3.627 Q7
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
        // inputs expr_23_in_0: int8 3.627 Q7
        // expr_23_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_23_in_0, int32), [-26]), Mul(GapMin([212], GapMax([0], Add(Sub(Cast(expr_23_in_0, int32), [-26]), [106]))), [1])), [128]), [15]), -128, 127), int8)
        // Min14 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.627 Q7
        int _SYMBOL_Constant318 = (212);
        // Max14 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.627 Q7
        int _SYMBOL_QuantizedConstant118 = (0);
        // Add12 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.627 Q7
        int _SYMBOL_Constant315 = (106);
        expr_23_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_23_in_0[i0])-(-26))*(gap_min(_SYMBOL_Constant318, gap_max(_SYMBOL_QuantizedConstant118, ((((int)expr_23_in_0[i0])-(-26))+_SYMBOL_Constant315)))*(1)))*(128)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_24(expr_24_args_t *Args) {
    signed char *__restrict__  expr_24_in_0 = Args->expr_24_in_0;
    signed char *__restrict__  expr_24_out_0 = Args->expr_24_out_0; // (16, 20, 184) int8 2.978 Q7
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
        // inputs expr_24_in_0: int8 2.978 Q7
        // expr_24_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_24_in_0, int32), [-12]), Mul(GapMin([258], GapMax([0], Add(Sub(Cast(expr_24_in_0, int32), [-12]), [129]))), [1])), [233]), [16]), -128, 127), int8)
        // Min15 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.978 Q7
        int _SYMBOL_Constant323 = (258);
        // Max15 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.978 Q7
        int _SYMBOL_QuantizedConstant135 = (0);
        // Add13 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.978 Q7
        int _SYMBOL_Constant319 = (129);
        expr_24_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_24_in_0[i0])-(-12))*(gap_min(_SYMBOL_Constant323, gap_max(_SYMBOL_QuantizedConstant135, ((((int)expr_24_in_0[i0])-(-12))+_SYMBOL_Constant319)))*(1)))*(233)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_7(expr_7_args_t *Args) {
    signed char *__restrict__  expr_7_in_0 = Args->expr_7_in_0;
    signed char *__restrict__  expr_7_out_0 = Args->expr_7_out_0; // (16, 20, 184) int8 3.087 Q7
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
        // inputs expr_7_in_0: int8 3.087 Q7
        // expr_7_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_7_in_0, int32), [-22]), Mul(GapMin([249], GapMax([0], Add(Sub(Cast(expr_7_in_0, int32), [-22]), [125]))), [1])), [224]), [16]), -128, 127), int8)
        // Min3 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.087 Q7
        int _SYMBOL_Constant328 = (249);
        // Max3 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.087 Q7
        int _SYMBOL_QuantizedConstant152 = (0);
        // Add1 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.087 Q7
        int _SYMBOL_Constant325 = (125);
        expr_7_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_7_in_0[i0])-(-22))*(gap_min(_SYMBOL_Constant328, gap_max(_SYMBOL_QuantizedConstant152, ((((int)expr_7_in_0[i0])-(-22))+_SYMBOL_Constant325)))*(1)))*(224)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_8(expr_8_args_t *Args) {
    signed char *__restrict__  expr_8_in_0 = Args->expr_8_in_0;
    signed char *__restrict__  expr_8_out_0 = Args->expr_8_out_0; // (16, 20, 184) int8 2.633 Q7
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
        // inputs expr_8_in_0: int8 2.633 Q7
        // expr_8_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_8_in_0, int32), [-11]), Mul(GapMin([292], GapMax([0], Add(Sub(Cast(expr_8_in_0, int32), [-11]), [146]))), [1])), [212]), [16]), -128, 127), int8)
        // Min4 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.633 Q7
        int _SYMBOL_Constant333 = (292);
        // Max4 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.633 Q7
        int _SYMBOL_QuantizedConstant169 = (0);
        // Add2 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.633 Q7
        int _SYMBOL_Constant329 = (146);
        expr_8_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_8_in_0[i0])-(-11))*(gap_min(_SYMBOL_Constant333, gap_max(_SYMBOL_QuantizedConstant169, ((((int)expr_8_in_0[i0])-(-11))+_SYMBOL_Constant329)))*(1)))*(212)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_9(expr_9_args_t *Args) {
    signed char *__restrict__  expr_9_in_0 = Args->expr_9_in_0;
    signed char *__restrict__  expr_9_out_0 = Args->expr_9_out_0; // (16, 20, 480) int8 3.631 Q7
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
        // inputs expr_9_in_0: int8 3.631 Q7
        // expr_9_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_9_in_0, int32), [-12]), Mul(GapMin([212], GapMax([0], Add(Sub(Cast(expr_9_in_0, int32), [-12]), [106]))), [1])), [142]), [15]), -128, 127), int8)
        // Min5 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.631 Q7
        int _SYMBOL_Constant338 = (212);
        // Max5 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.631 Q7
        int _SYMBOL_QuantizedConstant186 = (0);
        // Add3 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.631 Q7
        int _SYMBOL_Constant335 = (106);
        expr_9_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_9_in_0[i0])-(-12))*(gap_min(_SYMBOL_Constant338, gap_max(_SYMBOL_QuantizedConstant186, ((((int)expr_9_in_0[i0])-(-12))+_SYMBOL_Constant335)))*(1)))*(142)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_10(expr_10_args_t *Args) {
    signed char *__restrict__  expr_10_in_0 = Args->expr_10_in_0;
    signed char *__restrict__  expr_10_out_0 = Args->expr_10_out_0; // (16, 20, 480) int8 5.221 Q7
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
        // inputs expr_10_in_0: int8 5.221 Q7
        // expr_10_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_10_in_0, int32), [34]), Mul(GapMin([147], GapMax([0], Add(Sub(Cast(expr_10_in_0, int32), [34]), [74]))), [1])), [152]), [14]), -128, 127), int8)
        // Min6 scale arg 0 to 1 - int32 6.000 Q0 -> int32 5.221 Q7
        int _SYMBOL_Constant343 = (147);
        // Max6 scale arg 0 to 1 - int32 1.000 Q15 -> int32 5.221 Q7
        int _SYMBOL_QuantizedConstant203 = (0);
        // Add4 scale arg 1 to 0 - int32 3.000 Q0 -> int32 5.221 Q7
        int _SYMBOL_Constant340 = (74);
        expr_10_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_10_in_0[i0])-(34))*(gap_min(_SYMBOL_Constant343, gap_max(_SYMBOL_QuantizedConstant203, ((((int)expr_10_in_0[i0])-(34))+_SYMBOL_Constant340)))*(1)))*(152)), (14))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_12(expr_12_args_t *Args) {
    signed char *__restrict__  expr_12_in_0 = Args->expr_12_in_0;
    signed char *__restrict__  expr_12_out_0 = Args->expr_12_out_0; // (16, 20, 672) int8 2.524 Q7
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
        // inputs expr_12_in_0: int8 2.524 Q7
        // expr_12_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_12_in_0, int32), [3]), Mul(GapMin([304], GapMax([0], Add(Sub(Cast(expr_12_in_0, int32), [3]), [152]))), [1])), [244]), [16]), -128, 127), int8)
        // Min7 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.524 Q7
        int _SYMBOL_Constant348 = (304);
        // Max7 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.524 Q7
        int _SYMBOL_QuantizedConstant225 = (0);
        // Add5 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.524 Q7
        int _SYMBOL_Constant344 = (152);
        expr_12_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_12_in_0[i0])-(3))*(gap_min(_SYMBOL_Constant348, gap_max(_SYMBOL_QuantizedConstant225, ((((int)expr_12_in_0[i0])-(3))+_SYMBOL_Constant344)))*(1)))*(244)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_13(expr_13_args_t *Args) {
    signed char *__restrict__  expr_13_in_0 = Args->expr_13_in_0;
    signed char *__restrict__  expr_13_out_0 = Args->expr_13_out_0; // (16, 20, 672) int8 3.311 Q7
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
        // inputs expr_13_in_0: int8 3.311 Q7
        // expr_13_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_13_in_0, int32), [-3]), Mul(GapMin([232], GapMax([0], Add(Sub(Cast(expr_13_in_0, int32), [-3]), [116]))), [1])), [138]), [15]), -128, 127), int8)
        // Min8 scale arg 0 to 1 - int32 6.000 Q0 -> int32 3.311 Q7
        int _SYMBOL_Constant353 = (232);
        // Max8 scale arg 0 to 1 - int32 1.000 Q15 -> int32 3.311 Q7
        int _SYMBOL_QuantizedConstant242 = (0);
        // Add6 scale arg 1 to 0 - int32 3.000 Q0 -> int32 3.311 Q7
        int _SYMBOL_Constant350 = (116);
        expr_13_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_13_in_0[i0])-(-3))*(gap_min(_SYMBOL_Constant353, gap_max(_SYMBOL_QuantizedConstant242, ((((int)expr_13_in_0[i0])-(-3))+_SYMBOL_Constant350)))*(1)))*(138)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_15(expr_15_args_t *Args) {
    signed char *__restrict__  expr_15_in_0 = Args->expr_15_in_0;
    signed char *__restrict__  expr_15_out_0 = Args->expr_15_out_0; // (16, 20, 672) int8 2.317 Q7
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
        // inputs expr_15_in_0: int8 2.317 Q7
        // expr_15_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_15_in_0, int32), [-21]), Mul(GapMin([332], GapMax([0], Add(Sub(Cast(expr_15_in_0, int32), [-21]), [166]))), [1])), [179]), [16]), -128, 127), int8)
        // Min9 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.317 Q7
        int _SYMBOL_Constant358 = (332);
        // Max9 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.317 Q7
        int _SYMBOL_QuantizedConstant264 = (0);
        // Add7 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.317 Q7
        int _SYMBOL_Constant354 = (166);
        expr_15_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_15_in_0[i0])-(-21))*(gap_min(_SYMBOL_Constant358, gap_max(_SYMBOL_QuantizedConstant264, ((((int)expr_15_in_0[i0])-(-21))+_SYMBOL_Constant354)))*(1)))*(179)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_25(expr_25_args_t *Args) {
    signed char *__restrict__  expr_25_in_0 = Args->expr_25_in_0;
    signed char *__restrict__  expr_25_out_0 = Args->expr_25_out_0; // (8, 10, 672) int8 2.369 Q7
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
        // inputs expr_25_in_0: int8 2.369 Q7
        // expr_25_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_25_in_0, int32), [-22]), Mul(GapMin([324], GapMax([0], Add(Sub(Cast(expr_25_in_0, int32), [-22]), [162]))), [1])), [180]), [16]), -128, 127), int8)
        // Min16 scale arg 0 to 1 - int32 6.000 Q0 -> int32 2.369 Q7
        int _SYMBOL_Constant363 = (324);
        // Max16 scale arg 0 to 1 - int32 1.000 Q15 -> int32 2.369 Q7
        int _SYMBOL_QuantizedConstant281 = (0);
        // Add14 scale arg 1 to 0 - int32 3.000 Q0 -> int32 2.369 Q7
        int _SYMBOL_Constant359 = (162);
        expr_25_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_25_in_0[i0])-(-22))*(gap_min(_SYMBOL_Constant363, gap_max(_SYMBOL_QuantizedConstant281, ((((int)expr_25_in_0[i0])-(-22))+_SYMBOL_Constant359)))*(1)))*(180)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_27(expr_27_args_t *Args) {
    signed char *__restrict__  expr_27_in_0 = Args->expr_27_in_0;
    signed char *__restrict__  expr_27_out_0 = Args->expr_27_out_0; // (8, 10, 480) int8 1.437 Q7
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
        // inputs expr_27_in_0: int8 1.437 Q7
        // expr_27_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_27_in_0, int32), [15]), Mul(GapMin([536], GapMax([0], Add(Sub(Cast(expr_27_in_0, int32), [15]), [268]))), [1])), [196]), [16]), -128, 127), int8)
        // Min17 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.437 Q7
        int _SYMBOL_Constant369 = (536);
        // Max17 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.437 Q7
        int _SYMBOL_QuantizedConstant304 = (0);
        // Add15 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.437 Q7
        int _SYMBOL_Constant365 = (268);
        expr_27_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_27_in_0[i0])-(15))*(gap_min(_SYMBOL_Constant369, gap_max(_SYMBOL_QuantizedConstant304, ((((int)expr_27_in_0[i0])-(15))+_SYMBOL_Constant365)))*(1)))*(196)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_28(expr_28_args_t *Args) {
    signed char *__restrict__  expr_28_in_0 = Args->expr_28_in_0;
    signed char *__restrict__  expr_28_out_0 = Args->expr_28_out_0; // (8, 10, 480) int8 1.763 Q7
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
        // inputs expr_28_in_0: int8 1.763 Q7
        // expr_28_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_28_in_0, int32), [43]), Mul(GapMin([436], GapMax([0], Add(Sub(Cast(expr_28_in_0, int32), [43]), [218]))), [1])), [165]), [15]), -128, 127), int8)
        // Min18 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.763 Q7
        int _SYMBOL_Constant374 = (436);
        // Max18 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.763 Q7
        int _SYMBOL_QuantizedConstant321 = (0);
        // Add16 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.763 Q7
        int _SYMBOL_Constant370 = (218);
        expr_28_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_28_in_0[i0])-(43))*(gap_min(_SYMBOL_Constant374, gap_max(_SYMBOL_QuantizedConstant321, ((((int)expr_28_in_0[i0])-(43))+_SYMBOL_Constant370)))*(1)))*(165)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_30(expr_30_args_t *Args) {
    signed char *__restrict__  expr_30_in_0 = Args->expr_30_in_0;
    signed char *__restrict__  expr_30_out_0 = Args->expr_30_out_0; // (8, 10, 480) int8 1.580 Q7
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
        // inputs expr_30_in_0: int8 1.580 Q7
        // expr_30_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_30_in_0, int32), [25]), Mul(GapMin([486], GapMax([0], Add(Sub(Cast(expr_30_in_0, int32), [25]), [243]))), [1])), [236]), [16]), -128, 127), int8)
        // Min19 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.580 Q7
        int _SYMBOL_Constant379 = (486);
        // Max19 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.580 Q7
        int _SYMBOL_QuantizedConstant343 = (0);
        // Add17 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.580 Q7
        int _SYMBOL_Constant375 = (243);
        expr_30_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_30_in_0[i0])-(25))*(gap_min(_SYMBOL_Constant379, gap_max(_SYMBOL_QuantizedConstant343, ((((int)expr_30_in_0[i0])-(25))+_SYMBOL_Constant375)))*(1)))*(236)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_31(expr_31_args_t *Args) {
    signed char *__restrict__  expr_31_in_0 = Args->expr_31_in_0;
    signed char *__restrict__  expr_31_out_0 = Args->expr_31_out_0; // (8, 10, 480) int8 1.562 Q7
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
        // inputs expr_31_in_0: int8 1.562 Q7
        // expr_31_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_31_in_0, int32), [44]), Mul(GapMin([492], GapMax([0], Add(Sub(Cast(expr_31_in_0, int32), [44]), [246]))), [1])), [152]), [15]), -128, 127), int8)
        // Min20 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.562 Q7
        int _SYMBOL_Constant384 = (492);
        // Max20 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.562 Q7
        int _SYMBOL_QuantizedConstant360 = (0);
        // Add18 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.562 Q7
        int _SYMBOL_Constant380 = (246);
        expr_31_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_31_in_0[i0])-(44))*(gap_min(_SYMBOL_Constant384, gap_max(_SYMBOL_QuantizedConstant360, ((((int)expr_31_in_0[i0])-(44))+_SYMBOL_Constant380)))*(1)))*(152)), (15))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_33(expr_33_args_t *Args) {
    signed char *__restrict__  expr_33_in_0 = Args->expr_33_in_0;
    signed char *__restrict__  expr_33_out_0 = Args->expr_33_out_0; // (8, 10, 480) int8 1.669 Q7
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
        // inputs expr_33_in_0: int8 1.669 Q7
        // expr_33_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_33_in_0, int32), [-6]), Mul(GapMin([460], GapMax([0], Add(Sub(Cast(expr_33_in_0, int32), [-6]), [230]))), [1])), [173]), [16]), -128, 127), int8)
        // Min21 scale arg 0 to 1 - int32 6.000 Q0 -> int32 1.669 Q7
        int _SYMBOL_Constant389 = (460);
        // Max21 scale arg 0 to 1 - int32 1.000 Q15 -> int32 1.669 Q7
        int _SYMBOL_QuantizedConstant382 = (0);
        // Add19 scale arg 1 to 0 - int32 3.000 Q0 -> int32 1.669 Q7
        int _SYMBOL_Constant385 = (230);
        expr_33_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_33_in_0[i0])-(-6))*(gap_min(_SYMBOL_Constant389, gap_max(_SYMBOL_QuantizedConstant382, ((((int)expr_33_in_0[i0])-(-6))+_SYMBOL_Constant385)))*(1)))*(173)), (16))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2(expr_2_args_t *Args) {
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (16, 20, 24) int8 0.032 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (16, 20, 24) var shapes:
    // expr_2_out_0: (16, 20, 24) expr_2_in_0: (16, 20, 24)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: int8 0.032 Q7
        // expr_2_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [-12]), [1]), [234]), [8]), -128, 127), int8)
        expr_2_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(-12))*(1))*(234)), (8))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_1(expr_2_1_args_t *Args) {
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (8, 10, 24) int8 0.017 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (8, 10, 24) var shapes:
    // expr_2_out_0: (8, 10, 24) expr_2_in_0: (8, 10, 24)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: int8 0.017 Q7
        // expr_2_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [21]), [1]), [245]), [9]), -128, 127), int8)
        expr_2_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(21))*(1))*(245)), (9))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_2(expr_2_2_args_t *Args) {
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (4, 5, 24) int8 0.014 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (4, 5, 24) var shapes:
    // expr_2_out_0: (4, 5, 24) expr_2_in_0: (4, 5, 24)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: int8 0.014 Q7
        // expr_2_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [-23]), [1]), [210]), [9]), -128, 127), int8)
        expr_2_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(-23))*(1))*(210)), (9))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_3(expr_2_3_args_t *Args) {
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (2, 3, 24) int8 0.015 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (2, 3, 24) var shapes:
    // expr_2_out_0: (2, 3, 24) expr_2_in_0: (2, 3, 24)
    // Iteration reduced to spaces ((0, 1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((0, 1, 2),)
    // Paralelized space (0, 1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: int8 0.015 Q7
        // expr_2_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [4]), [1]), [227]), [9]), -128, 127), int8)
        expr_2_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(4))*(1))*(227)), (9))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_4(expr_2_4_args_t *Args) {
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (1, 2, 24) int8 0.007 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 2, 24) var shapes:
    // expr_2_out_0: (1, 2, 24) expr_2_in_0: (1, 2, 24)
    // Iteration reduced to spaces ((1, 2),)
    // Fixed spaces ()
    // Parameteric spaces ((1, 2),)
    // Paralelized space (1, 2)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: int8 0.007 Q7
        // expr_2_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [-14]), [1]), [204]), [10]), -128, 127), int8)
        expr_2_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(-14))*(1))*(204)), (10))), ((7))));
    }
    gap_waitbarrier(0);
}

// Output iteration space reduced to 0 internal and 1 external iteration spaces
void expr_2_5(expr_2_5_args_t *Args) {
    signed char *__restrict__  expr_2_in_0 = Args->expr_2_in_0;
    signed char *__restrict__  expr_2_out_0 = Args->expr_2_out_0; // (1, 1, 24) int8 0.006 Q7
    unsigned int CoreId = gap_coreid();
    unsigned int I0 = Args->W*Args->H*Args->Feat;
    unsigned int Chunk = ChunkSize(I0);
    unsigned int First = Chunk*CoreId;
    unsigned int Last = gap_min(First+Chunk, I0);
    // Max shape: (1, 1, 24) var shapes:
    // expr_2_out_0: (1, 1, 24) expr_2_in_0: (1, 1, 24)
    // Iteration reduced to spaces ((2,),)
    // Fixed spaces ()
    // Parameteric spaces ((2,),)
    // Paralelized space (2,)
    // Interior spaces ()
    for (int i0=First; i0<Last; i0++) {
        // inputs expr_2_in_0: int8 0.006 Q7
        // expr_2_out_0 = Cast(Clip(Norm(Mul(Mul(Sub(Cast(expr_2_in_0, int32), [12]), [1]), [172]), [10]), -128, 127), int8)
        expr_2_out_0[i0] = ((signed char)gap_clip((gap_roundnorm_reg((((((int)expr_2_in_0[i0])-(12))*(1))*(172)), (10))), ((7))));
    }
    gap_waitbarrier(0);
}


#pragma GCC diagnostic pop